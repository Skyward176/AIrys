import os
import gradio as gr
from gradio_client import Client, handle_file
import time
import random
import torch
import torchaudio
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import numpy as np
from kokoro import KPipeline
import queue
import pyaudio

# Constants for buffering behavior
BUFFER_FILL_SECONDS = .5         # Initial buffer fill duration (seconds)
BUFFER_REFILL_THRESHOLD = 0.8     # When buffer occupancy falls below this fraction of fill duration, trigger refill

# Device selection
def select_device():
    if torch.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = select_device()

# Text pipeline and model setup
text_pipeline = KPipeline(lang_code='a')
model_path = "src/models/airysLlama/airys_llama_character_8B"
print(f"Loading model: {model_path}...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("Model loaded successfully.")

# Audio settings
AUDIO_TOKEN_INTERVAL = 100
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SIZE = 512   # buffer size for streaming
FORMAT = pyaudio.paInt16

# Shared state
user_volume = 1.0
user_mute = False
user_speed = 1.0
user_token_interval = AUDIO_TOKEN_INTERVAL
latest_waveform = None
stop_generation_flag = threading.Event()

class AudioStreamer:
    def __init__(self):
        self.q = queue.Queue()  # buffer queue for audio data
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        )
        self.running = True
        self.thread = threading.Thread(target=self._play_loop, daemon=True)
        self.thread.start()

    def _play_loop(self):
        while self.running:
            data = self.q.get()
            if data is None:
                break
            if stop_generation_flag.is_set():
                continue
            self.stream.write(data)

    def stop(self):
        self.running = False
        self.q.put(None)
        self.thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def clear_queue(self):
        with self.q.mutex:
            self.q.queue.clear()

    def buffer_duration(self) -> float:
        """Estimate total buffered audio duration in seconds."""
        n_chunks = self.q.qsize()
        return (n_chunks * CHUNK_SIZE) / SAMPLE_RATE

    def push_audio(self, audio_array):
        global latest_waveform
        if user_mute or stop_generation_flag.is_set():
            return

        # Adjust volume
        adjusted = np.clip(audio_array * user_volume, -1.0, 1.0)
        if len(adjusted) < 2 or user_speed <= 0:
            return

        # Resample for playback speed
        adjusted = np.interp(
            np.linspace(0, len(adjusted), max(2, int(len(adjusted) / user_speed))),
            np.arange(len(adjusted)),
            adjusted
        )

        latest_waveform = adjusted
        int_audio = np.int16(adjusted * 32767)

        # Enqueue audio data
        self.q.put(int_audio.tobytes())

# Initialize shared audio streamer
shared_audio_streamer = AudioStreamer()

# Audio generation
def generate_and_stream_audio(text, pipeline, audio_streamer):
    for i, (_, _, audio) in enumerate(pipeline(text, voice='af_bella')):
        if stop_generation_flag.is_set():
            break
        print(
            f"[Audio Generated] Segment {i}:"
            f" shape={audio.shape}, min={audio.min()}, max={audio.max()}, mean={audio.mean()}"
        )
        audio_streamer.push_audio(audio)

# LLM streaming with audio
def transformers_streaming_llm(
    message: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    stop_generation_flag.clear()
    shared_audio_streamer.clear_queue()

    messages = [
        {"role": "system", "content": "(AIrys prompt...)"},
        {"role": "user", "content": message},
    ]
    prompt_formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([prompt_formatted], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    gen_thread = threading.Thread(
        target=model.generate,
        kwargs=generation_kwargs,
        daemon=True
    )
    gen_thread.start()

    cumulative_response = ""
    buffer = ""
    first_chunk_played = threading.Event()

    def audio_worker(text_chunk):
        generate_and_stream_audio(
            text_chunk,
            text_pipeline,
            shared_audio_streamer
        )
        first_chunk_played.set()

    try:
        for new_text in streamer:
            if stop_generation_flag.is_set():
                break
            buffer += new_text
            cumulative_response += new_text
            interval = user_token_interval

            # Launch initial chunk after buffer interval
            if not first_chunk_played.is_set() and len(buffer) >= interval:
                threading.Thread(
                    target=audio_worker,
                    args=(buffer,),
                    daemon=True
                ).start()
                buffer = ""
                time.sleep(BUFFER_FILL_SECONDS)
                first_chunk_played.wait()
                yield cumulative_response

            # Subsequent chunks: wait until buffer has room below threshold
            elif first_chunk_played.is_set() and len(buffer) >= interval:
                while shared_audio_streamer.buffer_duration() > (BUFFER_FILL_SECONDS * BUFFER_REFILL_THRESHOLD):
                    time.sleep(0.05)
                threading.Thread(
                    target=audio_worker,
                    args=(buffer,),
                    daemon=True
                ).start()
                buffer = ""

            yield cumulative_response

    except Exception as e:
        yield cumulative_response + f"\n\n[Error during generation: {e}]"

    finally:
        if buffer:
            print(f"[Final Buffer Audio Triggered]: '{buffer[:50]}...'" )
            t = threading.Thread(
                target=audio_worker,
                args=(buffer,),
                daemon=True
            )
            t.start()
            t.join()
        if gen_thread.is_alive():
            gen_thread.join(timeout=1)

# Utility functions
def test_tone():
    duration = 1.0  # seconds
    freq = 440.0    # Hz (A4)
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    shared_audio_streamer.push_audio(tone)

def stop_generation():
    stop_generation_flag.set()
    shared_audio_streamer.clear_queue()

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Chat AIrys <3")

    with gr.Row():
        prompt_input = gr.Textbox(
            label="Your Prompt",
            placeholder="Type your message here..."
        )
        submit_button = gr.Button("Send")
        stop_button = gr.Button("Stop")
        test_tone_button = gr.Button("Play Test Tone")

    output_display = gr.Textbox(
        label="LLM Response",
        interactive=False
    )

    with gr.Row():
        volume_slider = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            step=0.05,
            value=1.0,
            label="Volume"
        )
        mute_toggle = gr.Checkbox(label="Mute Audio")
        speed_slider = gr.Slider(
            minimum=0.5,
            maximum=2.0,
            step=0.05,
            value=1.0,
            label="Playback Speed"
        )

    def update_audio_controls(volume, mute, speed):
        global user_volume, user_mute, user_speed, user_token_interval
        user_volume = volume
        user_mute = mute
        user_speed = speed

    volume_slider.change(
        fn=update_audio_controls,
        inputs=[volume_slider, mute_toggle, speed_slider],
        outputs=[]
    )
    mute_toggle.change(
        fn=update_audio_controls,
        inputs=[volume_slider, mute_toggle, speed_slider],
        outputs=[]
    )
    speed_slider.change(
        fn=update_audio_controls,
        inputs=[volume_slider, mute_toggle, speed_slider],
        outputs=[]
    )

    submit_button.click(
        fn=transformers_streaming_llm,
        inputs=prompt_input,
        outputs=output_display
    )
    submit_button.click(
        lambda: "",
        inputs=[],
        outputs=prompt_input
    )
    stop_button.click(
        fn=stop_generation,
        inputs=[],
        outputs=[]
    )
    test_tone_button.click(
        fn=test_tone,
        inputs=[],
        outputs=[]
    )

    demo.load(
        fn=update_audio_controls,
        inputs=[volume_slider, mute_toggle, speed_slider],
        outputs=[]
    )

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7870)