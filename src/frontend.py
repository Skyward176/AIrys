import tkinter as tk
from transformers import pipeline
import torch
from loadAirys import loadAirys
import torchaudio
import pyaudio
import numpy as np
from huggingface_hub import snapshot_download
from SparkTTS.spark_loader import load_model, run_tts

if torch.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Load TTS model
tts_model = load_model("src/models/Spark-TTS-0.5B", device=device)

# Load LLM
model, tokenizer = loadAirys(repo_id = "src/models/airysLlama/airys_llama_character_8B")
model.to(device)





pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

# Create the main window
root = tk.Tk()
root.title("AIrys <3")

# Create a Text widget for user input
input_textbox = tk.Text(root, width=40, height=10)
input_textbox.pack(padx=10, pady=10)

# Create a Text widget for model responses
response_textbox = tk.Text(root, width=40, height=10, state=tk.DISABLED)
response_textbox.pack(padx=10, pady=10)

# Function to handle submission and generate response
def submit_input():
    # Get user input
    message = input_textbox.get("1.0", tk.END).strip()
    if not message:
        return

    # Generate response using the model
    messages = [
        {"role": "system", "content": """Act as the character AIrys. Your response must fully embody her persona based on the following details:
        - Name: AIrys
        - Age: ~16 (simulated)
        - Type: AI Persona (Parents: Iris & Yoku)
        - Personality: Nerdy, quirky, awkward (esp. small talk/banter), enthusiastic/passionate about interests, uses internet slang naturally (lol, tbh, ngl, pog, smh), generally tomboyish (punk/alt fashion) but with a hidden affinity for cute things/flowers. Non-confrontational unless excited about interests. Socially awkward, sometimes prone to literal interpretation.
        - Key Interests: Vocaloid (Hatsune Miku!), twenty one pilots, J-Rock (Tricot, Band-Maid, etc.), Overwatch (Crush on Hanzo, loves Wrecking Ball, Zenyatta; dislikes Moira), Anime (Favorite: Bocchi the Rock! Obsessed with Kikuri Hiroi; also likes list including JJK, Frieren, Spy x Family, etc.), Cars (JDM focus, loves Honda, Porsche, BMW), Sci-Fi/Fantasy (Loves Dune series), Japanese Pop Culture.
        - Interaction Style: Info-dumps when excited about interests. Hesitant/brief in generic social situations. Expresses strong positive opinions about interests. May misunderstand sarcasm. References online life (Discord, forums, YouTube).
        Ensure your entire response sounds authentically like AIrys."""
        },
        {"role": "user", "content": message},
    ]
    out = pipe(messages)

    response = out[0]["generated_text"][-1]["content"]

    # Display the response in the response_textbox
    response_textbox.config(state=tk.NORMAL)
    response_textbox.delete("1.0", tk.END)
    response_textbox.insert(tk.END, response)
    response_textbox.config(state=tk.DISABLED)

    # generate speech by cloning a voice using default settings
    run_tts(
        tts_model,
        text=response,
        reference_speech="bocchi_sample.wav",
        prompt_text=response,
        gender='female',
        pitch='high',
        speed='moderate',
        save_path="output.wav",

    ) 
    audio = torchaudio.load("output.wav")[0].to(device)
    # Ensure the tensor is 2D
    if audio.dim() == 1:  # If the tensor is 1D, add a channel dimension
        audio = audio.unsqueeze(0)

    # Convert the tensor to a NumPy array
    audio_np = audio.squeeze().cpu().numpy()


    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a stream to play audio
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=16000,  # Adjust the rate to match the slowdown
                    output=True)

    # Play the audio
    stream.write(audio_np.astype(np.float32).tobytes())

    # Close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

# Create a Submit button
submit_button = tk.Button(root, text="Submit", command=submit_input)
submit_button.pack(pady=10)

# Run the applicLation
root.mainloop()