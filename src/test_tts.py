import torch, torchaudio
import numpy as np
import time
from gradio_client import Client, handle_file
import pyaudio

if torch.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
client = Client("http://localhost:7860")
result = client.predict(
		model_choice="Zyphra/Zonos-v0.1-transformer",
		text="Zonos uses eSpeak for text to phoneme conversion!",
		language="en-us",
		speaker_audio=handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav'),
		prefix_audio=handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav'),
		e1=1,
		e2=0.05,
		e3=0.05,
		e4=0.05,
		e5=0.05,
		e6=0.05,
		e7=0.1,
		e8=0.2,
		vq_single=0.78,
		fmax=24000,
		pitch_std=45,
		speaking_rate=15,
		dnsmos_ovrl=4,
		speaker_noised=False,
		cfg_scale=2,
		top_p=0,
		top_k=0,
		min_p=0,
		linear=0.5,
		confidence=0.4,
		quadratic=0,
		seed=420,
		randomize_seed=True,
		unconditional_keys=["emotion"],
		api_name="/generate_audio"
)
print(result)
# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a stream to play audio
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=44000,  # Adjust the rate to match the slowdown
                output=True)

# Play the audio
audio = torchaudio.load(result[0])[0].to(device)
# Ensure the tensor is 2D
if audio.dim() == 1:  # If the tensor is 1D, add a channel dimension
    audio = audio.unsqueeze(0)

# Convert the tensor to a NumPy array
audio_np = audio.squeeze().cpu().numpy()
stream.write(audio_np.astype(np.float32).tobytes())

# Close the stream
stream.stop_stream()
stream.close()
p.terminate()