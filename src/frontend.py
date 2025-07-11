import tkinter as tk
from transformers import pipeline
import torch
from loadAirys import loadAirys
import torchaudio
import pyaudio
import numpy as np
from huggingface_hub import snapshot_download
from tkinter import ttk # Using themed widgets for potentially better look
from tkinter import scrolledtext # For scrollable text area

from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
import time
import torch
import torchaudio
from gradio_client import Client, handle_file


if torch.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")


# Load LLM
repo_id = "google/gemma-3-27b-it"
model, tokenizer = loadAirys(repo_id = repo_id)
#model.to(device)

client = Client("http://localhost:7860")


pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# --- Configuration ---
BG_COLOR = '#212121'       # Main background
FG_COLOR = '#FFFFFF'       # Main text color
SIDEBAR_BG = '#333333'    # Sidebar background
ENTRY_BG = '#424242'       # Input field background
BUTTON_BG = '#555555'    # Button background
BUTTON_FG = '#FFFFFF'    # Button text color
LISTBOX_SELECT_BG = '#555555' # Listbox selection background

# Conversation Thread Specific Colors (Optional)
USER_MSG_BG = '#2c3e50'  # Slightly different background for user messages
LLM_MSG_BG = '#34495e'   # Slightly different background for LLM messages
USER_MSG_FG = '#ecf0f1'
LLM_MSG_FG = '#ecf0f1'

# --- Functions ---
def send_message(event=None): # event=None allows binding to button click and Enter key
    """Handles sending the message and displaying the conversation."""
    message = input_entry.get()
    if message.strip(): # Don't send empty messages
        # --- Display User Message ---
        chat_display.config(state=tk.NORMAL) # Enable editing

        # Insert user message with 'user' tag
        chat_display.insert(tk.END, "You:\n", ("label", "user_label")) # Label
        chat_display.insert(tk.END, message + "\n\n", ("user",)) # Message content with tag

        chat_display.config(state=tk.DISABLED) # Disable editing
        chat_display.see(tk.END) # Scroll to the bottom

        # Clear the input field
        input_entry.delete(0, tk.END)

        # --- Placeholder for LLM interaction & Response ---
        # In a real app, you would call your LLM here
        # and then display the response using the 'llm' tag.
        get_llm_response(message) # Call the simulation function

    return "break" # Prevents default tkinter behavior (like adding newline on Enter)

def get_llm_response(message):
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
    start_time = time.time()  # Start timing

    out = pipe(messages, max_new_tokens=2048)

    end_time = time.time()  # End timing
    print(f"LLM inference execution time: {end_time - start_time:.2f} seconds")

    llm_response = out[0]["generated_text"][-1]["content"]
    print(llm_response) # Debug print

    chat_display.config(state=tk.NORMAL) # Enable editing

    # Insert LLM response with 'llm' tag
    chat_display.insert(tk.END, "AIrys:\n", ("label", "llm_label")) # Label
    chat_display.insert(tk.END, llm_response + "\n\n", ("AIrys",)) # Message content with tag

    chat_display.config(state=tk.DISABLED) # Disable editing
    chat_display.see(tk.END) # Scroll to the bottom
    run_audio(client, llm_response) # Call the audio function
    # generate speech by cloning a voice using default settings
def run_audio(client,text):

    start_time = time.time()  # Start timing

    result = client.predict(
            model_choice="Zyphra/Zonos-v0.1-transformer",
            text=text,
            language="en-us",
            speaker_audio=handle_file("./bocchi.wav"),
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

    end_time = time.time()  # End timing
    print(f"TTS execution time: {end_time - start_time:.2f} seconds")

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
                    rate=24000,  # Adjust the rate to match the slowdown
                    output=True)

    # Play the audio
    stream.write(audio_np.astype(np.float32).tobytes())

    # Close the stream
    stream.stop_stream()



def toggle_sound():
    """Handles the sound toggle button click."""
    if sound_enabled.get():
        print("Sound Generation: ON")
        sound_toggle_button.config(text="Sound ON")
        # Add logic here to enable sound generation
    else:
        print("Sound Generation: OFF")
        sound_toggle_button.config(text="Sound OFF")
        # Add logic here to disable sound generation

# --- Main Window Setup ---
root = tk.Tk()
root.title("AIrys Chat <3")
root.geometry("900x700") # Adjusted size
root.configure(bg=BG_COLOR)

# --- Style Configuration (Optional, for ttk widgets) ---
style = ttk.Style()
try:
    style.theme_use('clam') # 'clam' often looks better on Linux/Mac
except tk.TclError:
    print("Clam theme not available, using default.")

style.configure("TFrame", background=BG_COLOR)
style.configure("TLabel", background=BG_COLOR, foreground=FG_COLOR)
style.configure("TButton", background=BUTTON_BG, foreground=BUTTON_FG)
style.map("TButton", background=[('active', '#666666')]) # Button hover color
style.configure("Horizontal.TScrollbar", background=BUTTON_BG, troughcolor=BG_COLOR)
style.configure("Vertical.TScrollbar", background=BUTTON_BG, troughcolor=BG_COLOR)

# --- Main Layout Frames ---
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1, minsize=200) # Sidebar column
root.grid_columnconfigure(1, weight=4)             # Main content column

# Sidebar Frame (Left)
sidebar_frame = tk.Frame(root, width=250, bg=SIDEBAR_BG)
sidebar_frame.grid(row=0, column=0, sticky="nsew")
sidebar_frame.pack_propagate(False)

# Main Content Frame (Right)
main_frame = tk.Frame(root, bg=BG_COLOR)
main_frame.grid(row=0, column=1, sticky="nsew")
main_frame.grid_rowconfigure(0, weight=1) # Chat display row
main_frame.grid_rowconfigure(1, weight=0) # Input area row
main_frame.grid_columnconfigure(0, weight=1)

# --- Sidebar Widgets ---
logo_label = tk.Label(sidebar_frame, text="AIrys Chat <3", bg=SIDEBAR_BG, fg=FG_COLOR, font=("Arial", 14, "bold"))
logo_label.pack(pady=10, padx=10, anchor='w')
history_label = tk.Label(sidebar_frame, text="History", bg=SIDEBAR_BG, fg='#AAAAAA', font=("Arial", 10))
history_label.pack(pady=(10, 2), padx=10, anchor='w')
history_listbox = tk.Listbox(
    sidebar_frame, bg=SIDEBAR_BG, fg=FG_COLOR, border=0, highlightthickness=0,
    selectbackground=LISTBOX_SELECT_BG, activestyle='none'
)
history_listbox.pack(expand=True, fill="both", padx=10, pady=(0, 10))
for i in range(15): history_listbox.insert(tk.END, f"Chat Session {i+1}")
history_scrollbar = ttk.Scrollbar(history_listbox, orient="vertical", command=history_listbox.yview)
history_scrollbar.pack(side="right", fill="y")
history_listbox.config(yscrollcommand=history_scrollbar.set)

# --- Main Content Widgets ---

# Chat Display Area
chat_display = scrolledtext.ScrolledText(
    main_frame, wrap=tk.WORD, bg=BG_COLOR, fg=FG_COLOR, bd=0,
    highlightthickness=0, state=tk.DISABLED, font=("Arial", 11), padx=10, pady=10
)
chat_display.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 0))

# --- Configure Text Tags for Conversation Styling ---
# Common label style
chat_display.tag_configure("label", font=("Arial", 10, "italic"), foreground="#AAAAAA")
# User message style
chat_display.tag_configure("user", justify='right', background=USER_MSG_BG, foreground=USER_MSG_FG, rmargin=10, spacing1=2, spacing3=10) # Add right margin (rmargin)
chat_display.tag_configure("user_label", justify='right', rmargin=10)
# LLM message style
chat_display.tag_configure("AIrys <3", justify='left', background=LLM_MSG_BG, foreground=LLM_MSG_FG, lmargin1=10, lmargin2=10, spacing1=2, spacing3=10) # Add left margins (lmargin1, lmargin2)
chat_display.tag_configure("llm_label", justify='left', lmargin1=10, lmargin2=10)

# Initial message
chat_display.config(state=tk.NORMAL)
chat_display.insert(tk.END, "System:\n", ("label", "llm_label"))
chat_display.insert(tk.END, "Start Typing to AIrys below: \n\n", ("AIrys",))
chat_display.config(state=tk.DISABLED)

# Input Area Frame (Bottom)
input_frame = tk.Frame(main_frame, bg=BG_COLOR)
input_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)

# Input Entry Field
input_entry = tk.Entry(
    input_frame, bg=ENTRY_BG, fg=FG_COLOR, insertbackground=FG_COLOR, font=("Arial", 11),
    bd=0, highlightthickness=1, highlightbackground=SIDEBAR_BG, highlightcolor=BUTTON_BG
)
input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))
input_entry.bind("<Return>", send_message)

# Send Button
send_button = tk.Button(
    input_frame, text="Send", bg=BUTTON_BG, fg=BUTTON_FG, activebackground='#666666',
    activeforeground=FG_COLOR, command=send_message, bd=0, padx=10, pady=5
)
send_button.pack(side=tk.LEFT)

# Sound Toggle Button
sound_enabled = tk.BooleanVar(value=False)
sound_toggle_button = tk.Checkbutton(
    input_frame, text="Sound OFF", variable=sound_enabled, command=toggle_sound,
    indicatoron=False, bg=BUTTON_BG, fg=BUTTON_FG, selectcolor=ENTRY_BG,
    activebackground='#666666', activeforeground=FG_COLOR, bd=0,
    padx=10, pady=5, relief=tk.FLAT
)
sound_toggle_button.pack(side=tk.RIGHT, padx=(10, 0))

# --- Start GUI ---
root.mainloop()