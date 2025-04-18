import tkinter as tk
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig, Conv1D, AutoModelForCausalLM
import torch

from peft import LoraConfig, PeftConfig, get_peft_model

if torch.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



repo_id="src/models/airysLlama/airys_llama_character_8B"
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    device_map="auto",
    quantization_config=nf4_config,
)
def get_specific_layer_names(model):
    # Create a list to store the layer names
    layer_names = []
    
    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of the specified layers
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
            # model name parsing 

            layer_names.append('.'.join(name.split('.')[4:]).split('.')[0])
    
    return layer_names
loraConfig = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=get_specific_layer_names(model),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, loraConfig)
tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

# Create the main window
root = tk.Tk()
root.title("Basic Tkinter Window")

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

# Create a Submit button
submit_button = tk.Button(root, text="Submit", command=submit_input)
submit_button.pack(pady=10)

# Run the applicLation
root.mainloop()