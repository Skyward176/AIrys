from airySU.generateSU import generate
import tiktoken
from airysModels.airySU import airySU
from airysLib.Config import ConfigSU
import torch
import sys
from pathlib import Path

if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon Metal Performance Shaders
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

config = ConfigSU(
    vocab_size = 32000, #default, overridden later
    emb_dim = 1024,
    n_layers = 32,
    n_heads = 20,
    latent_qkv_dim = 128,
    d_rope = 16,
    context_length = 2048,
    batch_size = 1,
    hidden_dim = 2048,
    dtype= torch.bfloat16,  # Use float16 for training
)

model_path = Path(".") / "models/airySU.pth"

if not model_path.exists():
    print(f"Could not find the {model_path} file. Please run the chapter 5 code (ch05.ipynb) to generate the model.pth file.")
    sys.exit()

checkpoint = torch.load(model_path, weights_only=True)
tokenizer = tiktoken.get_encoding("gpt2")
config.vocab_size = tokenizer.n_vocab # set vocab size to the tokenizer vocab size
model = airySU(config)
model.load_state_dict(checkpoint)
model.to(device)

input = text_to_token_ids("Michael Jordan", tokenizer).to(device)
output = generate(model, input, 50,config.context_length, temperature=1.5, top_k = 2, eos_id = None)
print(token_ids_to_text(output,tokenizer))