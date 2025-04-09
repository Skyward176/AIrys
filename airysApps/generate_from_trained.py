import matplotlib.pyplot as plt
import torch
from airysLib.gpt2_weights_downloader import download_and_load_gpt2
import numpy as np
from ..airysModels import airysGPT2Flash
from ..airysLib.AirysLoader import create_dataloader as rsLoader
from airysApps.AirysGen import generate
import tiktoken
from ..airysLib.airysGPTweightLoader import airysGPTweightLoader

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=25,
            context_size=context_size,
            top_k=50,
            temperature=1.5
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

def main(gpt_config, model_size):
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon Metal Performance Shaders
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("GPT Config:", gpt_config)
    print("Model Size:", model_size)

    settings,params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    print("Keys in params:", params.keys())
    for key, value in params.items():
        print(f"{key}: {np.array(value).shape if isinstance(value, np.ndarray) else type(value)}")

    tokenizer = tiktoken.get_encoding("gpt2")

    model = airysGPT2Flash(gpt_config)
    airysGPTweightLoader(model, params)
    model.to(device)
    model.eval()

    #for i in range(1,10):
        #prompt = input("Write a prompt:\n")
    prompt = "Every effort moves you"
    generate_and_print_sample(
        model,
        tokenizer,
        device,
        prompt
    )


if __name__ == "__main__":
    torch.manual_seed(123)

    BASE_CONFIG = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 1024,  # Shortened context length (orig: 1025)
        "drop_rate": 0.0,       # Dropout rate
        "qkv_bias": True       # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-small (124M)"  # Choose the model you want to use
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    main(BASE_CONFIG, model_size)