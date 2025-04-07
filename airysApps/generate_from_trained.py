import matplotlib.pyplot as plt
import torch
from airysLib.gpt2_weights_downloader import download_and_load_gpt2
import numpy as np
from ..airysModels import airysGPT2
from ..airysLib.AirysLoader import create_dataloader as rsLoader
from airysApps.AirysGen import generate
import tiktoken

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

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_Airys(Airys, params):
    Airys.pos_emb.weight = assign(Airys.pos_emb.weight, params['wpe'])
    Airys.tok_emb.weight = assign(Airys.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        Airys.trf_blocks[b].att.W_query.weight = assign(
            Airys.trf_blocks[b].att.W_query.weight, q_w.T)
        Airys.trf_blocks[b].att.W_key.weight = assign(
            Airys.trf_blocks[b].att.W_key.weight, k_w.T)
        Airys.trf_blocks[b].att.W_value.weight = assign(
            Airys.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        Airys.trf_blocks[b].att.W_query.bias = assign(
            Airys.trf_blocks[b].att.W_query.bias, q_b)
        Airys.trf_blocks[b].att.W_key.bias = assign(
            Airys.trf_blocks[b].att.W_key.bias, k_b)
        Airys.trf_blocks[b].att.W_value.bias = assign(
            Airys.trf_blocks[b].att.W_value.bias, v_b)

        Airys.trf_blocks[b].att.out_proj.weight = assign(
            Airys.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        Airys.trf_blocks[b].att.out_proj.bias = assign(
            Airys.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        Airys.trf_blocks[b].ff.layers[0].weight = assign(
            Airys.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        Airys.trf_blocks[b].ff.layers[0].bias = assign(
            Airys.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        Airys.trf_blocks[b].ff.layers[2].weight = assign(
            Airys.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        Airys.trf_blocks[b].ff.layers[2].bias = assign(
            Airys.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        Airys.trf_blocks[b].norm1.scale = assign(
            Airys.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        Airys.trf_blocks[b].norm1.shift = assign(
            Airys.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        Airys.trf_blocks[b].norm2.scale = assign(
            Airys.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        Airys.trf_blocks[b].norm2.shift = assign(
            Airys.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    Airys.final_norm.scale = assign(Airys.final_norm.scale, params["g"])
    Airys.final_norm.shift = assign(Airys.final_norm.shift, params["b"])
    Airys.out_head.weight = assign(Airys.out_head.weight, params["wte"])


def main(gpt_config, model_size):
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon Metal Performance Shaders
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("GPT Config:", gpt_config)
    print("Model Size:", model_size)

    settings,params = download_and_load_gpt2(model_size=model_size, models_dir="gpt-2")

    print("Keys in params:", params.keys())
    for key, value in params.items():
        print(f"{key}: {np.array(value).shape if isinstance(value, np.ndarray) else type(value)}")

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)

    model = airysGPT2(gpt_config)
    load_weights_into_Airys(model, params)
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