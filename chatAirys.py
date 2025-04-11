from pathlib import Path
import sys

import tiktoken
import torch
import chainlit
from airySU.generateSU import generate
from airysModels.airySU import airySU
from airysLib.tokenIO import text_to_token_ids, token_ids_to_text
from airysLib.Config import ConfigSU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():

    # config
    config = ConfigSU(
        vocab_size = 32000, # default, overwritten later
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
    tokenizer = tiktoken.get_encoding("gpt2")
    config.vocab_size = tokenizer.n_vocab # set vocab size to the tokenizer vocab size

    model_path = Path(".") / "models/airySU.pth"

    if not model_path.exists():
        print(f"Could not find the {model_path} file. Please run the chapter 5 code (ch05.ipynb) to generate the model.pth file.")
        sys.exit()

    checkpoint = torch.load(model_path, weights_only=True)
    model = airySU(config)
    model.load_state_dict(checkpoint)
    model.to(device)

    return tokenizer, model, config


# Obtain the necessary tokenizer and model files for the chainlit function below
tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    token_ids = generate(  # function uses `with torch.no_grad()` internally already
        model=model,
        idx=text_to_token_ids(message.content, tokenizer).to(device),  # The user text is provided via as `message.content`
        max_new_tokens=50,
        context_size=model_config.context_length,
        top_k=1,
        temperature=0.0
    )

    text = token_ids_to_text(token_ids, tokenizer)

    await chainlit.Message(
        content=f"{text}",  # This returns the model response to the interface
    ).send()