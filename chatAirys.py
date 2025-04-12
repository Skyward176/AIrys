from pathlib import Path
import sys

import tiktoken
import torch
import chainlit
from airysLlama.loadLlamaFromPretrained import loadLlamaFromPretrained
from airysLlama.LlamaGenerate import generate, text_to_token_ids, token_ids_to_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_response(response_text, input_text):
    return response_text[len(input_text):].replace("### Response:", "").strip()


# Obtain the necessary tokenizer and model files for the chainlit function below
model,_, chat_tokenizer, model_config = loadLlamaFromPretrained("3B")  # Load the model and tokenizer
model.to(device)

@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """

    prompt = f"""Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    {message.content}
    """

    token_ids = generate(  # function uses `with torch.no_grad()` internally already
        model=model,
        idx=text_to_token_ids(prompt, chat_tokenizer).to(device),  # The user text is provided via as `message.content`
        max_new_tokens=2000,
        context_size=model_config.context_length,
        eos_id=50256
    )

    text = token_ids_to_text(token_ids, chat_tokenizer)
    response = extract_response(text, prompt)

    await chainlit.Message(
        content=f"{response}",  # This returns the model response to the interface
    ).send()
