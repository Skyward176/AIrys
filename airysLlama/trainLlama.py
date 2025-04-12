import torch
from huggingface_hub import hf_hub_download
from airysLlama.airysLlama import airysLlama
from airysLlama.LlamaTokenizer import Tokenizer, ChatFormat
from airysLib.config import ConfigLlama
from safetensors.torch import load_file
from airysLlama.LlamaGenerate import generate, text_to_token_ids, token_ids_to_text
from airysLlama.loadLlamaFromPretrained import loadLlamaFromPretrained
LLAMA_SIZE_STR = "1B"


def test_llama(model, tokenizer, chat_tokenizer, config, device, prompt="Airys,What do llamas eat?"):

    torch.manual_seed(123)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, chat_tokenizer).to(device),
        max_new_tokens=1000,
        context_size=config.context_length,
        top_k=1,
        temperature=0.,
        eos_id=chat_tokenizer.tokenizer.special_tokens["<|eot_id|>"]
    )

    output_text = token_ids_to_text(token_ids, tokenizer)
    print(output_text)

    def clean_text(text, header_end="assistant<|end_header_id|>\n\n"):
        # Find the index of the first occurrence of "<|end_header_id|>"
        index = text.find(header_end)

        if index != -1:
            # Return the substring starting after "<|end_header_id|>"
            return text[index + len(header_end):].strip()  # Strip removes leading/trailing whitespace
        else:
            # If the token is not found, return the original text
            return text

    print("Output text:\n", clean_text(output_text))
def main():
    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    model, tokenizer, chat_tokenizer, config = loadLlamaFromPretrained("1B") 
    model.to(device)
    test_llama(model,
                tokenizer=tokenizer,
                chat_tokenizer=chat_tokenizer,
                config = config,
                device = device
            )
if __name__ == "__main__":
    main()