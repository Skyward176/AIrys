import torch
from huggingface_hub import hf_hub_download
from airysLlama import airysLlama
from LlamaTokenizer import Tokenizer, ChatFormat
from airysLib.config import ConfigLlama
from safetensors.torch import load_file
from LlamaGenerate import generate, text_to_token_ids, token_ids_to_text
LLAMA_SIZE_STR = "1B"

def assign(left, right, tensor_name="unknown"):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.clone().detach())
    else:
        return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_llama(model, param_config, params):
    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for l in range(param_config.n_layers):

        # Load attention weights
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        model.trf_blocks[l].att.W_key.weight = assign(
            model.trf_blocks[l].att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        model.trf_blocks[l].att.W_value.weight = assign(
            model.trf_blocks[l].att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )
        model.trf_blocks[l].att.out_proj.weight = assign(
            model.trf_blocks[l].att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Load FeedForward weights
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        model.trf_blocks[l].ff.fc2.weight = assign(
            model.trf_blocks[l].ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        model.trf_blocks[l].ff.fc3.weight = assign(
            model.trf_blocks[l].ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )
        model.trf_blocks[l].norm2.weight = assign(
            model.trf_blocks[l].norm2.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Load output layer weights
    model.final_norm.weight = assign(model.final_norm.weight, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params.keys():
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = assign(model.out_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
        print("Model uses weight tying.")

def test_llama(model, tokenizer, chat_tokenizer, config, device, prompt="What do llamas eat?"):

    torch.manual_seed(123)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, chat_tokenizer).to(device),
        max_new_tokens=150,
        context_size=config.context_length,
        top_k=1,
        temperature=0.
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
    if torch.mps.is_available:
        device = "mps"
    elif torch.cuda.is_available:
        device = "cuda"
    else:
        device = "cpu"

    config = ConfigLlama()
    tokenizer_file_path = hf_hub_download(
        repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
        filename="original/tokenizer.model",
        local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
    )

    tokenizer = Tokenizer(tokenizer_file_path)
    chat_tokenizer = ChatFormat(tokenizer)

    model = airysLlama(config)
    if LLAMA_SIZE_STR == "1B":
        weights_file = hf_hub_download(
            repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
            filename=f"model.safetensors",
            local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct" 
        )
        combined_weights = load_file(weights_file)

    load_weights_into_llama(model, config, combined_weights)
    model.to(device)
    
    del(combined_weights)

    test_llama(model,
                tokenizer=tokenizer,
                chat_tokenizer=chat_tokenizer,
                config = config,
                device = device
            )