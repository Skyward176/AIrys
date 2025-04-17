import torch
from airysLib.config import ConfigLlama
from airysLlama.airysLlama import airysLlama
from huggingface_hub import hf_hub_download
#from airysLlama.LlamaTokenizer import Tokenizer, ChatFormat
from transformers import AutoTokenizer
from safetensors.torch import load_file

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
def loadLlamaFromPretrained(LLAMA_SIZE_STR, dtype=torch.bfloat16):
    config = ConfigLlama(
        dtype=dtype
    )
    if LLAMA_SIZE_STR == "3B":
        config.vocab_size = 128_256
        config.context_length = 131_072
        config.emb_dim = 3072
        config.n_heads = 24
        config.n_layers = 28
        config.hidden_dim = 8192
        config.n_kv_groups = 8
        config.rope_base = 500_000.0
        config.rope_freq = {
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192
        }

    repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
    tokenizer_file_path = hf_hub_download(
        repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
        filename="original/tokenizer.model",
        local_dir=f"models/Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
    )

    # tokenizer = Tokenizer(tokenizer_file_path)
    # chat_tokenizer = ChatFormat(tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    def rescale_theta(theta_old, context_length_old, context_length_new):
        scaling_factor = context_length_new / context_length_old
        print(f"Rescaling theta from {context_length_old} to {context_length_new} with scaling factor {scaling_factor}")
        print(f"Old theta: {theta_old}")
        theta_new = theta_old * scaling_factor
        return theta_new
    config.rope_base = rescale_theta(config.rope_base, config.context_length, 16384)
    config.context_length = 16384
    model = airysLlama(config)
    print("Requesting model weights from HuggingFace Hub...")
    #if LLAMA_SIZE_STR == "1B":
    #    weights_file = hf_hub_download(
    #        repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
    #        filename=f"model.safetensors",
    #        local_dir=f"models/Llama-3.2-{LLAMA_SIZE_STR}-Instruct" 
    #    )
    #    combined_weights = load_file(weights_file)
    #else:
    #    combined_weights = {}
    #    for i in range(1, 3):
    #        weights_file = hf_hub_download(
    #            repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
    #            filename=f"model-0000{i}-of-00002.safetensors",
    #            local_dir=f"models/Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
    #        )
    #        current_weights = load_file(weights_file)
    #        combined_weights.update(current_weights)
    combined_weights = load_file(
        "models/airysLlama/airys_llama_character/model.safetensors"
    )

    load_weights_into_llama(model, config, combined_weights)
    
    del(combined_weights)
    print("Model loaded into device")
    return model, tokenizer, config