import numpy as np
import torch 
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def airysGPTweightLoader(Airys, params):
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