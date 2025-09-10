import torch
import numpy as np

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    right_tensor = torch.tensor(right, dtype=left.dtype, device=left.device)
    left.data.copy_(right_tensor)
    return left

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.token_emb.weight = assign(gpt.token_emb.weight, params['wte'])


    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
        gpt.trf_blocks[b].attn.W_query.weight = assign(
            gpt.trf_blocks[b].attn.W_query.weight, q_w.T)
        gpt.trf_blocks[b].attn.W_key.weight = assign(
            gpt.trf_blocks[b].attn.W_key.weight, k_w.T)
        gpt.trf_blocks[b].attn.W_value.weight = assign(
            gpt.trf_blocks[b].attn.W_value.weight, v_w.T)


        q_b, k_b, v_b = np.split(
            params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)
        gpt.trf_blocks[b].attn.W_query.bias = assign(
            gpt.trf_blocks[b].attn.W_query.bias, q_b)
        gpt.trf_blocks[b].attn.W_key.bias = assign(
            gpt.trf_blocks[b].attn.W_key.bias, k_b)
        gpt.trf_blocks[b].attn.W_value.bias = assign(
            gpt.trf_blocks[b].attn.W_value.bias, v_b)


        gpt.trf_blocks[b].attn.out_proj.weight = assign(
            gpt.trf_blocks[b].attn.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].attn.out_proj.bias = assign(
            gpt.trf_blocks[b].attn.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])


        gpt.trf_blocks[b].feedforw.layers[0].weight = assign(
            gpt.trf_blocks[b].feedforw.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].feedforw.layers[0].bias = assign(
            gpt.trf_blocks[b].feedforw.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].feedforw.layers[2].weight = assign(
            gpt.trf_blocks[b].feedforw.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].feedforw.layers[2].bias = assign(
            gpt.trf_blocks[b].feedforw.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])


        gpt.trf_blocks[b].Layernorm1.scale_params = assign(
            gpt.trf_blocks[b].Layernorm1.scale_params,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].Layernorm1.shift_params = assign(
            gpt.trf_blocks[b].Layernorm1.shift_params,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].Layernorm2.scale_params = assign(
            gpt.trf_blocks[b].Layernorm2.scale_params,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].Layernorm2.shift_params = assign(
            gpt.trf_blocks[b].Layernorm2.shift_params,
            params["blocks"][b]["ln_2"]["b"])


    gpt.final_norm.scale_params = assign(gpt.final_norm.scale_params, params["g"])
    gpt.final_norm.shift_params = assign(gpt.final_norm.shift_params, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])