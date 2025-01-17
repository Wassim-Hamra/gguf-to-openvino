from openvino.runtime import Core, Model, Tensor, PartialShape, Type, serialize, opset_utils
from openvino.runtime import opset15 as opset
import numpy as np
import sys, os
import argparse
import time
from tqdm import tqdm
from typing import Dict, Any
from pathlib import Path

from transformers import AutoTokenizer, AutoConfig
from openvino.runtime.op import Constant
import numpy as np
import os
import sys
import torch
import mlx.core as mx
import openvino as ov

OV_XML_FILE_NAME="openvino_model.xml"

ext_path = None

custom_opset = opset_utils._get_node_factory()

configs = {
    "quant_type": ""#"nncf_w8",        # valid: "", "nncf_w8", "llama_w8_0",
}

def pt_as_np(t):
    if t is not None: return t.detach().numpy().astype(np.float32)
    return None

def show_model(m):
    print("inputs of the model:")
    for port, _input in enumerate(m.inputs):
        print("	[{}] {}".format(port, _input))
    print("outputs of the model:")
    for port, _output in enumerate(m.outputs):
        print("	[{}] {}".format(port, _output))

def make_mha(qkvs, kv_cache, beam_table, attn_mask, cos_tab, sin_tab,
             layer_idx, rotary_dim, n_hidden, n_head, name, num_kv_heads=0, rope_type="modified", multi_query_is_planar=False):
    qkvs_len = len(qkvs)
    mha_attr = {"arg_kv_cache": qkvs_len,
                "arg_beam_table": qkvs_len + 1,
                "arg_attn_mask": qkvs_len + 2,
                "arg_cos": qkvs_len + 3,
                "arg_sin": qkvs_len + 4,
                "layer_id": layer_idx,
                "rotary_dims": rotary_dim,
                "n_hidden": n_hidden,
                "n_head": n_head,
                "num_kv_heads": num_kv_heads,
                "multi_query_is_planar": multi_query_is_planar,
                "rope_type": ["original", "modified"].index(rope_type)}

    if qkvs_len == 1:
        mha_attr["arg_q"] = 0
        mha_attr["arg_k"] = 0
        mha_attr["arg_v"] = 0
    else:
        mha_attr["arg_q"] = 0
        mha_attr["arg_k"] = 1
        mha_attr["arg_v"] = 2

    output = custom_opset.create("MultiHeadAttention", 
        [*qkvs, kv_cache, beam_table, attn_mask, cos_tab, sin_tab], mha_attr)
    output.set_friendly_name(name)
    return output


#=========================================================================
def create_attention_mask(input_shape, opset):
    """Create causal attention mask"""
    seq_len = input_shape[1]
    # Create a matrix of shape [seq_len, seq_len] filled with ones in upper triangle
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    # Convert to float32 and negate to get proper attention mask (0 for attend, -inf for mask)
    mask = -10000.0 * mask
    return opset.constant(mask, Type.f32)

def rotate_half(x, head_dim):
    """Rotates half the hidden dimensions of the input tensor."""
    split_dim = head_dim // 2 #x.get_shape()[-1] // 2
    shape_of = opset.shape_of(x)
    gather = opset.gather(shape_of, indices=np.int64([0, 1, 2]), axis=np.int64(0))
    x1 = opset.slice(x, [0, 0, 0, 0], opset.concat([gather, np.int64([split_dim])], axis=0), [1, 1, 1, 1], axes=[0, 1, 2, 3])
    x2 = opset.slice(x, [0, 0, 0, split_dim], opset.concat([gather, np.int64([head_dim])], axis=0), [1, 1, 1, 1], axes=[0, 1, 2, 3])
    neg_x2 = opset.multiply(x2, opset.constant(-1.0, Type.f32))
    rotated = opset.concat([neg_x2, x1], axis=-1)
    return rotated

def apply_rotary_pos_emb(q, k, cos, sin, head_dim, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors using OpenVINO API.

    Args:
        q: Query tensor (OpenVINO node).
        k: Key tensor (OpenVINO node).
        cos: Cosine part of rotary embedding (OpenVINO node).
        sin: Sine part of rotary embedding (OpenVINO node).
        unsqueeze_dim: Dimension to unsqueeze cos and sin for broadcasting.
    Returns:
        Tuple of (q_rotated, k_rotated) tensors.
    """
    # Unsqueeze cos and sin along the specified dimension
    cos_unsqueezed = opset.unsqueeze(cos, np.int64(unsqueeze_dim))
    sin_unsqueezed = opset.unsqueeze(sin, np.int64(unsqueeze_dim))

    # Apply Rotary Positional Embedding
    q_rotated = opset.add(opset.multiply(q, cos_unsqueezed), opset.multiply(rotate_half(q, head_dim), sin_unsqueezed))
    k_rotated = opset.add(opset.multiply(k, cos_unsqueezed), opset.multiply(rotate_half(k, head_dim), sin_unsqueezed))

    return q_rotated, k_rotated


def rope_emb(x, rope_const, position_ids, seq_len=None):
    """
    Generates Rotary Position Embedding (RoPE) cosine and sine components using OpenVINO.

    Args:
        x: The input tensor to determine the device and dtype (OpenVINO node).
        rope_const: Tensor containing the rotary embedding constants (OpenVINO node).
        position_ids: Tensor with position IDs (OpenVINO node).
        seq_len: Optional sequence length (not used in computation here).

    Returns:
        cos: Cosine component of the rotary embedding.
        sin: Sine component of the rotary embedding.
    """
    # Expand dimensions for broadcasting
    inv_freq_expanded = opset.unsqueeze(rope_const, 0)  # Add batch dimension: [1, head_dim]
    inv_freq_expanded = opset.unsqueeze(inv_freq_expanded, -1)  # Shape: [1, head_dim, 1]

    position_ids_expanded = opset.convert(opset.unsqueeze(position_ids, 1), Type.f32)  # Add head_dim axis: [batch_size, 1, seq_len]

    # Compute frequencies
    freqs = opset.matmul(inv_freq_expanded, position_ids_expanded, transpose_a=False, transpose_b=False)  # Shape: [batch_size, seq_len, head_dim]
    freqs_transposed = opset.transpose(freqs, [0, 2, 1])  # Transpose to shape: [batch_size, head_dim, seq_len]

    # Concatenate frequencies along the last dimension
    emb = opset.concat([freqs_transposed, freqs_transposed], axis=-1)  # Shape: [batch_size, head_dim, seq_len * 2]

    # Compute cosine and sine
    cos = opset.cos(emb)
    sin = opset.sin(emb)

    return cos, sin


def multi_head_attention(query, key, value, 
                        num_heads,
                        head_dim,
                        batch_dim,
                        layer_idx,
                        mask=None,
                        position_ids=None,
                        rope_const=None,
                        beam_idx=None,
                        opset=opset):
    """
    Implements multi-head self-attention using OpenVINO operations
    
    Args:
        query: Query tensor of shape [batch_size, seq_len, hidden_dim]
        key: Key tensor of shape [batch_size, seq_len, hidden_dim]
        value: Value tensor of shape [batch_size, seq_len, hidden_dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        mask: Optional attention mask
        sin: Optional sine component for rotary embeddings
        cos: Optional cosine component for rotary embeddings
        opset: OpenVINO operation set to use
    """
    batch_size = opset.slice(opset.shape_of(query),
                         opset.constant([0], dtype=np.int64),
                         opset.constant([1], dtype=np.int64),
                         opset.constant([1], dtype=np.int64))
    #batch_size = opset.squeeze(batch_size, axes=opset.constant([0]))
    seq_len = opset.slice(opset.shape_of(query),
                         opset.constant([1], dtype=np.int64),
                         opset.constant([2], dtype=np.int64),
                         opset.constant([1], dtype=np.int64))
    #seq_len = opset.squeeze(seq_len, axes=opset.constant([0]))
    hidden_dim = num_heads * head_dim
    
    # 1. Reshape Q, K, V to split heads
    def split_heads(x, name):
        # Reshape from [batch, seq_len, hidden_dim] to [batch, seq_len, num_heads, head_dim]
        shape = opset.concat([batch_size, seq_len, np.int64([num_heads]), np.int64([head_dim])], axis=0)
        x = opset.reshape(x, shape, False)
        # Transpose to [batch, num_heads, seq_len, head_dim]
        x = opset.transpose(x, [0, 2, 1, 3], name=f"{name}_transpose")
        return x
    
    q = split_heads(query, "query")
    k = split_heads(key, "key")
    v = split_heads(value, "value")

    sin, cos = rope_emb(v, rope_const, position_ids)
    
    # 2. Apply rotary embeddings if provided
    if sin is not None and cos is not None:
        #q, k = apply_rotary_embedding(q, k, sin, cos, hidden_dim, opset)
        q, k = apply_rotary_pos_emb(q, k, sin, cos, head_dim)

    default_key = opset.broadcast(opset.constant([0.0], dtype=np.float32),
                                opset.concat([batch_dim,
                                            np.int64([num_heads]),
                                            np.int64([0]),
                                            np.int64([head_dim])],
                                            axis=0))
    k_cache_val = opset.read_value(default_key, 
                                   variable_shape=ov.PartialShape([-1,num_heads,-1,head_dim]),
                                   variable_type=Type.f32,
                                   variable_id=f"past_key_values.{layer_idx}.keypresent.{layer_idx}.key", name=f"past_key_values.{layer_idx}.keypresent.{layer_idx}.key")
    k_cache = opset.gather(k_cache_val, beam_idx, axis=0)
    default_value = opset.broadcast(opset.constant([0.0], dtype=np.float32),
                                opset.concat([batch_dim,
                                            np.int64([num_heads]),
                                            np.int64([0]),
                                            np.int64([head_dim])],
                                            axis=0))
    v_cache_val = opset.read_value(default_value, 
                                   variable_shape=ov.PartialShape([-1,num_heads,-1,head_dim]),
                                   variable_type=Type.f32,
                                   variable_id=f"past_key_values.{layer_idx}.valuepresent.{layer_idx}.key", name=f"past_key_values.{layer_idx}.valuepresent.{layer_idx}.key")
    v_cache = opset.gather(v_cache_val, beam_idx, axis=0)

    k_combined = opset.concat([k_cache, k], axis=2)
    v_combined = opset.concat([v_cache, v], axis=2)

    k_assigned = opset.assign(k_combined, f"past_key_values.{layer_idx}.keypresent.{layer_idx}.key", name=f"past_key_values.{layer_idx}.keypresent.{layer_idx}.key")
    v_assigned = opset.assign(v_combined, f"past_key_values.{layer_idx}.valuepresent.{layer_idx}.key", name=f"past_key_values.{layer_idx}.valuepresent.{layer_idx}.key")
    
    # 3. Calculate attention scores
    # Scale Q by sqrt(head_dim)
    scale = np.float32(1.0 / np.sqrt(head_dim))
    q_scaled = opset.multiply(q, opset.constant(scale, Type.f32))
    
    # Compute attention scores
    # [batch, num_heads, seq_len, seq_len]
    scores = opset.matmul(q_scaled, k, transpose_a=False, transpose_b=True)
    
    # 4. Apply attention mask if provided
    if mask is not None:
        scores = opset.add(opset.convert(scores, Type.f32), opset.convert(mask, Type.f32), name="masked_scores")
    
    # 5. Apply softmax
    attention_weights = opset.softmax(scores, axis=-1, name="attention_weights")
    
    # 6. Apply attention to values
    # [batch, num_heads, seq_len, head_dim]
    context = opset.matmul(attention_weights, v, transpose_a=False, transpose_b=False)
    
    # 7. Reshape output
    # Transpose back to [batch, seq_len, num_heads, head_dim]
    context_transposed = opset.transpose(context, [0, 2, 1, 3])
    
    # Combine heads: [batch, seq_len, hidden_dim]
    output = opset.reshape(context_transposed,
                          opset.concat([batch_size, seq_len, np.int64([hidden_dim])], axis=0),
                          False)
    
    return output, [k_assigned, v_assigned]
#=========================================================================

def make_fc(key, input, consts, name_suffix=""):
    # weight const f32 NxK
    weight = consts[f"{key}.weight"]

    weights = Constant(weight, True)
    weights.set_friendly_name(name=f"{key}.weight{name_suffix}")
    w_f32 = opset.convert(weights, Type.f32)

    matmul = opset.matmul(input, w_f32, transpose_a=False, transpose_b=True, name=f"{key}.matmul{name_suffix}")

    # add bias
    if consts[f"{key}.bias"] is not None:
        bias = Constant(consts[f"{key}.bias"], True)
        bias.set_friendly_name(name=f"{key}.bias{name_suffix}")
        matmul = opset.add(matmul, bias, auto_broadcast="numpy", name=f"{key}.add{name_suffix}")

    return matmul

def make_mvn(key, input, consts, configs, name_suffix=""):
    mvn = opset.mvn(input, axes=[-1], normalize_variance=True, eps=configs["layer_norm_eps"], eps_mode="inside_sqrt", name=f"{key}.mvn{name_suffix}")
    if consts[f"{key}.weight"] is not None:
        weights = opset.constant(consts[f"{key}.weight"], Type.f16, name=f"{key}.weight{name_suffix}")
        mvn = opset.multiply(mvn, weights, auto_broadcast="numpy", name=f"{key}.mul{name_suffix}")
    if consts[f"{key}.bias"] is not None:
        bias = opset.constant(consts[f"{key}.bias"], Type.f16, name=f"{key}.bias{name_suffix}")
        mvn = opset.add(mvn, bias, auto_broadcast="numpy", name=f"{key}.add{name_suffix}")
    return mvn

def make_rms_norm(key, input, consts, epsilon, name_suffix=""):
    weights = opset.constant(consts[f"{key}.weight"].astype(np.float32), Type.f32)
    epsilon_c = opset.constant(epsilon, Type.f32)
    #pow = opset.multiply(input, input, name=f"{key}.pow{name_suffix}")
    pow = opset.power(input, np.array([2], np.float32), name=f"{key}.pow{name_suffix}")
    variance = opset.reduce_mean(pow, reduction_axes=[-1], keep_dims=True, name=f"{key}.var{name_suffix}")
    add = opset.add(variance, epsilon_c, name=f"{key}.add{name_suffix}")
    sqrt = opset.sqrt(add, name=f"{key}.sqrt{name_suffix}")
    div = opset.divide(weights, sqrt, name=f"{key}.div{name_suffix}")
    mul = opset.multiply(div, input, auto_broadcast="numpy", name=f"{key}.mul{name_suffix}")
    return opset.convert(mul, input.get_element_type())

def make_embedding(key, input, consts):
    embed_in_const = Constant(consts[key], True)
    embed_f32 = opset.convert(embed_in_const, Type.f32)
    embed_f32.set_friendly_name(name=key)
    input_int32 = opset.convert(input, Type.i32)
    inputs_embeds = opset.gather(embed_f32, indices=input_int32, axis=0)
    return inputs_embeds

def save_tokenzier(orig_model_path, ov_model_path):
    tokenizer = AutoTokenizer.from_pretrained(orig_model_path)
    tokenizer.save_pretrained(ov_model_path)

    from openvino_tokenizers import convert_tokenizer
    OV_TOKENIZER_NAME = "openvino_tokenizer.xml"
    OV_DETOKENIZER_NAME = "openvino_detokenizer.xml"

    converted = convert_tokenizer(tokenizer, with_detokenizer=True)
    for model, file_name in zip(converted, (OV_TOKENIZER_NAME, OV_DETOKENIZER_NAME)):
        ov.save_model(model, Path(ov_model_path) / file_name)



def layer(configs, consts, layer_idx, hidden_states, attn_mask, position_ids, rope_const, beam_idx, batch_dim):
    name_suffix = f".layer{layer_idx}"
    name_prefix = "model.layers.self_attn"
    # layerNorm operation
    input_layernorm = make_rms_norm("model.layers.input_layernorm", hidden_states, consts["layers"][layer_idx], configs["rms_norm_eps"], name_suffix)

    q = make_fc("model.layers.self_attn.q_proj", input_layernorm, consts["layers"][layer_idx], name_suffix)
    k = make_fc("model.layers.self_attn.k_proj", input_layernorm, consts["layers"][layer_idx], name_suffix)
    v = make_fc("model.layers.self_attn.v_proj", input_layernorm, consts["layers"][layer_idx], name_suffix)

    # custom op
    # attn_output = make_mha([q, k, v], kv_cache, beam_table, attn_mask, cos_tab, sin_tab,
    #                        layer_idx, configs["rotary_dims"], configs["hidden_size"], configs["head_num"],
    #                        name=f"{name_prefix}.mha{name_suffix}")
    attn_output, sinks = multi_head_attention(q, k, v,
                        configs["head_num"],
                        configs["head_size"],
                        batch_dim=batch_dim,
                        layer_idx=layer_idx,
                        mask=attn_mask,
                        position_ids=position_ids,
                        rope_const=rope_const,
                        beam_idx=beam_idx,
                        opset=opset)


    attn_output = make_fc("model.layers.self_attn.o_proj", attn_output, consts["layers"][layer_idx], name_suffix)

    attn_output = opset.add(hidden_states, attn_output, auto_broadcast="numpy", name=f"{name_prefix}.add0{name_suffix}")
    post_attention_layernorm = make_rms_norm("model.layers.post_attention_layernorm", attn_output, consts["layers"][layer_idx], configs["rms_norm_eps"], name_suffix)

    # mlp
    def mlp(states):
        gate_proj = make_fc("model.layers.mlp.gate_proj", states, consts["layers"][layer_idx], name_suffix)
        silu = opset.swish(gate_proj)
        up_proj = make_fc("model.layers.mlp.up_proj", states, consts["layers"][layer_idx], name_suffix)
        mul = opset.multiply(silu, up_proj, auto_broadcast="numpy", name=f"{name_prefix}.mlp.mul{name_suffix}")
        down_proj = make_fc("model.layers.mlp.down_proj", mul, consts["layers"][layer_idx], name_suffix)
        return down_proj

    mlp_output = mlp(post_attention_layernorm)
    # residual connection.
    output = opset.add(attn_output, mlp_output, auto_broadcast="numpy", name=f"{name_prefix}.add1{name_suffix}")
    return output, sinks


def init_rope(head_dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(device) / head_dim))
    # For BC we register cos and sin cached
    rope_const = opset.constant(inv_freq.numpy(), Type.f32)
    return rope_const


def create_model(configs, consts):
    print(f"start generate ov model...")
    beg = time.time()
    # [batch, query_len]
    input_ids = opset.parameter([-1, -1], Type.i64, name="input_ids")
    # [batch, query_len+past_len]
    attention_mask = opset.parameter([-1, -1], Type.i64, name="attention_mask")
    # [batch, query_len+past_len]
    position_ids = opset.parameter([-1, -1], Type.i64, name="position_ids")
    # [batch, max_kv_len]
    beam_idx = opset.parameter([-1], Type.i32, name="beam_idx")

    inputs_embeds = make_embedding("model.embed_tokens.weight", input_ids, consts)
    hidden_states = inputs_embeds

    rope_const = init_rope(configs["head_size"], configs["max_position_embeddings"])

    input_shape = opset.shape_of(input_ids)
    batch_size = opset.gather(input_shape,
                         opset.constant([0], dtype=np.int64),
                         opset.constant([0], dtype=np.int64))

    sinks = []
    for i in tqdm(range(configs["layer_num"])):
        hidden_states, layer_sinks = layer(configs, consts, i, hidden_states, attention_mask, position_ids, rope_const, beam_idx, batch_size)
        sinks = sinks + layer_sinks
    # final_layernorm
    final_layernorm = make_rms_norm("model.norm", hidden_states, consts, configs["rms_norm_eps"])
    # embed_out
    embed_out = make_fc("lm_head", final_layernorm, consts)

    logits = opset.result(opset.convert(embed_out, Type.f32), name="logits")
    logits.set_friendly_name("logits")
    cost = time.time() - beg
    print(f"generate ov model done, cost {cost:.2f} seconds.")
    model = Model([logits], sinks,
                 [input_ids, attention_mask, position_ids, beam_idx])
    model.outputs[0].get_tensor().set_names({"logits"})
    return model


def load_gguf_model(model_path: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract configurations and weights from GGUF model"""
    print(f"extracting from GGUF model '{model_path}'...")
    beg = time.time()

    # Load GGUF model
    weights, metadata = mx.load(model_path, return_metadata=True)

    print(metadata.keys())

    config = {
        "layer_num": metadata["llama.block_count"].item(),
        "head_num": metadata["llama.attention.head_count"].item(),
        "head_size": metadata["llama.embedding_length"].item() // metadata["llama.attention.head_count"].item(),
        "hidden_size": metadata["llama.embedding_length"].item(),
        "max_position_embeddings": metadata.get("llama.context_length", np.int32([2048])).item(),
        "rotary_dims": metadata["llama.rope.dimension_count"].item(),
        "rms_norm_eps": metadata["llama.attention.layer_norm_rms_epsilon"].item(),
    }

    print("config: ", config)

    # Extract weights and biases
    print("Extract weights and biases")
    print('weights["token_embd.weight"]: ', type(weights["token_embd.weight"]))
    consts = {
        "model.embed_tokens.weight": np.array(weights["token_embd.weight"]),
        "model.norm.weight": np.array(weights["output_norm.weight"]),
        "lm_head.weight": np.array(weights["output.weight"]),
        "lm_head.bias": None,  # GGUF models typically don"t have this bias
        "layers": []
    }
    
    # Extract layer weights
    print("Extract layer weights")
    for i in range(config["layer_num"]):
        layer_weights = {
            "model.layers.input_layernorm.weight": np.array(weights[f"blk.{i}.attn_norm.weight"]),
            "model.layers.post_attention_layernorm.weight": np.array(weights[f"blk.{i}.ffn_norm.weight"]),
            "model.layers.self_attn.q_proj.bias": None,
            "model.layers.self_attn.q_proj.weight": np.array(weights[f"blk.{i}.attn_q.weight"]),
            "model.layers.self_attn.k_proj.bias": None,
            "model.layers.self_attn.k_proj.weight": np.array(weights[f"blk.{i}.attn_k.weight"]),
            "model.layers.self_attn.v_proj.bias": None,
            "model.layers.self_attn.v_proj.weight": np.array(weights[f"blk.{i}.attn_v.weight"]),
            "model.layers.self_attn.o_proj.bias": None,
            "model.layers.self_attn.o_proj.weight": np.array(weights[f"blk.{i}.attn_output.weight"]),
            "model.layers.mlp.gate_proj.bias": None,
            "model.layers.mlp.gate_proj.weight": np.array(weights[f"blk.{i}.ffn_gate.weight"]),
            "model.layers.mlp.up_proj.bias": None,
            "model.layers.mlp.up_proj.weight": np.array(weights[f"blk.{i}.ffn_up.weight"]),
            "model.layers.mlp.down_proj.bias": None,
            "model.layers.mlp.down_proj.weight": np.array(weights[f"blk.{i}.ffn_down.weight"])
        }
        consts["layers"].append(layer_weights)
    
    cost = time.time() - beg
    print(f"extracting done, cost {cost:.2f} seconds.\nmodel configs:")
    for k, v in config.items():
        print(f"    {k}: {v}")
    return config, consts


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--org_model_path", type=str, default="Model ID (can be a Hugginface Hub id, or a local directory)")
    parser.add_argument("--ov_model_path", type=str, nargs="?", default="./gen/llama-2-7b-chat/")
    parser.add_argument("--compressed_weight", type=bool, nargs="?", default=False)
    parser.add_argument("--quant_type", type=str, nargs="?", default="")
    args = parser.parse_args()
    # for compatible, will remove
    if args.compressed_weight:
        print(f"warning: please use '--quant=nncf_w8' instead.")
        if args.quant_type:
            raise ValueError("compressed_weight and quant_type can not be set at the same time.")
        args.quant_type = "nncf_w8"
    configs["quant_type"] = args.quant_type

    if args.quant_type:
        args.ov_model_path = os.path.join(args.ov_model_path, args.quant_type)
    os.makedirs(args.ov_model_path, exist_ok=True)

    model_configs, consts = load_gguf_model(args.org_model_path)
    model = create_model(model_configs, consts)
    show_model(model)
    print(f"serialize ov model to '{args.ov_model_path}'...")
    beg = time.time()
    serialize(model, os.path.join(args.ov_model_path, OV_XML_FILE_NAME))
    cost = time.time() - beg
    print(f"serialize done, cost {cost:.2f} seconds.")
    print(f"save tokenzier to '{args.ov_model_path}' ...")
    # save tokenizer and config to load with GenAI and Optimum
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    save_tokenzier(model_id, args.ov_model_path)
    config = AutoConfig.from_pretrained(model_id)
    config.save_pretrained(args.ov_model_path)