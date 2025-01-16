from openvino.runtime import Core, Model, Tensor, PartialShape, Type, serialize, opset_utils
from openvino.runtime import opset10 as opset
import numpy as np
import sys, os
import argparse
import time
from tqdm import tqdm
from typing import Dict, Any
from pathlib import Path

from transformers import AutoTokenizer
from openvino.runtime import Core, Model, Tensor, PartialShape, Type, serialize, opset_utils
from openvino.runtime import opset10 as opset
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

# def apply_rotary_embedding(q, k, sin, cos, hidden_dim, opset):
#     """Apply rotary position embeddings to Q and K"""
#     # Reshape q and k to separate heads and rotary dimensions
#     # q_rot, q_pass = opset.split(q, axis=-1, num_splits=2)
#     # k_rot, k_pass = opset.split(k, axis=-1, num_split=2)
#     print("hidden_dim: ", hidden_dim)
#     q_rot = opset.slice(q,
#                         opset.constant([0], dtype=np.int64),
#                         opset.constant([hidden_dim // 2], dtype=np.int64),
#                         opset.constant([1], dtype=np.int64),
#                         axes=np.int64([3]))
#     q_pass = opset.slice(q,
#                         opset.constant([hidden_dim // 2], dtype=np.int64),
#                         opset.constant([hidden_dim], dtype=np.int64),
#                         opset.constant([1], dtype=np.int64),
#                         axes=np.int64([3]))
    
#     k_rot = opset.slice(k,
#                         opset.constant([0], dtype=np.int64),
#                         opset.constant([hidden_dim // 2], dtype=np.int64),
#                         opset.constant([1], dtype=np.int64),
#                         axes=np.int64([3]))
#     k_pass = opset.slice(k,
#                         opset.constant([hidden_dim // 2], dtype=np.int64),
#                         opset.constant([hidden_dim], dtype=np.int64),
#                         opset.constant([1], dtype=np.int64),
#                         axes=np.int64([3]))
    
#     # Reshape for rotation
#     q_rot_reshape = opset.reshape(q_rot, [-1, hidden_dim // 2, 2], False)
#     k_rot_reshape = opset.reshape(k_rot, [-1, hidden_dim // 2, 2], False)
    
#     # Apply rotation using sin and cos
#     q_rot_cos = opset.multiply(q_rot_reshape, opset.convert(cos, Type.f16))
#     q_rot_sin = opset.multiply(q_rot_reshape, opset.convert(sin, Type.f16))
#     k_rot_cos = opset.multiply(k_rot_reshape, opset.convert(cos, Type.f16))
#     k_rot_sin = opset.multiply(k_rot_reshape, opset.convert(sin, Type.f16))
    
#     # Reshape back and concatenate with pass-through part
#     q_rot_final = opset.reshape(q_rot_cos - q_rot_sin, opset.shape_of(q_rot), False)
#     k_rot_final = opset.reshape(k_rot_cos - k_rot_sin, opset.shape_of(k_rot), False)
    
#     # Concatenate rotated and pass-through parts
#     q_out = opset.concat([q_rot_final, q_pass], axis=-1)
#     k_out = opset.concat([k_rot_final, k_pass], axis=-1)
    
#     return q_out, k_out

def rotate_half(x, head_dim):
    """Rotates half the hidden dimensions of the input tensor."""
    split_dim = head_dim // 2 #x.get_shape()[-1] // 2
    shape_of = opset.shape_of(x)
    gather = opset.gather(shape_of, indices=np.int64([0, 1, 2]), axis=np.int64(0))
    x1 = opset.slice(x, [0, 0, 0, 0], opset.concat([gather, np.int64([split_dim])], axis=0), [1, 1, 1, 1], axes=[0, 1, 2, 3])
    x2 = opset.slice(x, [0, 0, 0, split_dim], opset.concat([gather, np.int64([head_dim])], axis=0), [1, 1, 1, 1], axes=[0, 1, 2, 3])
    neg_x2 = opset.multiply(x2, opset.constant(-1.0, Type.f16))
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
    cos_unsqueezed = opset.convert(opset.unsqueeze(cos, np.int64(unsqueeze_dim)), Type.f16)
    sin_unsqueezed = opset.convert(opset.unsqueeze(sin, np.int64(unsqueeze_dim)), Type.f16)

    # Apply Rotary Positional Embedding
    q_rotated = opset.add(opset.multiply(q, cos_unsqueezed), opset.multiply(rotate_half(q, head_dim), sin_unsqueezed))
    k_rotated = opset.add(opset.multiply(k, cos_unsqueezed), opset.multiply(rotate_half(k, head_dim), sin_unsqueezed))

    return q_rotated, k_rotated

def multi_head_attention(query, key, value, 
                        num_heads, 
                        head_dim,
                        mask=None,
                        sin=None,
                        cos=None,
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
    
    # 2. Apply rotary embeddings if provided
    if sin is not None and cos is not None:
        #q, k = apply_rotary_embedding(q, k, sin, cos, hidden_dim, opset)
        q, k = apply_rotary_pos_emb(q, k, sin, cos, head_dim)
    
    # 3. Calculate attention scores
    # Scale Q by sqrt(head_dim)
    scale = np.float32(1.0 / np.sqrt(head_dim))
    q_scaled = opset.multiply(q, opset.constant(scale, Type.f16))
    
    # Compute attention scores
    # [batch, num_heads, seq_len, seq_len]
    scores = opset.matmul(q_scaled, k, transpose_a=False, transpose_b=True)
    
    # 4. Apply attention mask if provided
    if mask is not None:
        scores = opset.add(scores, mask, name="masked_scores")
    
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
    
    return output
#=========================================================================


def make_experimental_fc(input, weight, name):
    quant_type = configs['quant_type']

    def quantize_weights(weight, quant_type):
        try:
            # build a FC node in `evaluate_qweight` mode to quantize & relayout weight
            qweight_node = custom_opset.create('FC', [Constant(weight, True)], {
                'quant_type':quant_type,
                'N' : weight.shape[0],
                'K' : weight.shape[1],
                'evaluate_qweight' : 1
            })
        except RuntimeError:
            # unsupported quant type
            return []

        # unsupported quant type
        if qweight_node.get_output_size() == 0:
            return []

        # create tensors with required shape & dtype to hold quantized weights
        output_vec = []
        for i in range(qweight_node.get_output_size()):
            ov_type = qweight_node.get_output_element_type(i)
            ov_shape = qweight_node.get_output_shape(i)
            output_vec.append(Tensor(ov_type, ov_shape))

        # evaluate_qweight
        if not qweight_node.evaluate(output_vec, [Tensor(weight)]):
            raise Exception("weight quantization failed!")

        return [Constant(w) for w in output_vec]

    quantized_weights_list = quantize_weights(weight, quant_type)

    if len(quantized_weights_list) == 0:
        return None

    return custom_opset.create('FC', [input, *quantized_weights_list] , {
        'quant_type':quant_type,
        'N' : weight.shape[0],
        'K' : weight.shape[1],
        'evaluate_qweight' : 0
    })


def make_fc(key, input, consts, name_suffix=""):
    # weight const f32 NxK
    weight = consts[f"{key}.weight"]

    # try experimental fc first
    matmul = make_experimental_fc(input, weight, key)

    # fallbacks
    if not matmul:
        if configs["quant_type"] == "nncf_w8":
            weights = _make_compressed_weight_nncf(weight, key)
        elif configs["quant_type"] == "":
            weights = Constant(weight, True)
            weights.set_friendly_name(name=f"{key}.weight{name_suffix}")
        else:
            raise Exception(f"Unknown quant type: {configs['quant_type']}")
        matmul = opset.matmul(input, weights, transpose_a=False, transpose_b=True, name=f"{key}.matmul{name_suffix}")

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
    weights = opset.constant(consts[f"{key}.weight"], Type.f16, name=f"{key}.weight{name_suffix}")
    pow = opset.multiply(input, input, name=f"{key}.pow{name_suffix}")
    #pow = opset.power(input, np.array([2], np.float32), name=f"{key}.pow{name_suffix}")
    variance = opset.reduce_mean(pow, reduction_axes=[-1], keep_dims=True, name=f"{key}.var{name_suffix}")
    add = opset.add(variance, opset.constant(epsilon, Type.f16), name=f"{key}.add{name_suffix}")
    sqrt = opset.sqrt(add, name=f"{key}.sqrt{name_suffix}")
    div = opset.divide(input, sqrt, name=f"{key}.div{name_suffix}")
    mul = opset.multiply(div, weights, auto_broadcast="numpy", name=f"{key}.mul{name_suffix}")
    return mul

def make_embedding(key, input, consts):
    if configs["quant_type"] != "":
        embed_in_const = _make_compressed_weight_nncf(consts[key], key)
    else:
        embed_in_const = Constant(consts[key], True)
        embed_in_const.set_friendly_name(name=key)
    inputs_embeds = opset.gather(embed_in_const, indices=input, axis=0)
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



def layer(configs, consts, layer_idx, hidden_states, kv_cache, beam_table, attn_mask, cos_tab, sin_tab):
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
    attn_output = multi_head_attention(q, k, v,
                        configs["head_num"],
                        configs["head_size"],
                        mask=attn_mask,
                        sin=sin_tab,
                        cos=cos_tab,
                        opset=opset)


    attn_output = make_fc("model.layers.self_attn.o_proj", attn_output, consts["layers"][layer_idx], name_suffix)

    attn_output = opset.add(hidden_states, attn_output, auto_broadcast="numpy", name=f"{name_prefix}.add0{name_suffix}")
    post_attention_layernorm = make_rms_norm("model.layers.post_attention_layernorm", attn_output, consts["layers"][layer_idx], configs["rms_norm_eps"], name_suffix)

    # mlp
    def mlp(states):
        gate_proj = make_fc("model.layers.mlp.gate_proj", states, consts["layers"][layer_idx], name_suffix)
        silu = opset.swish(gate_proj, beta=opset.constant(1.0, Type.f16))
        up_proj = make_fc("model.layers.mlp.up_proj", states, consts["layers"][layer_idx], name_suffix)
        mul = opset.multiply(silu, up_proj, auto_broadcast="numpy", name=f"{name_prefix}.mlp.mul{name_suffix}")
        down_proj = make_fc("model.layers.mlp.down_proj", mul, consts["layers"][layer_idx], name_suffix)
        return down_proj

    mlp_output = mlp(post_attention_layernorm)
    # residual connection.
    output = opset.add(attn_output, mlp_output, auto_broadcast="numpy", name=f"{name_prefix}.add1{name_suffix}")
    return output

def create_model(configs, consts):
    print(f"start generate ov model...")
    beg = time.time()
    # [batch, query_len]
    input_ids = opset.parameter([-1, -1], Type.i32, name="input_ids")
    # [2 * n_layers, batch, n_head, max_kv_len, head_size]
    kv_cache = opset.parameter([2 * configs["layer_num"], -1, configs["head_num"], -1, configs["head_size"]], Type.f16, name="kv_cache")
    # [batch, max_kv_len]
    beam_table = opset.parameter([-1, -1], Type.i32, name="beam_table")
    # [batch, query_len+past_len]
    attn_mask = opset.parameter([-1, -1], Type.f16, name="attn_mask")
    # [max_kv_len, rotary_dims//2]
    cos_tab = opset.parameter([-1, configs["rotary_dims"]], Type.f32, name="cos_tab")
    sin_tab = opset.parameter([-1, configs["rotary_dims"]], Type.f32, name="sin_tab")

    inputs_embeds = make_embedding("model.embed_tokens.weight", input_ids, consts)
    hidden_states = inputs_embeds

    for i in tqdm(range(configs["layer_num"])):
        hidden_states = layer(configs, consts, i, hidden_states, kv_cache, beam_table, attn_mask, cos_tab, sin_tab)
    # final_layernorm
    final_layernorm = make_rms_norm("model.norm", hidden_states, consts, configs["rms_norm_eps"])
    # embed_out
    embed_out = make_fc("lm_head", final_layernorm, consts)
    embed_out_result = opset.result(embed_out, name="logits")
    cost = time.time() - beg
    print(f"generate ov model done, cost {cost:.2f} seconds.")
    return Model([embed_out_result],
                 [input_ids, kv_cache, beam_table, attn_mask, cos_tab, sin_tab])

def load_hf_model(path):
    print(f"extracting from model '{path}'...")
    beg = time.time()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to("cpu").eval()

    assert(model.config.num_key_value_heads == model.config.num_attention_heads)
    assert(model.config.hidden_act in ["silu"])
    assert(model.config.rope_scaling is None)

    configs = {
        "layer_num": model.config.num_hidden_layers,
        "head_num": model.config.num_attention_heads,
        "head_size": model.config.hidden_size // model.config.num_attention_heads,
        "hidden_size": model.config.hidden_size,
        "max_position_embeddings": model.config.max_position_embeddings,
        "rotary_dims": int(model.config.hidden_size // model.config.num_attention_heads),
        #"gelu_mode": model.config.hidden_act,
        #"intermediate_size": model.config.intermediate_size,
        #"num_key_value_heads": model.config.num_key_value_heads,
        "rms_norm_eps": model.config.rms_norm_eps,
    }

    consts = {
        "model.embed_tokens.weight": pt_as_np(model.model.embed_tokens.weight),
        "model.norm.weight": pt_as_np(model.model.norm.weight),
        "lm_head.weight": pt_as_np(model.lm_head.weight),
        "lm_head.bias": pt_as_np(model.lm_head.bias),
        "layers": [
            {
                "model.layers.input_layernorm.weight": pt_as_np(l.input_layernorm.weight),
                "model.layers.post_attention_layernorm.weight": pt_as_np(l.post_attention_layernorm.weight),
                "model.layers.self_attn.q_proj.bias": pt_as_np(l.self_attn.q_proj.bias),
                "model.layers.self_attn.q_proj.weight": pt_as_np(l.self_attn.q_proj.weight),
                "model.layers.self_attn.k_proj.bias": pt_as_np(l.self_attn.k_proj.bias),
                "model.layers.self_attn.k_proj.weight": pt_as_np(l.self_attn.k_proj.weight),
                "model.layers.self_attn.v_proj.bias": pt_as_np(l.self_attn.v_proj.bias),
                "model.layers.self_attn.v_proj.weight": pt_as_np(l.self_attn.v_proj.weight),
                "model.layers.self_attn.o_proj.bias": pt_as_np(l.self_attn.o_proj.bias),
                "model.layers.self_attn.o_proj.weight": pt_as_np(l.self_attn.o_proj.weight),
                "model.layers.mlp.gate_proj.bias": pt_as_np(l.mlp.gate_proj.bias),
                "model.layers.mlp.gate_proj.weight": pt_as_np(l.mlp.gate_proj.weight),
                "model.layers.mlp.up_proj.bias": pt_as_np(l.mlp.up_proj.bias),
                "model.layers.mlp.up_proj.weight": pt_as_np(l.mlp.up_proj.weight),
                "model.layers.mlp.down_proj.bias": pt_as_np(l.mlp.down_proj.bias),
                "model.layers.mlp.down_proj.weight": pt_as_np(l.mlp.down_proj.weight)
            } for l in model.model.layers
        ],
    }
    cost = time.time() - beg
    print(f"extracting done, cost {cost:.2f} seconds.\nmodel configs:")
    for k, v in configs.items():
        print(f"	{k}: {v}")
    return configs, consts


def load_gguf_model(model_path: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract configurations and weights from GGUF model"""
    print(f"extracting from GGUF model '{model_path}'...")
    beg = time.time()

    # Load GGUF model
    weights, metadata = mx.load(model_path, return_metadata=True)

    config = {
        "layer_num": metadata["llama.block_count"].item(),
        "head_num": metadata["llama.attention.head_count"].item(),
        "head_size": metadata["llama.embedding_length"].item() // metadata["llama.attention.head_count"].item(),
        "hidden_size": metadata["llama.embedding_length"].item(),
        "max_position_embeddings": metadata["llama.context_length"].item(),
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
    save_tokenzier("meta-llama/Llama-2-7b-chat-hf", args.ov_model_path)