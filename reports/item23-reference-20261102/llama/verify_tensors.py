import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

p = argparse.ArgumentParser()
p.add_argument('model_dir')
p.add_argument('gguf')
p.add_argument('--gguf-python', required=True)
a = p.parse_args()
sys.path.insert(0, a.gguf_python)
import gguf
from gguf.quants import dequantize

root = Path(a.model_dir)
index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
cfg = json.loads((root / 'config.json').read_text())['text_config']
k = cfg['linear_num_key_heads']; v = cfg['linear_num_value_heads']; kd = cfg['linear_key_head_dim']; vd = cfg['linear_value_head_dim']; ratio = v // k
reader = gguf.GGUFReader(a.gguf)
gg = {t.name: t for t in reader.tensors}

def hf(name):
    with safe_open(root / index[name], framework='pt', device='cpu') as f:
        return f.get_tensor(name)

def reorder(t, dim, width):
    shape = list(t.shape)
    shape[dim:dim+1] = [k, ratio, width]
    axes = list(range(len(shape)))
    axes[dim], axes[dim+1] = axes[dim+1], axes[dim]
    return t.reshape(shape).permute(axes).contiguous().reshape(t.shape)

def values(t):
    if t.tensor_type == gguf.GGMLQuantizationType.BF16:
        return dequantize(t.data, t.tensor_type).reshape(tuple(reversed(t.shape)))
    if t.tensor_type == gguf.GGMLQuantizationType.F32:
        return t.data.reshape(tuple(reversed(t.shape)))
    raise ValueError(f'Unexpected GGUF type {t.name}: {t.tensor_type}')

def check(src, dest, transform=lambda x: x, rows=None):
    original = hf(src)
    expected_tensor = transform(original.float())
    t = gg[dest]
    if t.tensor_type == gguf.GGMLQuantizationType.BF16:
        expected_tensor = expected_tensor.to(torch.bfloat16)
    expected = expected_tensor.float().numpy()
    actual = values(t)
    if expected.shape != actual.shape: raise ValueError(f'{dest}: shapes {expected.shape} vs {actual.shape}')
    if rows is not None:
        expected, actual = expected[rows], actual[rows]
    if not np.array_equal(expected, actual):
        raise ValueError(f'{src} -> {dest} mismatch max abs {np.max(np.abs(expected-actual))}')
    print(json.dumps({'source':src,'target':dest,'shape':list(t.shape[::-1]),'target_type':str(t.tensor_type),'rows_checked':rows,'values_sha256':hashlib.sha256(np.ascontiguousarray(actual).view(np.uint8)).hexdigest(),'exact':True}), flush=True)

def check_matrix_stream(src, dest, chunk=256):
    t = gg[dest]
    expected_shape = tuple(reversed(t.shape))
    digest = hashlib.sha256()
    with safe_open(root / index[src], framework='pt', device='cpu') as f:
        tensor = f.get_slice(src)
        if tuple(tensor.get_shape()) != expected_shape: raise ValueError(f'{src}: shape mismatch')
        for start in range(0, expected_shape[0], chunk):
            stop = min(start + chunk, expected_shape[0])
            expected = tensor[start:stop].float()
            if t.tensor_type == gguf.GGMLQuantizationType.BF16:
                expected = expected.to(torch.bfloat16).float()
                actual = dequantize(t.data[start:stop], t.tensor_type).reshape(stop-start, *expected_shape[1:])
            elif t.tensor_type == gguf.GGMLQuantizationType.F32:
                actual = t.data[start:stop].reshape(stop-start, *expected_shape[1:])
            else: raise ValueError(f'{dest}: unsupported {t.tensor_type}')
            expected = expected.numpy()
            if not np.array_equal(expected, actual): raise ValueError(f'{src} mismatch in rows {start}:{stop}')
            digest.update(np.ascontiguousarray(actual).view(np.uint8))
    print(json.dumps({'source':src,'target':dest,'shape':list(expected_shape),'rows_checked':'all including padded tail','target_type':str(t.tensor_type),'values_sha256':digest.hexdigest(),'exact':True}),flush=True)

print(json.dumps({'coverage':'all embedding and output-head rows, plus listed representative tensors; not all checkpoint tensors'}),flush=True)
base='model.language_model.layers.0.linear_attn.'
check_matrix_stream('model.language_model.embed_tokens.weight', 'token_embd.weight')
check_matrix_stream('lm_head.weight', 'output.weight')
check('model.language_model.layers.0.input_layernorm.weight', 'blk.0.attn_norm.weight', lambda t: 1.0 + t)
check('model.language_model.layers.0.post_attention_layernorm.weight', 'blk.0.ffn_norm.weight', lambda t: 1.0 + t)
check('model.language_model.norm.weight', 'output_norm.weight', lambda t: 1.0 + t)
check(base+'norm.weight', 'blk.0.ssm_norm.weight')
check('model.language_model.layers.3.self_attn.q_proj.weight', 'blk.3.attn_q.weight')
check('model.language_model.layers.3.self_attn.k_proj.weight', 'blk.3.attn_k.weight')
check(base+'in_proj_qkv.weight', 'blk.0.attn_qkv.weight', lambda t: torch.cat((t[:2*k*kd], reorder(t[2*k*kd:],0,vd)),0))
check(base+'conv1d.weight', 'blk.0.ssm_conv1d.weight', lambda t: torch.cat((t.squeeze()[:2*k*kd],reorder(t.squeeze()[2*k*kd:],0,vd)),0))
check(base+'in_proj_a.weight', 'blk.0.ssm_alpha.weight', lambda t: reorder(t,0,1))
check(base+'in_proj_b.weight', 'blk.0.ssm_beta.weight', lambda t: reorder(t,0,1))
check(base+'out_proj.weight', 'blk.0.ssm_out.weight', lambda t: reorder(t,1,vd))
check(base+'dt_bias', 'blk.0.ssm_dt.bias', lambda t: reorder(t,0,1))
check(base+'A_log', 'blk.0.ssm_a', lambda t: -torch.exp(reorder(t,0,1)))
