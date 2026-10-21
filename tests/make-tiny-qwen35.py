#!/usr/bin/env python3
"""Create random, non-language weights for integration tests. Requires llama.cpp/gguf-py and numpy."""
import argparse
from pathlib import Path
import sys
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--llama-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--predictable', action='store_true', help='make target and draft prefer ASCII A for acceptance tests')
args = parser.parse_args()
sys.path.insert(0, str(args.llama_dir.resolve() / 'gguf-py'))
import gguf

rng = np.random.default_rng(41)
w = gguf.GGUFWriter(str(args.output), 'qwen35')
w.add_name('saker-random-integration-fixture')
w.add_context_length(4096)
w.add_embedding_length(64)
w.add_block_count(5)
w.add_nextn_predict_layers(1)
w.add_feed_forward_length(128)
w.add_head_count(4)
w.add_head_count_kv(2)
w.add_layer_norm_rms_eps(1e-6)
w.add_rope_dimension_count(16)
w.add_rope_dimension_sections([4, 4, 0, 0])
w.add_rope_freq_base(10000)
w.add_ssm_conv_kernel(4)
w.add_ssm_inner_size(32)
w.add_ssm_state_size(16)
w.add_ssm_time_step_rank(2)
w.add_ssm_group_count(1)
w.add_uint32('qwen35.full_attention_interval', 4)
w.add_tokenizer_model('gpt2')
w.add_tokenizer_pre('gpt-2')
# GPT-2 byte alphabet, no merges. Text tokenization is sufficient for smoke tests.
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
cs = list(bs)
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + len(cs) - 188)
alphabet = dict(zip(bs, map(chr, cs)))
w.add_token_list([alphabet[i] for i in range(256)] + ['th', '<|endoftext|>'])
w.add_token_types([1] * 257 + [3])
w.add_token_merges(['t h'])
w.add_bos_token_id(257)
w.add_eos_token_id(257)
w.add_add_bos_token(False)
w.add_add_eos_token(False)

def tensor(name, shape, norm=False, constant=None):
    a = (rng.standard_normal(shape) * 0.02).astype(np.float32)
    if args.predictable:
        a = np.abs(a)
        if name == "output":
            a.fill(0)
            a[65].fill(1)
    if norm:
        a.fill(1)
    if constant is not None:
        a.fill(constant)
    w.add_tensor(name + '.weight', a)

tensor('token_embd', (258, 64))
tensor('output', (258, 64))
tensor('output_norm', (64,), norm=True)
for layer in range(5):
    prefix = f'blk.{layer}.'
    tensor(prefix + 'attn_norm', (64,), norm=True)
    tensor(prefix + 'post_attention_norm', (64,), norm=True)
    tensor(prefix + 'ffn_up', (128, 64))
    tensor(prefix + 'ffn_gate', (128, 64))
    tensor(prefix + 'ffn_down', (64, 128))
    if layer < 3:
        tensor(prefix + 'attn_qkv', (64, 64))
        tensor(prefix + 'attn_gate', (32, 64))
        tensor(prefix + 'ssm_conv1d', (64, 4))
        w.add_tensor(prefix + 'ssm_dt.bias', np.zeros(2, np.float32))
        w.add_tensor(prefix + 'ssm_a', np.full(2, -0.1, np.float32))
        tensor(prefix + 'ssm_beta', (2, 64))
        tensor(prefix + 'ssm_alpha', (2, 64))
        tensor(prefix + 'ssm_norm', (16,), norm=True)
        tensor(prefix + 'ssm_out', (64, 32))
    else:
        tensor(prefix + 'attn_q', (128, 64))
        tensor(prefix + 'attn_k', (32, 64))
        tensor(prefix + 'attn_v', (32, 64))
        tensor(prefix + 'attn_output', (64, 64))
        tensor(prefix + 'attn_q_norm', (16,), norm=True)
        tensor(prefix + 'attn_k_norm', (16,), norm=True)
    if layer == 4:
        tensor(prefix + 'nextn.eh_proj', (64, 128))
        tensor(prefix + 'nextn.enorm', (64,), norm=True)
        tensor(prefix + 'nextn.hnorm', (64,), norm=True)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(args.output)
