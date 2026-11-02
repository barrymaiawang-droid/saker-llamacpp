#!/usr/bin/env python3
"""Export independent HF BF16-weight logits from explicit token IDs.

Input NPZ: input_ids (1D integer), positions (1D integer, sorted unique).
Position p predicts the token after input_ids[p].  No tokenizer or sampler runs.
"""
import argparse, gc, hashlib, json, os, platform, time
from collections import Counter
from pathlib import Path
os.environ['ONEDNN_DEFAULT_FPMATH_MODE'] = 'STRICT'
import numpy as np
import torch
import transformers
from transformers import AutoModelForImageTextToText
import torch.nn.functional as F
import inspect
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_impl

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--repo', default='Qwen/Qwen3.8-27B')
p.add_argument('--revision', default='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
p.add_argument('--checkpoint-description', default='official unquantized BF16 release; native shard tensor dtypes may vary')
p.add_argument('--jobs', help='JSON list of {fixture,out,mode,chunk_size?,decode_start?}; model loads once')
p.add_argument('--fixture')
p.add_argument('--out')
p.add_argument('--mode', choices=['prefill','chunked','cached'])
p.add_argument('--threads', type=int, default=16)
p.add_argument('--chunk-size', type=int, default=256)
p.add_argument('--decode-start', type=int, default=None, help='Number of prefix tokens chunked before single-token decode')
p.add_argument('--dtype', choices=['bfloat16','float32'], default='bfloat16')
p.add_argument('--vocab-sha256', required=True, help='Verified canonical token ID mapping digest')
a = p.parse_args()
torch.set_num_threads(a.threads)
torch.set_float32_matmul_precision('highest')
for name in ('matmul','conv'):
    backend = getattr(torch.backends.mkldnn,name,None)
    if backend is not None and hasattr(backend,'fp32_precision'):
        backend.fp32_precision = 'ieee'
torch.set_autocast_enabled('cpu',False)
if torch.get_float32_matmul_precision() != 'highest' or torch.is_autocast_enabled('cpu'):
    raise RuntimeError('F32 precision or CPU autocast configuration did not take effect')
torch.set_grad_enabled(False)
load_started = time.monotonic()
model, loading_info = AutoModelForImageTextToText.from_pretrained(a.model, dtype=getattr(torch,a.dtype), attn_implementation='eager', local_files_only=True, low_cpu_mem_usage=True, output_loading_info=True)
model = model.eval()
print('loaded', round(time.monotonic()-load_started, 2), flush=True)
# Transformers ignores ^mtp.* via class-level ignore rules, so inspect the
# pinned index against actual model names as well as output_loading_info.
model_root = Path(a.model)
index_path = model_root/'model.safetensors.index.json'
if index_path.exists():
    checkpoint_keys = set(json.loads(index_path.read_text())['weight_map'])
else:
    from safetensors import safe_open
    checkpoint_keys = set()
    for shard in model_root.glob('*.safetensors'):
        with safe_open(shard, framework='pt', device='cpu') as f: checkpoint_keys.update(f.keys())
model_keys = set(model.state_dict())
checkpoint_extra = sorted(checkpoint_keys - model_keys)
mtp_expected = {'mtp.fc.weight','mtp.layers.0.input_layernorm.weight','mtp.layers.0.mlp.down_proj.weight','mtp.layers.0.mlp.gate_proj.weight','mtp.layers.0.mlp.up_proj.weight','mtp.layers.0.post_attention_layernorm.weight','mtp.layers.0.self_attn.k_norm.weight','mtp.layers.0.self_attn.k_proj.weight','mtp.layers.0.self_attn.o_proj.weight','mtp.layers.0.self_attn.q_norm.weight','mtp.layers.0.self_attn.q_proj.weight','mtp.layers.0.self_attn.v_proj.weight','mtp.norm.weight','mtp.pre_fc_norm_embedding.weight','mtp.pre_fc_norm_hidden.weight'} if a.repo == 'Qwen/Qwen3.8-27B' else set()
info = {k: sorted((str(x) for x in v)) for k,v in loading_info.items()}
loading_report = {'output_loading_info':info,'checkpoint_keys':len(checkpoint_keys),'model_state_keys':len(model_keys),'checkpoint_extra_keys':checkpoint_extra,'expected_unused_mtp_keys':sorted(mtp_expected),'model_class':type(model).__name__,'model_has_vision':any(k.startswith('model.visual.') for k in model_keys),'checkpoint_has_vision':any(k.startswith('model.visual.') for k in checkpoint_keys)}
report_base = Path(a.jobs) if a.jobs else Path(a.out)
report_path = report_base.with_suffix('.loading.json')
report_path.parent.mkdir(parents=True,exist_ok=True)
report_path.write_text(json.dumps(loading_report,indent=2)+'\n')
if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs') or any(x not in mtp_expected for x in info.get('unexpected_keys',[])):
    raise RuntimeError(f'unacceptable Transformers loading info; see {report_path}')
if set(checkpoint_extra) != mtp_expected:
    raise RuntimeError(f'checkpoint keys not accounted for by explicit unused MTP list; see {report_path}')
if a.repo == 'Qwen/Qwen3.8-27B' and not (loading_report['model_has_vision'] and loading_report['checkpoint_has_vision']):
    raise RuntimeError(f'expected vision tower missing from model or checkpoint; see {report_path}')

parameter_dtypes = Counter()
for param in model.parameters(): parameter_dtypes[str(param.dtype)] += param.numel()
print('parameter element dtypes', dict(parameter_dtypes), flush=True)
# The model forward yields hidden states; project only selected rows in float32.
# This avoids materializing all sequence-by-vocabulary logits and avoids BF16
# rounding of the final output projection; trunk uses the requested load dtype.
head = model.lm_head.weight.float()

model_load_seconds = time.monotonic()-load_started
def memory_status():
    status = Path('/proc/self/status').read_text()
    return {line.split(':',1)[0]:line.split(':',1)[1].strip() for line in status.splitlines() if line.startswith(('VmRSS:','VmHWM:','VmSize:'))}

def run_one(a):
    t0 = time.monotonic()
    model.model.rope_deltas = None
    fixture = json.loads(Path(a.fixture).read_text()) if a.fixture.endswith('.json') else np.load(a.fixture, allow_pickle=False)
    ids = np.asarray(fixture['token_ids'] if 'token_ids' in fixture else fixture['input_ids'], dtype=np.int64)
    pos = np.asarray(fixture['positions'], dtype=np.int64)
    if ids.ndim != 1 or pos.ndim != 1 or not len(pos) or np.any(pos < 0) or np.any(pos >= len(ids)) or np.any(np.diff(pos) <= 0):
        raise ValueError('expected 1D IDs and ascending unique in-range positions')
    rows = []
    with torch.inference_mode():
        tokens = torch.from_numpy(ids).unsqueeze(0)
        if a.mode == 'prefill':
            out = model.model(input_ids=tokens[:, :int(pos[-1])+1], use_cache=False)
            hidden = out.last_hidden_state[0, torch.from_numpy(pos), :].float()
            rows = F.linear(hidden, head).cpu().numpy()
        elif a.mode == 'chunked':
            if a.chunk_size < 1: raise ValueError('chunk size must be positive')
            past = None
            start = 0
            while start <= int(pos[-1]):
                end = min(start + a.chunk_size, int(pos[-1]) + 1)
                out = model.model(input_ids=tokens[:, start:end], past_key_values=past, use_cache=True)
                past = out.past_key_values
                chosen = pos[(pos >= start) & (pos < end)]
                if len(chosen):
                    selected = out.last_hidden_state[0, torch.from_numpy(chosen-start), :].float()
                    rows.extend(F.linear(selected, head).cpu().numpy())
                start = end
                print('chunk end', end, 'seconds', round(time.monotonic()-t0, 2), flush=True)
            rows = np.stack(rows)
        else:
            prefix_len = a.decode_start if a.decode_start is not None else int(pos[0])+1
            if not 1 <= prefix_len <= int(pos[-1])+1: raise ValueError('invalid decode-start')
            past = None
            for start in range(0, prefix_len, a.chunk_size):
                end = min(start + a.chunk_size, prefix_len)
                out = model.model(input_ids=tokens[:, start:end], past_key_values=past, use_cache=True)
                past = out.past_key_values
                chosen = pos[(pos >= start) & (pos < end)]
                if len(chosen):
                    selected = out.last_hidden_state[0, torch.from_numpy(chosen-start), :].float()
                    rows.extend(F.linear(selected, head).cpu().numpy())
            for i in range(prefix_len, int(pos[-1])+1):
                out = model.model(input_ids=tokens[:, i:i+1], past_key_values=past, use_cache=True)
                past = out.past_key_values
                if i in pos:
                    rows.append(F.linear(out.last_hidden_state[0,-1,:].float(), head).cpu().numpy())
                    print('position', i, 'seconds', round(time.monotonic()-t0, 2), flush=True)
            rows = np.stack(rows)
    rows = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(rows).all(): raise ValueError('non-finite logits')
    outpath = Path(a.out); outpath.parent.mkdir(parents=True,exist_ok=True)
    np.savez(outpath, input_ids=ids, token_ids=ids, vocab_sha256=np.asarray(a.vocab_sha256), positions=pos, logits=rows, vocab_ids=np.arange(rows.shape[1],dtype=np.int64))
    meta = {'repo':a.repo,'revision':a.revision,'mode':a.mode,'chunk_size':a.chunk_size if a.mode in ('chunked','cached') else None,'decode_start':(a.decode_start if a.decode_start is not None else int(pos[0])+1) if a.mode == 'cached' else None,'checkpoint':a.checkpoint_description,'parameter_elements_by_dtype':dict(parameter_dtypes),'requested_load_dtype':a.dtype,'device':'cpu','output_projection':'float32','attention':'eager','qwen_impl_source_sha256':hashlib.sha256(Path(inspect.getfile(qwen_impl)).read_bytes()).hexdigest(),'torch':torch.__version__,'transformers':transformers.__version__,'platform':platform.platform(),'float32_matmul_precision':torch.get_float32_matmul_precision(),'mkldnn_fp32_precision':{name:getattr(getattr(torch.backends.mkldnn,name,None),'fp32_precision',None) for name in ('matmul','conv')},'onednn_default_fpmath_mode':os.environ.get('ONEDNN_DEFAULT_FPMATH_MODE'),'cpu_autocast_enabled':torch.is_autocast_enabled('cpu'),'threads':a.threads,'torch_num_threads':torch.get_num_threads(),'hf_enable_parallel_loading':os.environ.get('HF_ENABLE_PARALLEL_LOADING'),'fixture_sha256':hashlib.sha256(Path(a.fixture).read_bytes()).hexdigest(),'elapsed_seconds':time.monotonic()-t0,'shape':list(rows.shape),'model_path':str(Path(a.model).resolve()),'memory_status':memory_status(),'model_load_seconds':model_load_seconds}
    outpath.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)

if a.jobs:
    if any(x is not None for x in (a.fixture,a.out,a.mode)):
        raise ValueError('--jobs cannot be combined with --fixture, --out or --mode')
    jobs = json.loads(Path(a.jobs).read_text())
    if not isinstance(jobs,list) or not jobs: raise ValueError('jobs must be a non-empty JSON list')
else:
    if not all((a.fixture,a.out,a.mode)): raise ValueError('provide --fixture, --out and --mode')
    jobs = [{}]
for job in jobs:
    case = argparse.Namespace(**vars(a))
    for key in ('fixture','out','mode','chunk_size','decode_start'):
        if key in job: setattr(case,key,job[key])
    if not all((case.fixture,case.out,case.mode)) or case.mode not in ('prefill','chunked','cached'):
        raise ValueError('each job must provide fixture, out and a valid mode')
    if Path(case.out).exists(): raise ValueError(f'refusing to overwrite {case.out}')
    print('starting job',case.mode,case.fixture,'memory',memory_status(),flush=True)
    run_one(case)
    gc.collect()
    print('finished job; released case state; memory',memory_status(),flush=True)
