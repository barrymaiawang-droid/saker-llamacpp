#!/usr/bin/env python3
"""Inventory all GGUF tensor types and fail closed unless all are F32."""
import argparse, json, sys
from collections import Counter
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('gguf'); p.add_argument('--gguf-python',required=True); p.add_argument('--output',required=True)
a=p.parse_args(); sys.path.insert(0,a.gguf_python)
import gguf
reader=gguf.GGUFReader(a.gguf)
counts=Counter(str(t.tensor_type.name) for t in reader.tensors)
exceptions=[{'name':t.name,'type':t.tensor_type.name,'shape':[int(x) for x in t.shape]} for t in reader.tensors if t.tensor_type != gguf.GGMLQuantizationType.F32]
report={'path':str(Path(a.gguf).resolve()),'tensor_count':len(reader.tensors),'type_counts':dict(counts),'non_f32_tensors':exceptions,'ok':bool(reader.tensors) and not exceptions,'policy':'all GGUF tensors must be F32; no tensor exceptions accepted'}
Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
if not report['ok']: raise SystemExit(1)
