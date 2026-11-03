#!/usr/bin/env python3
"""Compare HF tokenizer.json ID-to-token mapping with GGUF vocabulary export.

GGUF export is JSON mapping tokenizer.ggml.{tokens,token_type,...} to values.
A matching digest certifies only identical ID token strings; verify special
semantics, template, BOS/EOS and padded logits independently before comparing.
"""
import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('tokenizer_json'); p.add_argument('gguf_vocab_json'); p.add_argument('--output',required=True)
a=p.parse_args()
hf=json.loads(Path(a.tokenizer_json).read_text())
g=json.loads(Path(a.gguf_vocab_json).read_text())
gtokens=g['tokenizer.ggml.tokens']; vocab=hf['model']['vocab']
if not isinstance(vocab,dict): raise ValueError('expected tokenizer model.vocab object')
hid={int(i):str(t) for t,i in vocab.items()}
for x in hf.get('added_tokens',[]):
    i=int(x['id']); token=x['content']
    if i in hid and hid[i]!=token: raise ValueError(f'conflicting HF token ID {i}')
    hid[i]=token
padding = [i for i,gt in enumerate(gtokens) if i not in hid and g.get('tokenizer.ggml.token_type',[])[i] == 5 and gt == f'[PAD{i}]']
mismatches=[{'id':i,'hf':hid.get(i),'gguf':gt} for i,gt in enumerate(gtokens) if hid.get(i)!=gt and i not in padding]
extra=[{'id':i,'hf':t} for i,t in hid.items() if i>=len(gtokens)]
canonical=json.dumps(gtokens,ensure_ascii=False,separators=(',',':')).encode('utf-8')
report={'hf_size':len(hid),'gguf_size':len(gtokens),'gguf_unused_padding_count':len(padding),'mismatch_count':len(mismatches),'hf_extra_count':len(extra),
        'mismatches':mismatches[:100], 'hf_extra':extra[:100],
        'hf_added_tokens':hf.get('added_tokens',[]),
        'gguf_special_ids':{k:v for k,v in g.items() if k.endswith('_token_id')},
        'gguf_token_types_count':{str(t):g.get('tokenizer.ggml.token_type',[]).count(t) for t in set(g.get('tokenizer.ggml.token_type',[]))},
        'matched_id_strings_sha256':hashlib.sha256(canonical).hexdigest() if not mismatches and not extra else None,
        'hash_definition':'SHA256 of UTF-8 JSON array of GGUF ID-order token strings, ensure_ascii=False, compact separators',
        'limitations':'String IDs and conventionally named GGUF UNUSED padding only; this does not validate that HF output head rows and GGUF tensor rows correspond. Compare special-token semantics, BOS/EOS policy, template, pretokenization and padded output rows separately. Validate actual BF16 GGUF, not only a related quantized file.'}
Path(a.output).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('hf_added_tokens','mismatches','hf_extra')},ensure_ascii=False))
if mismatches or extra: raise SystemExit(1)
