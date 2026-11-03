#!/usr/bin/env python3
"""Describe next-token logits for a standalone captured HTML code context."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
p=argparse.ArgumentParser();p.add_argument('--tokenizer',required=True);p.add_argument('--html',required=True);p.add_argument('--fixture',required=True);p.add_argument('--logits',nargs='*',default=[],help='label=NPZ path');p.add_argument('--output',required=True);a=p.parse_args()
tok=Tokenizer.from_file(a.tokenizer);html=Path(a.html).read_text();enc=tok.encode(html,add_special_tokens=False);fixture=json.loads(Path(a.fixture).read_text());ids=np.array(fixture['token_ids'],dtype='i8');positions=np.array(fixture['positions'],dtype='i8')
if not np.array_equal(ids,np.array(enc.ids[:len(ids)])):raise ValueError('fixture ID mismatch with tokenizer and HTML')
marker='new THREE.Wheel';start=html.index(marker);end=start+len(marker)
entries=[]
for i in range(4629,4642):
 s,e=enc.offsets[i];entries.append({'position':i,'token_id':enc.ids[i],'token_repr':enc.tokens[i],'source_text':html[s:e],'character_start':s,'character_end':e})
outputs=[]
for spec in a.logits:
 label,path=spec.split('=',1);x=np.load(path,allow_pickle=False); tokens=x['token_ids'];pos=x['positions']; logits=x['logits']
 if not np.array_equal(tokens,ids) or not np.array_equal(pos,positions) or logits.shape!=(len(pos),248320) or logits.dtype!=np.float32 or str(x['vocab_sha256'].item())!='5ee0f927bcaa4b9fe85c244776ae9487468e427f83e053fc81f2a186f14936a3' or not np.isfinite(logits).all(): raise ValueError(f'incompatible saved logits: {label}')
 cases=[]
 for row_pos in [4631,4632]:
  row=logits[np.flatnonzero(pos==row_pos)[0]].astype('f8');logz=np.log(np.exp(row-row.max()).sum())+row.max(); target=int(ids[row_pos+1]); rank=int(1+np.sum(row>row[target]));top=np.argsort(row)[-5:][::-1]
  cases.append({'last_consumed_position':row_pos,'target_position':row_pos+1,'target_id':target,'target_text':entries[row_pos+1-4629]['source_text'], 'probability':float(np.exp(row[target]-logz)),'rank':rank,'top5':[{'id':int(t),'text':tok.decode([int(t)],skip_special_tokens=False),'probability':float(np.exp(row[t]-logz))} for t in top]})
 outputs.append({'label':label,'path':path,'logits_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'predictions':cases,'teacher_forced_joint_probability_W_then_heel':cases[0]['probability']*cases[1]['probability']})
result={'scope':'standalone CPU-generated HTML teacher-forced context; not original sampled conversation; not causal proof', 'marker':marker,'marker_character_start':start,'marker_character_end_exclusive':end,'position_semantics':'row p predicts token p+1','identifier_tokens':'Wheel is split across .W and heel; p=4631 predicts .W and p=4632 predicts heel; .W alone does not establish an invalid identifier','tokens':entries,'evaluations':outputs}
Path(a.output).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result,indent=2,ensure_ascii=False))
