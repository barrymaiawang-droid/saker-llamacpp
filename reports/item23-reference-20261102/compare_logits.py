#!/usr/bin/env python3
"""Compare matched, full-vocabulary float32 logits stored as NPZ archives.

Each archive: logits [rows,vocab] float32, positions [rows] int64 (zero-based,
row predicts next token), token_ids [sequence] int64, vocab_sha256 scalar string.
The vocabulary hash must encode an independently verified identical ID mapping.
"""
import argparse
import hashlib
import json
import numpy as np


def read(path):
    x = np.load(path, allow_pickle=False)
    required = {'logits', 'positions', 'token_ids', 'vocab_sha256'}
    if not required.issubset(x.files):
        raise ValueError(f'{path}: missing {required - set(x.files)}')
    logits, pos, ids = x['logits'], x['positions'], x['token_ids']
    if logits.dtype != np.float32 or logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError(f'{path}: logits must be finite 2D float32')
    if pos.ndim != 1 or ids.ndim != 1 or pos.dtype.kind not in 'iu' or ids.dtype.kind not in 'iu':
        raise ValueError(f'{path}: positions and token_ids must be 1D integers')
    if not len(pos) or len(pos) != len(logits) or len(set(pos.tolist())) != len(pos) or np.any(pos < 0) or np.any(pos >= len(ids)):
        raise ValueError(f'{path}: invalid/duplicate positions')
    if np.any(ids < 0) or np.any(ids >= logits.shape[1]):
        raise ValueError(f'{path}: token ID out of vocabulary')
    v = str(x['vocab_sha256'].item())
    if len(v) != 64 or any(c not in '0123456789abcdef' for c in v):
        raise ValueError(f'{path}: invalid vocabulary hash')
    return logits, pos, ids, v


def logsoftmax(x):
    x = x.astype(np.float64)
    x -= np.max(x)
    return x - np.log(np.exp(x).sum())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference'); p.add_argument('candidate'); p.add_argument('--output')
    a = p.parse_args()
    ref, rp, ri, rv = read(a.reference)
    got, gp, gi, gv = read(a.candidate)
    if ref.shape != got.shape or not np.array_equal(rp, gp) or not np.array_equal(ri, gi) or rv != gv:
        raise ValueError('archives differ in shape, positions, token IDs, or verified vocabulary hash')
    rows = []
    for pos, x, y in zip(rp, ref, got):
        lp, lq = logsoftmax(x), logsoftmax(y)
        probs = np.exp(lp)
        delta = y.astype(np.float64) - x
        # Shift logits by a constant for a distribution-invariant diagnostic.
        centered = delta - np.mean(delta)
        k = min(10, len(x))
        topx = np.argpartition(x, -k)[-k:]
        topy = np.argpartition(y, -k)[-k:]
        rows.append({'position': int(pos), 'next_token_id': int(ri[pos+1]) if pos+1 < len(ri) else None,
                     'kl_ref_to_candidate': float(np.sum(probs * (lp-lq))),
                     'reference_top_id': int(np.argmax(x)), 'candidate_top_id': int(np.argmax(y)),
                     'top1_agrees': bool(np.argmax(x)==np.argmax(y)),
                     'top10_intersection': int(len(np.intersect1d(topx,topy))),
                     'max_abs_logit_delta': float(np.max(np.abs(delta))),
                     'mean_offset': float(np.mean(delta)),
                     'max_abs_centered_logit_delta': float(np.max(np.abs(centered))),
                     'reference_top_probability': float(np.max(probs)),
                     'reference_top_probability_candidate': float(np.exp(lq[np.argmax(x)]))})
    kl = np.array([r['kl_ref_to_candidate'] for r in rows])
    result = {'reference':a.reference, 'candidate':a.candidate, 'vocab_sha256':rv,
              'token_ids_sha256_le_i64':hashlib.sha256(ri.astype('<i8').tobytes()).hexdigest(),
              'vocab_size':ref.shape[1], 'row_count':len(rows),
              'mean_kl_ref_to_candidate':float(np.mean(kl)), 'median_kl':float(np.median(kl)),
              'p95_kl':float(np.percentile(kl,95)), 'p99_kl':float(np.percentile(kl,99)),
              'max_kl':float(np.max(kl)), 'top1_agreement':float(np.mean([r['top1_agrees'] for r in rows])),
              'max_abs_logit_delta':max(r['max_abs_logit_delta'] for r in rows),
              'max_abs_centered_logit_delta':max(r['max_abs_centered_logit_delta'] for r in rows), 'rows':rows}
    s=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if a.output:
        with open(a.output,'w') as f: f.write(s)
    else: print(s,end='')

if __name__ == '__main__': main()
