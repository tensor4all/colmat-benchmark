#!/usr/bin/env python3
"""Pair two recorder runs (default vs x86-v4) cell by cell. colmat-rs#39.

usage: scripts/pair_feature_arms.py <default run dir> <x86-v4 run dir> [--row colmat] [--csv out.csv]

A cell is (regime, family, dtype, order, shape, batch, row, threads, variant) -- the variant is the CSV's
`note` (`api-inclusive`, a Householder `block(...)`/`sequence(...)` form), so those rows are never pooled. Each run has N process sets
(runK-*.csv); a set's value for a cell is its recorded total_ms (best of its reps).
  ratio   = median(default) / median(x86-v4)   (> 1: x86-v4 faster)
  spread  = (max - min) / median over one arm's sets
  regress = x86-v4 slower by more than max(spread_default, spread_v4) AND the arms' ranges do not overlap
  win     = x86-v4 faster by more than max(spreads) AND ranges do not overlap
"""
import argparse, collections, csv, pathlib, re, statistics, sys

def load(run):
    cells = collections.defaultdict(list)
    for f in sorted(pathlib.Path(run).glob('run*-*.csv')):
        for r in csv.DictReader(open(f)):
            if r['status'] != 'ok':
                continue
            shape = 'x'.join(v for v in (r['m'], r['n'], r['k']) if v)
            key = (r['regime'], r['family'], r['dtype'], r['order'], shape, r['batch'], r['row'], int(r['threads']), r['note'])
            cells[key].append(float(r['total_ms']))
    return cells

ap = argparse.ArgumentParser()
ap.add_argument('base'); ap.add_argument('cand')
ap.add_argument('--row', default=None, help='only this engine row (e.g. colmat)')
ap.add_argument('--csv')
a = ap.parse_args()
A, B = load(a.base), load(a.cand)
rows = []
for key in sorted(set(A) & set(B)):
    if a.row and key[6] != a.row:
        continue
    x, y = A[key], B[key]
    mx, my = statistics.median(x), statistics.median(y)
    sx, sy = (max(x) - min(x)) / mx, (max(y) - min(y)) / my
    ratio = mx / my
    sp = max(sx, sy)
    overlap = not (min(y) > max(x) or max(y) < min(x))
    verdict = 'regress' if (1 / ratio - 1 > sp and not overlap) else 'win' if (ratio - 1 > sp and not overlap) else 'flat'
    rows.append(dict(regime=key[0], family=key[1], dtype=key[2], order=key[3], shape=key[4], batch=key[5],
                     row=key[6], threads=key[7], variant=key[8], n_base=len(x), n_cand=len(y), base_ms=mx, v4_ms=my,
                     ratio=ratio, spread_base=sx, spread_v4=sy, verdict=verdict))
only_a = len(set(A) - set(B)); only_b = len(set(B) - set(A))
if a.csv:
    with open(a.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f'paired cells: {len(rows)}  (unpaired: {only_a} default-only, {only_b} v4-only)')
by = collections.defaultdict(list)
for r in rows:
    by[(r['row'], r['family'])].append(r)
print(f"{'row':16} {'family':18} {'cells':>5} {'geomean':>8} {'min':>6} {'max':>6} {'win':>4} {'flat':>4} {'regr':>4} {'win5':>5} {'reg5':>5}")
for (row, fam), rs in sorted(by.items()):
    g = statistics.geometric_mean(r['ratio'] for r in rs)
    c = collections.Counter(r['verdict'] for r in rs)
    print(f"{row:16} {fam:18} {len(rs):5} {g:8.3f} {min(r['ratio'] for r in rs):6.3f} {max(r['ratio'] for r in rs):6.3f} {c['win']:4} {c['flat']:4} {c['regress']:4} {sum(r['verdict']=='win' and r['ratio']>1.05 for r in rs):5} {sum(r['verdict']=='regress' and r['ratio']<1/1.05 for r in rs):5}")
reg = [r for r in rows if r['verdict'] == 'regress']
print(f'\nregressions beyond own spread: {len(reg)}; of them beyond the 5% verdict floor: {sum(r["ratio"]<1/1.05 for r in reg)}')
for r in reg:
    print(f"  {r['row']} {r['family']} {r['shape']} {r['dtype']} {r['order']} {r['threads']}T ratio {r['ratio']:.3f} "
          f"spread {r['spread_base']:.3f}/{r['spread_v4']:.3f}")
