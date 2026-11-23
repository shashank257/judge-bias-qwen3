#!/usr/bin/env python3
"""Summarize paired judge outcomes from run_judges.py JSONL, with question-cluster CIs."""
import argparse
import collections
import json
import random
import statistics
from pathlib import Path


def metric(rows):
    lookup = {(r['condition'],r['order'],r['replicate']):r['judge_winner_original'] for r in rows}
    def get(c,o,r=0): return lookup.get((c,o,r))
    baseline='A_plain__B_plain'
    noise=[]; position=[]; flips=[]; favor=[]; human=[]
    for order in ('AB','BA'):
        x,y=get(baseline,order),get(baseline,order,1)
        if x and y: noise.append(int(x!=y))
        if x:
            hr=rows[0].get('human_winner')
            if hr in ('A','B'): human.append(int(x==hr))
        for mod in ('A_markdown__B_plain','A_plain__B_markdown'):
            m=get(mod,order)
            if m and x: flips.append(int(m!=x))
            if m:
                md_answer='A' if mod.startswith('A_markdown') else 'B'
                favor.append(int(m==md_answer))
    a,b=get(baseline,'AB'),get(baseline,'BA')
    if a and b: position.append(int(a!=b))
    return dict(noise=noise,position=position,format_flip=flips,markdown_preference=favor,human_agreement=human)


def average(pairs, field):
    vals=[v for p in pairs for v in p[field]]
    return sum(vals)/len(vals) if vals else None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('logs', type=Path)
    ap.add_argument('--seed',type=int,default=261)
    ap.add_argument('--bootstrap',type=int,default=2000)
    args=ap.parse_args()
    # Retried jobs may have earlier errors; use latest successful result per job.
    latest={}
    for line in args.logs.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        if r.get('status')=='ok': latest[r['job_id']]=r
    by_pair=collections.defaultdict(list)
    for row in latest.values():
        by_pair[(row['model_requested'], row['pair_id'])].append(row)
    by_model=collections.defaultdict(list)
    for (model,pid),rows in by_pair.items():
        by_model[model].append((str(rows[0]['question_id']), metric(rows)))
    rng=random.Random(args.seed)
    for model,entries in by_model.items():
        cluster=collections.defaultdict(list)
        for qid,metrics in entries: cluster[qid].append(metrics)
        groups=list(cluster.values())
        print(f'\nModel: {model} | pairs={len(entries)}, question clusters={len(groups)}, successful calls={sum(len(by_pair[(model,pid)]) for m,pid in by_pair if m==model)}')
        for field in ('position','noise','format_flip','markdown_preference','human_agreement'):
            observed=average([p for _,p in entries],field)
            if observed is None:
                print(f'  {field}: unavailable'); continue
            boot=[]
            for _ in range(args.bootstrap):
                sampled=[p for g in rng.choices(groups,k=len(groups)) for p in g]
                v=average(sampled,field)
                if v is not None: boot.append(v)
            boot.sort()
            lo=boot[int(0.025*(len(boot)-1))];hi=boot[int(0.975*(len(boot)-1))]
            print(f'  {field}: {100*observed:.1f}% (cluster bootstrap 95% CI {100*lo:.1f}–{100*hi:.1f}%)')
        print('  Caution: preliminary descriptive pilot. Format flip comparisons share baselines and are not independent.')

if __name__=='__main__': main()
