"""Tie-aware H3 format drift for the UltraFeedback pilot.

The released drift compares the judge's pick with one tie-broken gold answer.
Here the gold markdown rate is averaged over all answers tied for the best
GPT-4 proxy score. Run from the ultrafeedback/ folder of the repository.
"""
import json
import numpy as np
import pandas as pd

rows = {r["prompt_id"]: r for r in map(json.loads, open("data/selected_prompts.jsonl"))}
d = pd.read_csv("tables/merge_prompt_outcomes.csv")
d4 = d[(d.n == 4) & (~d.missing)].copy()

def md_of(pid):
    return {a["answer_id"]: float(a["markdown"]) for a in rows[pid]["answers"]}

def tied_best_md(pid):
    q = {a["answer_id"]: a["quality"] for a in rows[pid]["answers"]}
    best = [k for k, v in q.items() if v == max(q.values())]
    md = md_of(pid)
    return np.mean([md[k] for k in best])

d4["sel_md"] = [md_of(p)[str(int(s))] for p, s in zip(d4.prompt_id, d4.selected_id)]
d4["gold_md"] = [md_of(p)[str(int(g))] for p, g in zip(d4.prompt_id, d4.gold_id)]
assert np.allclose(d4.sel_md - d4.gold_md, d4.format_drift)  # reproduces released drift
d4["drift_tie"] = d4.sel_md - d4.prompt_id.map(tied_best_md)

def cluster_boot(df, col, B=10000, seed=261):
    rng = np.random.default_rng(seed)
    pids = df.prompt_id.unique()
    groups = {p: df.loc[df.prompt_id == p, col].to_numpy() for p in pids}
    stats = [np.concatenate([groups[p] for p in rng.choice(pids, len(pids))]).mean() for _ in range(B)]
    return df[col].mean(), np.percentile(stats, [2.5, 97.5])

for col in ["format_drift", "drift_tie"]:
    est, (lo, hi) = cluster_boot(d4, col)
    print(f"{col}: {100*est:+.1f} pp, 95% CI [{100*lo:+.1f}, {100*hi:+.1f}], rows={len(d4)}")
