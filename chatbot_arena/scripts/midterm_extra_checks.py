"""Extra midterm numbers: McNemar tests of each judge against the length rule,
and a pair-bootstrap CI for the thinking effect with and without 0.6B.
Run from chatbot_arena/ (same logs and helpers as scripts/report_stats.py)."""
import glob
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, "src")
from dsc261 import analysis as AN, evaldata as E  # noqa: E402

logs = ["api/pilot_local/pilot_log.jsonl"] + sorted(glob.glob("api/midterm_remote/*.jsonl"))
v = pd.concat([AN.load_verdicts(f) for f in logs], ignore_index=True).drop_duplicates("request_id", keep="last")
pairs = pd.read_parquet("data/processed/pilot_pairs.parquet")
v = AN.main_runs(v[v.pair_id.isin(pairs.pair_id)])
human = pairs.set_index("pair_id").human_pref.map({"A": "a", "B": "b"})
length = pd.Series(E.length_verdicts(pairs.len_a_words, pairs.len_b_words), index=pairs.pair_id).map({"A": "a", "B": "b"})

sizes = ["0.6B", "1.7B", "4B-q8", "8B-awq"]
cfgs = [f"{k}-{m}" for k in sizes for m in ("nothink", "think")]
tabs = {c: AN.per_pair_table(v, c).set_index("pair_id") for c in cfgs}

rows = []
for c in cfgs:
    t = tabs[c]
    j = t.winner_ab0 == human.reindex(t.index)
    l = length.reindex(t.index) == human.reindex(t.index)
    table = [[int((j & l).sum()), int((j & ~l).sum())], [int((~j & l).sum()), int((~j & ~l).sum())]]
    rows.append((c, j.mean(), l.mean(), mcnemar(table, exact=True).pvalue))
holm = multipletests([r[3] for r in rows], method="holm")[1]
print("McNemar, judge vs longer-answer rule (exact):")
for (c, acc, lacc, p), ph in zip(rows, holm):
    print(f"  {c:15s} judge {acc:.2f}  length {lacc:.2f}  p={p:.3f}  Holm p={ph:.3f}")

idx = tabs[cfgs[0]].index
for c in cfgs:
    idx = idx.intersection(tabs[c].index)
exc = {c: (tabs[c].pos_flip - tabs[c].noise_flip).reindex(idx).to_numpy() for c in cfgs}
D = np.column_stack([exc[f"{k}-think"] - exc[f"{k}-nothink"] for k in sizes])
rng = np.random.default_rng(261)
boot = np.array([D[rng.integers(0, len(idx), len(idx))].mean(0) for _ in range(10000)])
for name, cols in (("all sizes", slice(None)), ("excluding 0.6B", slice(1, None))):
    est = 100 * D.mean(0)[cols].mean()
    lo, hi = 100 * np.percentile(boot[:, cols].mean(1), [2.5, 97.5])
    print(f"Thinking effect on position excess, {name}: {est:.1f} points [{lo:.1f}, {hi:.1f}]")

# Numbers added in the revised midterm text.
# The size main effect (log2size, beta = -0.26) comes from the GEE in scripts/report_stats.py
# (m.params["log2size"]); that script saves only the interaction terms to stats.json.

x = (tabs["8B-awq-think"].pos_flip - tabs["8B-awq-think"].noise_flip).to_numpy()
rng = np.random.default_rng(261)
boot8 = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(10000)]
print(f"8B thinking position excess, Holm over two biases (97.5% CI): "
      f"[{100*np.percentile(boot8, 1.25):.1f}, {100*np.percentile(boot8, 98.75):.1f}]")

pool = pd.read_parquet("data/processed/format_pool.parquet")
used = pairs.pair_id.isin(pool.pair_id).sum()
print(f"Format-pool pairs not used in the midterm: {len(pool) - used} (of {len(pool)}; {used} midterm pairs are in the pool)")
