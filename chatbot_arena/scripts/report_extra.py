"""Robustness analyses requested by the report reviewers. Writes report/extra.json."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dsc261 import analysis as AN  # noqa: E402

logs = [os.path.join(ROOT, "api", "pilot_local", "pilot_log.jsonl")] + sorted(
    glob.glob(os.path.join(ROOT, "api", "midterm_remote", "*.jsonl")))
v = pd.concat([AN.load_verdicts(f) for f in logs], ignore_index=True).drop_duplicates("request_id", keep="last")
pairs = pd.read_parquet(os.path.join(ROOT, "data", "processed", "pilot_pairs.parquet"))
v = v[v.pair_id.isin(pairs.pair_id)]
v_all = v  # includes sampling-check runs (used only for that check)
v = AN.main_runs(v)
fz = json.load(open(os.path.join(ROOT, "data", "processed", "freeze_manifest.json")))
flagged = set(fz["audit_replacements"]["removed"])
human = pairs.set_index("pair_id").human_pref
src = pairs.set_index("pair_id").source
CFGS = ["0.6B-nothink", "0.6B-think", "1.7B-nothink", "1.7B-think", "4B-q8-nothink", "4B-q8-think",
        "8B-awq-nothink", "8B-awq-think"]
rng = np.random.default_rng(261)
out = {"flagged_in_subset": sorted(flagged & set(pairs.pair_id))}


def boot(x, n=10000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    idx = rng.integers(0, len(x), (n, len(x)))
    s = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)), s


tables = {c: AN.per_pair_table(v, c) for c in CFGS}
rows = {}
md_p = []
for c, t in tables.items():
    r = {}
    # 1) without the two audit-flagged pairs
    t98 = t[~t.pair_id.isin(flagged)]
    r["pos_excess_98"] = float((t98.pos_flip - t98.noise_flip).mean())
    r["fmt_excess_98"] = float((t98.fmt_flip - t98.noise_flip).mean())
    # 2) by source
    for s in ("arena2023", "arena2025"):
        ts = t[t.pair_id.map(src) == s]
        m, lo, hi, _ = boot((ts.pos_flip - ts.noise_flip).to_numpy())
        r[f"pos_excess_{s}"] = [m, lo, hi, int(len(ts))]
    # 3) human accuracy CI
    h = t.pair_id.map(human).map({"A": "a", "B": "b"})
    m, lo, hi, _ = boot((t.winner_ab0 == h).astype(float).to_numpy())
    r["human_acc"] = [m, lo, hi]
    # 4) markdown preference: two-sided bootstrap p-value vs 0.5
    m, lo, hi, s_ = boot(t.md_pref.to_numpy())
    p = float(min(1.0, 2 * min((s_ <= 0.5).mean(), (s_ >= 0.5).mean())))
    r["md_pref"] = [m, lo, hi, p]
    md_p.append(p)
    # 5) independence of repeats: identical full outputs between repeat 0 and 1 (same order)
    g = v[(v.config == c) & (v.slot1_var == "orig") & (v.slot2_var == "orig")]
    rows[c] = r
holm = multipletests(md_p, method="holm")[1]
for c, ph in zip(CFGS, holm):
    rows[c]["md_pref"].append(float(ph))

# 5) repeat independence from raw outputs
raw = {}
for f in logs:
    for line in open(f):
        x = json.loads(line)
        if x.get("error") is None and x.get("verdict"):
            raw[x["request_id"]] = x.get("output", "")
ident = {}
for c in CFGS:
    same = tot = 0
    for rid, o in raw.items():
        if f"|{c}|" in rid and "|s=think|" not in rid.replace("-think|s=think", "") and "orig>" in rid and rid.endswith("|r0") and rid.count(":orig") == 2:
            o1 = raw.get(rid[:-1] + "1")
            if o1 is not None:
                tot += 1
                same += int(o.strip() == o1.strip())
    ident[c] = [same, tot]
out["identical_repeat_outputs"] = ident

# 6) thinking effect per size (difference of excess flips; paired bootstrap over pairs)
eff = {}
for size in ("0.6B", "1.7B", "4B-q8", "8B-awq"):
    a, b = tables[f"{size}-nothink"], tables[f"{size}-think"]
    mm = a.merge(b, on="pair_id", suffixes=("_off", "_on"))
    d = (mm.pos_flip_on - mm.noise_flip_on) - (mm.pos_flip_off - mm.noise_flip_off)
    m, lo, hi, _ = boot(d.to_numpy())
    eff[size] = [m, lo, hi]
out["thinking_effect_by_size"] = eff

# 7) GEE without 0.6B (thinking AME), p reported to sensible precision
gee_rows = []
size_of = {"0.6B": 0.6, "1.7B": 1.7, "4B-q8": 4.0, "8B-awq": 8.0}
for c, t in tables.items():
    size, mode = c.rsplit("-", 1)
    g = v[(v.config == c) & (v.slot1_var == "orig") & (v.slot2_var == "orig")]
    w = {(r.pair_id, r.slot1, r.repeat): r.winner for r in g.itertuples()}
    for pid in t.pair_id:
        ab0, ba0, ab1, ba1 = (w.get((pid, s_, k)) for s_, k in (("a", 0), ("b", 0), ("a", 1), ("b", 1)))
        f = lambda x, y: int(x is None or y is None or x != y)  # noqa: E731
        base = {"pair_id": pid, "log2size": np.log2(size_of[size]), "think": int(mode == "think"), "size": size}
        gee_rows += [{**base, "edit": 1, "y": f(ab0, ba0)}, {**base, "edit": 0, "y": f(ab0, ab1)},
                     {**base, "edit": 0, "y": f(ba0, ba1)}]
gd = pd.DataFrame(gee_rows)
sub = gd[gd["size"] != "0.6B"]
m = smf.gee("y ~ edit * (log2size + think)", groups="pair_id", data=sub, family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable()).fit()
b = sub.drop_duplicates(["pair_id", "log2size", "think"])
ex = lambda df: float((m.predict(df.assign(edit=1)) - m.predict(df.assign(edit=0))).mean())  # noqa: E731
out["gee_without_0.6B"] = {"edit:think_coef": float(m.params["edit:think"]), "edit:think_p": float(m.pvalues["edit:think"]),
                           "edit:log2size_coef": float(m.params["edit:log2size"]),
                           "edit:log2size_p": float(m.pvalues["edit:log2size"]),
                           "ame_thinking_pts": (ex(b.assign(think=1)) - ex(b.assign(think=0))) * 100}
out["per_config"] = rows
json.dump(out, open(os.path.join(ROOT, "report", "extra.json"), "w"), indent=1)
print(json.dumps(out, indent=1))

# 8) Bonferroni-adjusted (8 configs) CIs for position excess; format cancellation; Wilson CIs; sampling check
from statsmodels.stats.proportion import proportion_confint  # noqa: E402
adj = {}
for c, t in tables.items():
    x = (t.pos_flip - t.noise_flip).to_numpy()
    idx = rng.integers(0, len(x), (10000, len(x)))
    s_ = x[idx].mean(1)
    a = 0.05 / 8 / 2 * 100
    acc = rows[c]["human_acc"][0]
    lo, hi = proportion_confint(round(acc * len(t)), len(t), method="wilson")
    # format cells split by whether the stripped answer was the baseline winner
    g = v[(v.config == c) & (v.sampling == ("think" if c.endswith("-think") else "nothink"))]
    w = {(r.pair_id, r.slot1, r.slot1_var, r.slot2_var, r.repeat): r.winner for r in g.itertuples()}
    win_flip, lose_flip = [], []
    for pid in t.pair_id:
        for first in ("a", "b"):
            second = "b" if first == "a" else "a"
            base = w.get((pid, first, "orig", "orig", 0))
            for stripped, key in ((first, (pid, first, "plain", "orig", 0)), (second, (pid, first, "orig", "plain", 0))):
                new = w.get(key)
                if base is None or new is None:
                    continue
                (win_flip if stripped == base else lose_flip).append(int(new != base))
    adj[c] = {"pos_excess_bonf": [float(np.percentile(s_, a)), float(np.percentile(s_, 100 - a))],
              "human_wilson": [lo, hi],
              "flip_strip_winner": float(np.mean(win_flip)), "flip_strip_loser": float(np.mean(lose_flip))}
out["adjusted"] = adj
sc = {}
for size in ("0.6B", "1.7B"):
    t = AN.per_pair_table(v_all, f"{size}-nothink", sampling="think")
    if len(t) >= 90:
        s = AN.summarize(t)
        sc[size] = {k: s[k] for k in ("pairs", "pos_flip_rate", "noise_floor", "pos_excess", "pos_excess_lo",
                                       "pos_excess_hi", "slot1_pref", "md_pref", "md_pref_lo", "md_pref_hi")}
out["sampling_check"] = sc
json.dump(out, open(os.path.join(ROOT, "report", "extra.json"), "w"), indent=1)
print(json.dumps({"adjusted": adj, "sampling_check": sc}, indent=1))
