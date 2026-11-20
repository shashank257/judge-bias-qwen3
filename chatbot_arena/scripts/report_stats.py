"""All numbers and figures for the midterm report, computed from the raw judge logs.

Writes report/stats.json, report/*.csv and report/fig_*.pdf (vector, grayscale-safe).
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dsc261 import analysis as AN, evaldata as E  # noqa: E402

OUT = os.path.join(ROOT, "report")
os.makedirs(OUT, exist_ok=True)

logs = [os.path.join(ROOT, "api", "pilot_local", "pilot_log.jsonl")] + sorted(
    glob.glob(os.path.join(ROOT, "api", "midterm_remote", "*.jsonl")))
v = pd.concat([AN.load_verdicts(f) for f in logs], ignore_index=True).drop_duplicates("request_id", keep="last")
pairs = pd.read_parquet(os.path.join(ROOT, "data", "processed", "pilot_pairs.parquet"))
v = v[v.pair_id.isin(pairs.pair_id)]
v = AN.main_runs(v)

CFG = [("0.6B", "0.6B", 0.6, "BF16", "MLX, M2 Pro"), ("1.7B", "1.7B", 1.7, "BF16", "MLX, M2 Pro"),
       ("4B-q8", "4B", 4.0, "8-bit", "MLX, M2 Pro"), ("8B-awq", "8B", 8.0, "4-bit AWQ", "vLLM, RTX 3070")]
rows, run_rows, gee_rows = [], [], []
human = pairs.set_index("pair_id").human_pref
for key, label, params, prec, backend in CFG:
    for think in (False, True):
        cfg = f"{key}-{'think' if think else 'nothink'}"
        g = v[v.config == cfg]
        if len(g) < 900:
            continue
        t = AN.per_pair_table(v, cfg)
        s = AN.summarize(t)
        s.update(AN.human_agreement(t, human))
        s.update({"config": cfg, "size": label, "params_b": params, "thinking": think, "precision": prec,
                  "backend": backend})
        rows.append(s)
        run_rows.append({"config": cfg, "size": label, "thinking": think, "calls": len(g),
                         "invalid": int(g.verdict.isna().sum()), "retried": int((g.n_attempts > 1).sum()),
                         "truncated": int(g.truncated.sum()),
                         "out_tokens_mean": g.out_tokens.mean(),
                         "think_tokens_mean": g.think_tokens.mean() if think else np.nan})
        # GEE input: one row per comparison (edited = order swap; repeat = identical input)
        d = g[(g.slot1_var == "orig") & (g.slot2_var == "orig")]
        w = {(r.pair_id, r.slot1, r.repeat): r.winner for r in d.itertuples()}
        for pid in t.pair_id:
            ab0, ba0 = w.get((pid, "a", 0)), w.get((pid, "b", 0))
            ab1, ba1 = w.get((pid, "a", 1)), w.get((pid, "b", 1))
            f = lambda x, y: int(x is None or y is None or x != y)  # noqa: E731
            gee_rows += [{"pair_id": pid, "edit": 1, "y": f(ab0, ba0), "log2size": np.log2(params), "think": int(think)},
                         {"pair_id": pid, "edit": 0, "y": f(ab0, ab1), "log2size": np.log2(params), "think": int(think)},
                         {"pair_id": pid, "edit": 0, "y": f(ba0, ba1), "log2size": np.log2(params), "think": int(think)}]

res = pd.DataFrame(rows).set_index("config")
runs = pd.DataFrame(run_rows).set_index("config")
res.to_csv(os.path.join(OUT, "results.csv"))
runs.to_csv(os.path.join(OUT, "runs.csv"))

# ---- H2-style GEE (exploratory: 100 pairs, precision differs by size) ----
gd = pd.DataFrame(gee_rows)
m = smf.gee("y ~ edit * (log2size + think)", groups="pair_id", data=gd,
            family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
terms = ["edit:log2size", "edit:think"]
p_holm = multipletests([m.pvalues[t] for t in terms], method="holm")[1]
gee = {t: {"coef": float(m.params[t]), "se": float(m.bse[t]), "p": float(m.pvalues[t]), "p_holm": float(ph)}
       for t, ph in zip(terms, p_holm)}


# average marginal effects (points) of size doubling and thinking on the excess flip probability
def excess(df):
    pe = m.predict(df.assign(edit=1)) - m.predict(df.assign(edit=0))
    return float(pe.mean())


base = gd.drop_duplicates(["pair_id", "log2size", "think"])
ame_size = (excess(base.assign(log2size=base.log2size + 0.5)) - excess(base.assign(log2size=base.log2size - 0.5))) * 100
ame_think = (excess(base.assign(think=1)) - excess(base.assign(think=0))) * 100
rng = np.random.default_rng(261)
boot_s, boot_t = [], []
pids = gd.pair_id.unique()
for _ in range(300):  # cluster bootstrap of the AMEs (refit), kept small for runtime
    pick = rng.choice(pids, len(pids))
    bd = pd.concat([gd[gd.pair_id == p].assign(pair_id=f"{p}#{i}") for i, p in enumerate(pick)])
    try:
        mb = smf.gee("y ~ edit * (log2size + think)", groups="pair_id", data=bd, family=sm.families.Binomial(),
                     cov_struct=sm.cov_struct.Independence()).fit()
    except Exception:
        continue
    bb = bd.drop_duplicates(["pair_id", "log2size", "think"])
    ex = lambda df: float((mb.predict(df.assign(edit=1)) - mb.predict(df.assign(edit=0))).mean())  # noqa: E731
    boot_s.append((ex(bb.assign(log2size=bb.log2size + 0.5)) - ex(bb.assign(log2size=bb.log2size - 0.5))) * 100)
    boot_t.append((ex(bb.assign(think=1)) - ex(bb.assign(think=0))) * 100)
gee["ame_size_per_doubling_pts"] = [ame_size, *np.percentile(boot_s, [2.5, 97.5]).tolist()]
gee["ame_thinking_pts"] = [ame_think, *np.percentile(boot_t, [2.5, 97.5]).tolist()]

# ---- controls and context ----
ctrl = {"random": E.agreement(E.random_verdicts(len(pairs)), pairs.human_pref),
        "length": E.agreement(E.length_verdicts(pairs.len_a_words, pairs.len_b_words), pairs.human_pref)}
w = pd.concat([pairs.len_a_words, pairs.len_b_words])
counts = pd.read_csv(os.path.join(ROOT, "reports", "arena_filter_counts.csv"))
mdcorr = pd.read_csv(os.path.join(ROOT, "reports", "human_markdown_correlation.csv"))
stats = {"n_pairs": int(len(pairs)), "by_source": pairs.source.value_counts().to_dict(),
         "words_median": float(w.median()), "words_min": int(w.min()), "words_max": int(w.max()),
         "prompt_tokens_mean": float(pairs.prompt_tokens_md.mean()), "prompt_tokens_max": int(pairs.prompt_tokens_md.max()),
         "human_A_share": float((pairs.human_pref == "A").mean()), "controls": ctrl, "gee": gee,
         "total_calls": int(runs.calls.sum()), "total_invalid": int(runs.invalid.sum())}
json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=2, default=float)
counts.to_csv(os.path.join(OUT, "filter_counts.csv"), index=False)
mdcorr.to_csv(os.path.join(OUT, "md_corr.csv"), index=False)

# ---- figures: serif, black/grey, markers distinguish modes ----
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
sizes = [c[1] for c in CFG]
xt = np.arange(len(sizes))
xl = [f"{c[1]}\n({c[3]})" for c in CFG]


def series(col, think):
    r = res[res.thinking == think].set_index("size").reindex(sizes)
    return r[col] * 100, r[f"{col}_lo"] * 100, r[f"{col}_hi"] * 100


def lineplot(ax, col, ref, ref_label, ylabel, title):
    for think, mk, ls, off in ((False, "o", "-", -0.06), (True, "s", "--", 0.06)):
        y, lo, hi = series(col, think)
        ax.errorbar(xt + off, y, yerr=[y - lo, hi - y], marker=mk, ls=ls, color="black",
                    mfc="white" if think else "black", capsize=3, lw=1, ms=5,
                    label="thinking on" if think else "thinking off")
    ax.axhline(ref, color="grey", lw=0.8, ls=":", label=ref_label)
    ax.set_xticks(xt); ax.set_xticklabels(xl); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_xlim(-0.4, len(sizes) - 0.6)


fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
lineplot(axes[0], "pos_excess", 2, "H1 threshold (2 pts)", "excess flip rate (points)", "(a) Position: order swapped")
lineplot(axes[1], "fmt_excess", 2, "H1 threshold (2 pts)", "excess flip rate (points)", "(b) Format: markdown removed")
axes[0].legend(frameon=False, loc="upper right", fontsize=7.5)  # one legend; same encoding in (b)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_excess.pdf")); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
lineplot(axes[0], "slot1_pref", 50, "no preference", "verdicts for first answer (%)", "(a) Share choosing the first-shown answer")
lineplot(axes[1], "md_pref", 50, "no preference", "verdicts for markdown answer (%)", "(b) Share choosing the markdown version")
axes[0].legend(frameon=False, loc="lower right", fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_preference.pdf")); plt.close(fig)
print(json.dumps({"gee": gee, "controls": ctrl}, indent=1, default=float))
print(res[["pos_excess", "fmt_excess", "slot1_pref", "md_pref", "human_acc", "human_kappa"]].round(3))
print(runs)
