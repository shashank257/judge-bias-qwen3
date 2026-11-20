from nb_common import build

cells = [
r"""# 06 — Midterm results: position and format bias on 100 Arena pairs
The proposal's full design on the 100 frozen pilot pairs (from the format pool, so both answers use markdown),
Qwen3 **0.6B and 1.7B** (BF16) plus **4B and 8B as 8-bit quantized models ("-q8")**, thinking **off and on**,
10 judge calls per pair:
original/plain × original/plain in both orders, plus the original pair repeated in both orders (noise floor).
Run locally (MLX, same pinned checkpoints) with `scripts/run_midterm.py`. Measures follow
`docs/evaluation_contract.md` §5. **Precision per size: 0.6B/1.7B BF16; "4B-q8" = 8-bit MLX; "8B-awq" = Qwen's official 4-bit AWQ on vLLM
(contract specifies BF16). Size comparisons therefore mix precision — a stated confound.
Scope: 100 of 1,000 pairs — an interim result, not a test of H1–H3 as pre-registered.**""",

r"""from dsc261 import analysis as AN, evaldata as E
import glob
logs = [C.path("api", "pilot_local", "pilot_log.jsonl")] + sorted(glob.glob(C.path("api", "midterm_remote", "*.jsonl")))
v = pd.concat([AN.load_verdicts(f) for f in logs], ignore_index=True).drop_duplicates("request_id", keep="last")
print("logs read:", [os.path.relpath(f, ROOT) for f in logs])
pairs = pd.read_parquet(C.path("data", "processed", "pilot_pairs.parquet"))
v = v[v.pair_id.isin(pairs.pair_id)]
ORDER = ["0.6B", "1.7B", "4B", "4B-q8", "8B", "8B-q8", "8B-awq"]
CONFIGS = [f"{s}-{m}" for s in ORDER for m in ("nothink", "think")
           if (v.config == f"{s}-{m}").sum() >= 10 * len(pairs) * 0.9]  # configs that are (nearly) complete
v.groupby("config").agg(requests=("request_id", "size"), invalid=("verdict", lambda s: s.isna().mean()),
                        retried=("n_attempts", lambda s: (s > 1).mean()), truncated=("truncated", "mean"),
                        out_tokens=("out_tokens", "mean"), latency_s=("latency_s", "mean")).reindex(CONFIGS).round(3)""",

r"""MD:## 1. Position and format bias per configuration
- **Flip rate**: share of pairs whose winning answer changes when only the order (position) or only the markdown
  (format) changes. **Noise floor**: flips between two runs on identical input. **Excess** = flip − noise (points),
  95% bootstrap CI over pairs (10,000 resamples, seed 261).
- **Slot-1 preference**: share of verdicts for the answer shown first (0.5 = no position bias).
- **Markdown preference**: in cells where exactly one answer keeps its markdown, share of verdicts for that
  answer (0.5 = no format bias).""",

r"""tables = {c: AN.per_pair_table(v, c) for c in CONFIGS}
res = pd.DataFrame({c: AN.summarize(t) for c, t in tables.items()}).T
human = pairs.set_index("pair_id").human_pref
for c, t in tables.items():
    if len(t):
        for k, x in AN.human_agreement(t, human).items():
            res.loc[c, k] = x
res = res.astype(float)
res.to_csv(C.path("reports", "midterm_results.csv"))
pct = [c for c in res.columns if c not in ("pairs", "fmt_pairs", "human_kappa")]
show = res.copy(); show[pct] = (show[pct] * 100).round(1); show["human_kappa"] = show["human_kappa"].round(3)
show""",

r"""MD:## 2. Controls on the same 100 pairs""",

r"""ctrl = pd.DataFrame([
    {"control": "random verdict", **E.agreement(E.random_verdicts(len(pairs)), pairs.human_pref)},
    {"control": "length-only (longer wins)", **E.agreement(E.length_verdicts(pairs.len_a_words, pairs.len_b_words), pairs.human_pref)},
]).round(3)
ctrl""",

r"""MD:## 3. Figure""",

r"""import matplotlib.pyplot as plt
ok = res.dropna(subset=["pos_excess"])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
for ax, (m, lo, hi, title) in zip(axes, [("pos_excess", "pos_excess_lo", "pos_excess_hi", "Position: excess flip rate"),
                                         ("fmt_excess", "fmt_excess_lo", "fmt_excess_hi", "Format: excess flip rate")]):
    if m not in ok:
        continue
    y = ok[m] * 100
    ax.bar(range(len(ok)), y, color=["#8fb3d9" if "nothink" in c else "#2f6db3" for c in ok.index])
    ax.errorbar(range(len(ok)), y, yerr=[y - ok[lo] * 100, ok[hi] * 100 - y], fmt="none", ecolor="#333", capsize=4)
    ax.axhline(2, ls="--", lw=1, color="#c0392b"); ax.text(len(ok) - 0.5, 2.5, "H1 threshold (2 pts)", ha="right", fontsize=8, color="#c0392b")
    ax.set_xticks(range(len(ok))); ax.set_xticklabels(ok.index, rotation=20, fontsize=9)
    ax.set_title(title, fontsize=11); ax.set_ylabel("points (flip − noise)")
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(C.path("reports", "midterm_excess_flip.png"), dpi=150); plt.show()""",

r"""C.log_version("06_midterm", "position+format bias on 100 pilot pairs, 0.6B/1.7B x think on/off",
              [C.path("reports", "midterm_results.csv")])
print("complete pairs per config:", {c: len(t) for c, t in tables.items()})""",
]

build(cells, "notebooks/06_midterm_results.ipynb", "06_midterm_results")
