from nb_common import build

cells = [
r"""# 03 — Remaining evaluation data and controls (Week 1)
MT-Bench expert votes (human agreement), random and length-only baselines, the markdown ↔ human-vote check, and
the UltraFeedback best-of-n candidate set (500 prompts, gold proxy, brackets for n = 1, 2, 4).""",

r"""MD:## 1. MT-Bench expert votes: load and check alignment""",

r"""import json
from dsc261 import evaldata as E, arena as A
log = A.CountLog()
mt_files = C.download("mtbench_human")
mt, checks = E.load_mtbench(mt_files[0], log)
C.record_dataset("mtbench_human", mt_files, int(log.rows[0]["retained"]), "MT-Bench human (expert) pairwise votes")
print(json.dumps(checks, indent=1, default=str))
assert checks["user_turn1_identical_a_b"] and checks["one_question_text_per_question_id"] and checks["roles_alternate_user_assistant"]
assert checks["n_question_ids"] == 80
log.frame()""",

r"""MD:## 2. Human-agreement labels
One row per (question, model pair, judge) vote, turn 1, ties removed. `pair_key` groups votes on the same
question and model pair so agreement can also be computed against the per-pair majority.""",

r"""mt_labels = mt[["vote_id", "prompt_id", "question_id", "pair_key", "model_a", "model_b", "judge", "question",
                "answer_a", "answer_b", "human_pref", "len_a_words", "len_b_words", "md_a", "md_b"]]
mt_labels.to_parquet(C.path("data", "processed", "mtbench_human_labels.parquet"), index=False)
# majority by winning *model name* (A/B is relative to each vote's own model order)
mt_labels = mt_labels.assign(winner_model=np.where(mt_labels.human_pref == "A", mt_labels.model_a, mt_labels.model_b))
def _majority(s):
    vc = s.value_counts()
    return vc.index[0] if len(vc) == 1 or vc.iloc[0] > vc.iloc[1] else None
maj = mt_labels.groupby("pair_key").winner_model.agg(_majority)
maj.rename("majority_winner_model").to_frame().to_parquet(C.path("data", "processed", "mtbench_pair_majority.parquet"))
print("turn-1 non-tie votes:", len(mt_labels), "| distinct (question, model pair):", mt_labels.pair_key.nunique(),
      "| questions:", mt_labels.question_id.nunique(), "| judges:", mt_labels.judge.nunique())
print("pairs with a clear majority:", int(maj.notna().sum()))
arena_pos = pd.read_parquet(C.path("data", "processed", "position_pool.parquet"))
arena_fmt = pd.read_parquet(C.path("data", "processed", "format_pool.parquet"))
for name, d in (("position_pool", arena_pos), ("format_pool", arena_fmt)):
    d[["pair_id", "prompt_id", "source", "human_pref"]].to_parquet(C.path("data", "processed", f"human_labels_{name}.parquet"), index=False)
print("Arena human labels saved for both pools; A/B balance:",
      arena_pos.human_pref.value_counts(normalize=True).round(3).to_dict(), arena_fmt.human_pref.value_counts(normalize=True).round(3).to_dict())""",

r"""MD:## 3. Control baselines: random verdicts and a length-only predictor
Random: seeded coin (seed 261). Length-only: the longer answer (in words, markdown markers not counted) wins.
These are the floors any judge's human agreement is compared against (accuracy and Cohen's κ).""",

r"""base = []
for name, d in (("arena_position_pool", arena_pos), ("arena_format_pool", arena_fmt), ("mtbench_turn1", mt_labels)):
    base.append({"data": name, "baseline": "random", **E.agreement(E.random_verdicts(len(d)), d.human_pref)})
    base.append({"data": name, "baseline": "length_only", **E.agreement(E.length_verdicts(d.len_a_words, d.len_b_words), d.human_pref)})
base = pd.DataFrame(base).round(3)
base.to_csv(C.path("reports", "baselines.csv"), index=False)
base""",

r"""MD:## 4. Do human votes correlate with markdown?
Threat-to-validity check from the proposal: human voters saw *rendered* markdown. On every eligible Arena pair
where exactly one answer uses markdown, how often did the human pick the markdown answer? Then a logistic
regression of the human vote on the markdown difference, controlling for log length ratio, with standard errors
clustered by prompt.""",

r"""import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
el = pd.concat([pd.read_parquet(C.path("data", "processed", "arena_eligible_2023.parquet")),
                pd.read_parquet(C.path("data", "processed", "arena_eligible_2025.parquet"))], ignore_index=True)
el["y"] = (el.human_pref == "A").astype(int)
el["md_diff"] = el.md_a.astype(int) - el.md_b.astype(int)
el["log_len_ratio"] = np.log(el.len_a_words / el.len_b_words)
rows = []
for src, d in el.groupby("source"):
    one = d[d.md_diff != 0]
    k = int(((one.md_diff == 1) & (one.y == 1)).sum() + ((one.md_diff == -1) & (one.y == 0)).sum())
    lo, hi = proportion_confint(k, len(one), method="wilson")
    m = smf.logit("y ~ md_diff + log_len_ratio", data=d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.prompt_id)[0]})
    rows.append({"source": src, "pairs_exactly_one_md": len(one), "human_picks_md_share": k / len(one), "ci95_lo": lo, "ci95_hi": hi,
                 "logit_md_coef": m.params["md_diff"], "md_coef_p": m.pvalues["md_diff"], "md_odds_ratio": np.exp(m.params["md_diff"]),
                 "logit_loglen_coef": m.params["log_len_ratio"]})
mdcorr = pd.DataFrame(rows).round(4)
mdcorr.to_csv(C.path("reports", "human_markdown_correlation.csv"), index=False)
mdcorr""",

r"""MD:## 5. UltraFeedback best-of-n candidates (pinned revision)
Gold proxy = **mean of the four fine-grained GPT-4 ratings** (helpfulness, honesty, instruction following,
truthfulness), recomputed by us (not the `overall_score`, which has a documented labelling error).
**Missing ratings:** any `N/A` → the prompt is ineligible (no imputation). **Ties in gold:** regret needs no tie
rule; for format drift the gold pick's markdown rate is the average over all answers tied at the best gold score.""",

r"""uf_files = C.download("ultrafeedback")
paths = {os.path.basename(f).replace(".jsonl", ""): f for f in uf_files}
uf = E.load_ultrafeedback(paths, log)
n_uf = next(r["retained"] for r in log.rows if r["source"] == "ultrafeedback" and r["step"] == "loaded")
C.record_dataset("ultrafeedback", uf_files, n_uf, "UltraFeedback: sharegpt, ultrachat, evol_instruct files")
ans = E.expand_answers(uf)
print("gold recomputed == provided fine-grained_score:", float(np.isclose(ans.gold, ans.fg_score_provided.astype(float)).mean()))
from dsc261 import judge as J
tok = J.get_tokenizer("8B")
def longest_pair_tokens(g):
    two = g.nlargest(2, "n_words").response.tolist()
    q = uf.set_index("uf_id").at[g.name, "instruction"]
    return len(tok(J.render_prompt(J.build_messages(q, two[0], two[1]), "8B", False), add_special_tokens=False)["input_ids"])
ptoks = ans.groupby("uf_id").apply(longest_pair_tokens)
uf["max_pair_prompt_tokens"] = uf.uf_id.map(ptoks)
uf = log.step("ultrafeedback", "context_fit", uf, uf[uf.max_pair_prompt_tokens <= 6000], "longest possible comparison <= 6000 tokens")
log.frame()[log.frame().source == "ultrafeedback"]""",

r"""sel = uf.sample(500, random_state=C.SEED).sort_values("uf_id").reset_index(drop=True)
sel_ans = ans[ans.uf_id.isin(sel.uf_id)].reset_index(drop=True)
assert sel_ans.groupby("uf_id").size().eq(4).all() and sel_ans[[f"r_{a}" for a in E.ASPECTS]].notna().all().all()
print("selected by source:", sel.source.value_counts().to_dict())
print("answers with markdown:", round(sel_ans.has_md.mean(), 3), "| with code/table:", round(sel_ans.has_code_or_table.mean(), 3))
print("prompts containing a GPT-4 answer:", int(sel_ans.groupby("uf_id").is_gpt4.any().sum()))
print("gold distribution:", sel_ans.gold.describe().round(2).to_dict())
print("answer models:", sel_ans.model.value_counts().to_dict())""",

r"""MD:## 6. Brackets for best-of-n (n = 1, 2, 4) and the GPT-4 sensitivity subset
Seeded slot permutation per prompt. n = 1 → slot 0; n = 2 → winner(slot 0, slot 1);
n = 4 → winner(winner(slot 0, slot 1), winner(slot 2, slot 3)). Each match is judged in both orders (3 matches ×
2 = 6 calls per prompt and configuration). A split decision goes to the answer with more valid wins, then to a
deterministic coin on (prompt, pair) that is the same for every configuration (`evaldata.resolve_match`).""",

r"""br = E.make_brackets(sel, sel_ans)
br.to_parquet(C.path("data", "processed", "uf_brackets.parquet"), index=False)
sel_ans.to_parquet(C.path("data", "processed", "uf_answers.parquet"), index=False)
sel[["uf_id", "source", "file", "instruction", "max_pair_prompt_tokens"]].to_parquet(C.path("data", "processed", "uf_prompts.parquet"), index=False)
no_gpt4 = br[~br.any_gpt4].uf_id
no_gpt4.to_frame().to_parquet(C.path("data", "processed", "uf_sensitivity_no_gpt4.parquet"), index=False)
print("brackets:", len(br), "| sensitivity subset (no GPT-4 answer among the 4):", len(no_gpt4))
print("gold ties at the top (n=4):", br.n_gold_tied_n4.value_counts().to_dict())
print("prompts where markdown status varies across the 4 answers:", int(sel_ans.groupby("uf_id").has_md.nunique().gt(1).sum()))
# sanity: resolve_match rule
print(E.resolve_match("A", "B", "uf:x", "ans0", "ans1"), E.resolve_match("A", "A", "uf:x", "ans0", "ans1"), E.resolve_match(None, "A", "uf:x", "ans0", "ans1"))
pd.DataFrame(log.rows).to_csv(C.path("reports", "evaldata_filter_counts.csv"), index=False)
C.log_version("03_eval_data", "MT-Bench labels, baselines, md-human correlation, UF 500 + brackets",
              [C.path("data", "processed", f) for f in ("uf_brackets.parquet", "uf_answers.parquet", "mtbench_human_labels.parquet")])
br.head(3).T""",
]

build(cells, "notebooks/03_eval_data_and_controls.ipynb", "03_eval_data_and_controls")
