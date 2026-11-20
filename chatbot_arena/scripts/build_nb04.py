from nb_common import build

cells = [
r"""# 04 — Experiment specification and request manifest (Week 1)
Final judge prompt, answer delimiters, verdict format and invalid-response rules; a request manifest with one row
per judge call (pair, presentation condition, order, configuration, repeat); request counts with reuse; and the
metric / analysis definitions that go into the evaluation contract (`docs/evaluation_contract.md`).""",

r"""MD:## 1. Judge prompt, delimiters, verdict format
MT-Bench `pair-v2` (FastChat `judge_prompts.jsonl`) with the tie clause removed (our pools have no ties).
Delimiters are MT-Bench's `[The Start of Assistant A's Answer]` … `[The End of Assistant A's Answer]`.
Verdict = the **last** `[[A]]` or `[[B]]` after the final `</think>`; nothing parseable → one retry with a fresh seed
→ still nothing → **invalid**. `[[C]]` (tie) is not allowed and is treated as unparseable.""",

r"""from dsc261 import judge as J, manifest as MF, evaldata as E
print("SYSTEM PROMPT (main):\n" + J.SYSTEM_PROMPTS["mtbench_pair_v2_notie"] + "\n")
print("SYSTEM PROMPT (paraphrase check):\n" + J.SYSTEM_PROMPTS["paraphrase_v1"] + "\n")
print("USER TEMPLATE:\n" + J.USER_TEMPLATE)
for t in ["... so [[A]]", "<think>maybe [[B]]</think> Final: [[A]]", "A is better", "[[C]]", "[[A]] ... actually [[B]]"]:
    print(repr(t), "->", J.parse_verdict(t))""",

r"""MD:## 2. Rendered chat template (thinking on vs off, same weights)
We render Qwen3's own chat template from the pinned revision. Thinking **off** pre-fills an empty
`<think>\n\n</think>\n\n` block, so the model never reasons (reasoning is disabled, not hidden). The template is
byte-identical across the four sizes.""",

r"""msgs = J.build_messages("What is 2+2?", "4", "Four.")
for th in (False, True):
    print(f"--- enable_thinking={th} ---\n" + J.render_prompt(msgs, "8B", th)[-140:])
print("templates identical across sizes:", all(J.render_prompt(msgs, s, th) == J.render_prompt(msgs, "8B", th) for s in J.MODEL_REVISIONS for th in (True, False)))
print("sampling:", J.SAMPLING, "| think budget:", J.THINK_BUDGET, "| verdict budget:", J.VERDICT_BUDGET, "| no-think max tokens:", J.NOTHINK_MAX_TOKENS)""",

r"""MD:## 3. Request manifest
- **position**: position pool, original answers, both orders × repeats {0, 1} → 4 per pair (repeat 1 = noise floor).
- **format**: format pool, (A, B) ∈ {orig(markdown), plain}², both orders, repeat 0 → 8; plus orig/orig repeated in
  both orders (noise floor) → **10 per pair**.
- **h3**: 500 UltraFeedback brackets, 2 semifinals × 2 orders (fixed) + final × 2 orders (built adaptively from
  semifinal winners; 2 placeholder rows per prompt) → 6 per prompt.
- **sampling_check**: 200 check pairs × 4 sizes, thinking off with *thinking-mode* sampling, 8 format cells.
- **prompt_check**: same 200 pairs, paraphrased prompt, 1.7B and 8B, both modes, 8 format cells.
Identical requests needed by two experiments appear once (`reused_by`). Repeats differ in `repeat` → different
`request_id` → different seed → fresh generation; the harness only skips request ids that already finished.""",

r"""pos = pd.read_parquet(C.path("data", "processed", "position_pool.parquet"))
fmt = pd.read_parquet(C.path("data", "processed", "format_pool.parquet"))
br = pd.read_parquet(C.path("data", "processed", "uf_brackets.parquet"))
check = fmt.sample(200, random_state=C.SEED).sort_values("pair_id").reset_index(drop=True)
check[["pair_id"]].to_parquet(C.path("data", "processed", "check_pairs.parquet"), index=False)
man = MF.build_manifest(pos, fmt, br, check)
assert man.request_id.is_unique
man.to_parquet(C.path("data", "manifest", "request_manifest.parquet"), index=False)
print("total requests:", len(man))
man.groupby(["experiment"]).size().rename("requests").to_frame()""",

r"""per_cfg = man[man.experiment.isin(["position", "format", "h3"])].groupby(["config", "experiment"]).size().unstack(fill_value=0)
per_cfg["total"] = per_cfg.sum(axis=1)
per_cfg""",

r"""overlap = set(pos.pair_id) & set(fmt.pair_id)
print("pairs in both pools:", len(overlap), "| requests shared between experiments:", int(man.reused_by.str.contains(",").sum()))
# repeats must be distinct request ids with distinct seeds
r0 = man[(man.experiment == "position") & (man.repeat == 0)].request_id.head(3).tolist()
r1 = [x.replace("|r0", "|r1") for x in r0]
print([(J.request_seed(a, 0), J.request_seed(b, 0)) for a, b in zip(r0, r1)])
assert all(J.request_seed(a, 0) != J.request_seed(b, 0) for a, b in zip(r0, r1))
print("example ids:", *man.request_id.sample(3, random_state=1).tolist(), sep="\n")""",

r"""MD:## 4. Token volume (inputs) for cost estimates
Input tokens per request from the actual texts (Qwen3 tokenizer). Output tokens come from the pilot (notebook 05).""",

r"""tok = J.get_tokenizer("8B")
def ntok(texts):
    return np.array([len(x) for x in tok(list(texts), add_special_tokens=False)["input_ids"]])
fp = fmt.set_index("pair_id"); pp = pos.set_index("pair_id")
def text_of(row):
    src = fp if row.pair_id in fp.index else pp
    r = src.loc[row.pair_id]
    t = lambda ans, var: r[f"answer_{ans}"] if var == "orig" else r[f"plain_{ans}"]
    return J.render_prompt(J.build_messages(r.question, t(row.slot1_answer, row.slot1_variant), t(row.slot2_answer, row.slot2_variant), row.prompt_version), "8B", False)
arena_req = man[man.experiment != "h3"]
uniq = arena_req.drop_duplicates(["pair_id", "slot1_answer", "slot1_variant", "slot2_answer", "slot2_variant", "prompt_version"])
uniq = uniq.assign(in_tokens=ntok([text_of(r) for r in uniq.itertuples()]))
key = ["pair_id", "slot1_answer", "slot1_variant", "slot2_answer", "slot2_variant", "prompt_version"]
man = man.merge(uniq[key + ["in_tokens"]], on=key, how="left")
ua = pd.read_parquet(C.path("data", "processed", "uf_answers.parquet")); up = pd.read_parquet(C.path("data", "processed", "uf_prompts.parquet")).set_index("uf_id")
ua["tok"] = ntok(ua.response); base_tok = 330  # system prompt + template + delimiters
h3_tok = {u: base_tok + len(tok(up.at[u, "instruction"])["input_ids"]) + 2 * g.tok.mean() for u, g in ua.groupby("uf_id")}
m = man.experiment == "h3"
man.loc[m, "in_tokens"] = man.loc[m, "pair_id"].map(h3_tok)
man[["request_id", "in_tokens"]].to_parquet(C.path("data", "manifest", "request_input_tokens.parquet"), index=False)
vol = man.groupby("experiment").in_tokens.agg(["count", "mean", "sum"]); vol["sum_M"] = vol["sum"] / 1e6
print("total input tokens (M):", round(man.in_tokens.sum() / 1e6, 1))
vol.round(1)""",

r"""import json
summary = {"total_requests": int(len(man)), "by_experiment": man.groupby("experiment").size().to_dict(),
           "per_config_main": per_cfg["total"].to_dict(), "input_tokens_total": float(man.in_tokens.sum()),
           "input_tokens_by_config": man.groupby("config").in_tokens.sum().round(0).to_dict()}
json.dump(summary, open(C.path("data", "manifest", "request_counts.json"), "w"), indent=2)
C.log_version("04_manifest", "request manifest + counts", [C.path("data", "manifest", "request_manifest.parquet")])
summary["by_experiment"]""",

r"""MD:## 5. Metric and analysis definitions (frozen in `docs/evaluation_contract.md`)
- **Flip** (position): verdicts pick different *original answers* in the two orders. **Flip rate** = share of pairs
  that flip between orig and swap (repeat 0). **Noise floor** = share that flip between repeat 0 and repeat 1 of the
  *same* order (averaged over the two orders). **Excess flip rate** = flip rate − noise floor, in points.
- **Flip** (format): for answer X, compare the verdict in (X md, Y md) with (X plain, Y md) and (X md, Y plain);
  flip = the winning original answer changes when only markdown changes. Noise floor from the md/md repeats.
- **Signed preference**: position = share of verdicts for slot 1; format = share of verdicts for the markdown
  variant in the asymmetric cells (orig/plain and plain/orig), 0.5 = no bias.
- **Invalid verdicts** count as a flip in both the edited and the noise-floor comparison; sensitivity analysis
  drops pairs with any invalid verdict. Invalid and truncation rates are reported per configuration.
- **Human agreement**: accuracy and Cohen's κ vs the human vote on original pairs (repeat 0, orig order), and
  **swap-and-average** agreement: a pair counts as agreeing when the order-averaged vote (A wins in both orders →
  A; split → 0.5 credit) matches the human.
- **Format drift (H3)**: per prompt, markdown indicator of the judge's n = 4 pick minus the gold pick's markdown rate
  (ties averaged); pooled over configurations with prompt-cluster bootstrap, also per configuration.
- **Selection regret**: best gold among the n candidates minus the gold of the judge's pick (n = 1, 2, 4).
- **Statistics**: 95% cluster-bootstrap CIs (clusters = prompt_id, 10,000 resamples, seed 261); H2 by GEE logistic
  regression on flip indicators from edited and noise-floor comparisons with condition × log2(size) × thinking,
  effects as average marginal effects in points; Holm correction across biases and effects; the cross-configuration
  correlation of drift/regret with format bias is exploratory (|ρ| ≥ 0.74 needed at n = 8).""",
]

build(cells, "notebooks/04_experiment_spec.ipynb", "04_experiment_spec")
