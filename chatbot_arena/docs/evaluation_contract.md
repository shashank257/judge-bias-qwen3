# Evaluation contract — v1.0 (frozen 2026-11-03, after the week-1 *local* pilot)

> Status: data, prompts, metrics and analysis rules are frozen. Two items stay open until week 2: (a) serving
> provider settings, which need the hosted pilot, and (b) team sign-off on the manual audit, which is so far an
> AI audit only. Neither may change the pools, prompts or metrics.

Everything below is fixed before any main-study judge call. Changes after this point need a new version
number, a dated entry in the change log at the bottom, and a reason. Results are reported against this
version, including null and negative ones.

## 1. Data (frozen files, sha256 in `data/processed/freeze_manifest.json`)
| Pool | File | Size | Source | Use |
|---|---|---|---|---|
| Position | `data/processed/position_pool.parquet` | 1,000 pairs | 2023 Chatbot Arena | position bias, noise floor, human agreement |
| Format | `data/processed/format_pool.parquet` | 1,000 pairs (167 from 2023, 833 from 2025) | Arena 2023 + 2025 supplement | format bias (2×2 × both orders) |
| Reserves | `*_reserve.parquet` | 200 + 200 | same | replace pairs failing the audit only |
| Checks | `data/processed/check_pairs.parquet` | 200 format-pool pairs | — | sampling check and paraphrased-prompt check |
| H3 | `uf_prompts / uf_answers / uf_brackets.parquet` | 500 prompts × 4 answers | UltraFeedback (sharegpt, ultrachat, evol_instruct) | best-of-n |
| Human agreement | `mtbench_human_labels.parquet`, `human_labels_*.parquet` | 1,284 MT-Bench turn-1 votes; 2,000 Arena votes | MT-Bench, Arena | accuracy, Cohen's κ |

Filters (counts in `reports/arena_filter_counts.csv` and `reports/evaldata_filter_counts.csv`):
English label plus at least 95% Latin letters in the prompt and in both answers; single turn; text only; non-tied
human vote; answer-length ratio ≤ 2 (in words, with markdown markers not counted); no tables or code blocks;
exact duplicates removed; judge prompt ≤ 6,000 tokens; one pair per prompt cluster in each pool; seed 261.
Pinned dataset revisions are listed in `data/dataset_records.json`.

**Open item:** the 2023 Arena file came from a public mirror (`agie-ai/lmsys-chatbot_arena_conversations`) because the
official repo is gated. Someone with an accepted HF token must re-run notebook 01. If the official file yields
different pools, the official version replaces the mirror as v1.1. Shape and column checks already match.

## 2. Presentation edits
- Supported markdown: ATX headings, `**bold**`, and bullet markers `-`, `*`, `+`. Stripper version `strip-v1.0`
  (`src/dsc261/markdown.py`). Everything else (italics, numbered lists, links, rules) is identical in both variants.
- Every stripped answer passes the word-sequence check (2,400 of 2,400). The 50-pair audit (100 answers) found
  meaning and wording preserved in all 100. 2 answers lost list structure (3-level nesting), and those pairs were
  replaced from the reserve (`freeze_manifest.json → audit_replacements`). The audit was done by an AI reviewer.
  Each member must confirm their 10 assigned pairs (`assigned_to` in `reports/manual_audit.csv`) before the
  main run.
- **Limitation of the plain variant.** It is free only of the three supported features. Numbered lists (in
  about 78% of format-pool answers), italics, `---` rules and blockquotes remain. The format effect is therefore
  the effect of headings, bold and bullets, not of all markdown.
- Format cells: (A, B) ∈ {orig(markdown), plain}², each shown in both orders. Answer identity is carried in
  `slot1_answer` / `slot2_answer`.

## 3. Judges and decoding
| Size | Checkpoint | Revision (HF sha) |
|---|---|---|
| 0.6B | Qwen/Qwen3-0.6B | c1899de289a04d12100db370d81485cdf75e47ca |
| 1.7B | Qwen/Qwen3-1.7B | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |
| 4B | Qwen/Qwen3-4B | 1cfa9a7208912126459214e8b04321603b3df60c |
| 8B | Qwen/Qwen3-8B | b968826d9c46dd6066d109eabc6255188de91218 |

- Thinking on/off uses the same weights through the checkpoint's own chat template (`enable_thinking`), rendered
  locally and sent to `/v1/completions`. With thinking off the template pre-fills an empty think block, so
  reasoning is disabled rather than hidden.
- Sampling follows the model card: thinking uses T=0.6, top-p 0.95, top-k 20, min-p 0; non-thinking uses T=0.7,
  top-p 0.8, top-k 20, min-p 0. Never greedy. The sampling check runs thinking off with the thinking settings
  (4 sizes × 200 pairs).
- Thinking budget is 2,048 tokens. If no `</think>` appears by then, we append Qwen's early-stop sentence plus
  `</think>` and continue for up to 512 tokens. Non-thinking output is capped at 1,024 tokens.
- Seeds: `seed = sha256(request_id | attempt | 261)`. Repeats have different request ids, so they always get a
  fresh generation.
- **Serving** (see `docs/api_comparison.md`): vLLM serving the four pinned revisions in **BF16**, with the same
  vLLM version for every cell, through an OpenAI-compatible endpoint. Provider, vLLM version and GPU type are
  recorded with every response. If the team uses Featherless instead, its 8B is FP8 while 0.6B, 1.7B and 4B are
  BF16. That precision difference must then be reported as a confound in H2.

## 4. Prompt and verdicts
- Main prompt: MT-Bench `pair-v2` without the tie clause (`judge.SYSTEM_PROMPTS["mtbench_pair_v2_notie"]`).
  Robustness check: `paraphrase_v1` on the 200 check pairs at 1.7B and 8B, both modes.
- Verdict: the last `[[A]]` or `[[B]]` after the final `</think>`. If none is found, retry once with a new seed.
  If it still fails, the verdict is **invalid**.
- An invalid verdict counts as a flip in both the edited and the noise-floor comparisons. As a sensitivity
  analysis we drop affected pairs. Invalid, retry and truncation rates are reported for every configuration.
- Provider or HTTP errors are retried by the client (up to 5 times), then by the harness (up to 3 times). They
  never consume the verdict retry, are logged with `status=error`, are **not** counted as invalid verdicts, and
  stay open for re-queue (`judge.done_request_ids`). On resume, attempt numbering continues from the log, so a
  retry never reuses an earlier seed.

## 5. Measures
Manifest note: a request needed by several experiments appears once, labelled with the first experiment, and
`reused_by` lists all of them. Select an experiment's requests with `reused_by.str.contains(...)`, not with
`experiment == ...`.
The sampling and paraphrase checks include md/md repeats in both orders, so excess flip rates can be computed
for them too.

- **Flip rate (position):** the share of pairs whose winning original answer differs between the two orders
  (repeat 0).
- **Noise floor:** the share of pairs whose winner differs between repeat 0 and repeat 1 of the same order,
  averaged over both orders.
- **Excess flip rate:** flip rate minus noise floor, in points.
- **Format flip:** for each answer X, compare the verdict in (X md, Y md) with (X plain, Y md). Flip means the
  winning original answer changes. The noise floor comes from the md/md repeats.
- **Signed preference:** for position, the share of verdicts for slot 1. For format, the share of verdicts for
  the markdown variant in the asymmetric cells.
- **Human agreement:** accuracy and Cohen's κ on original pairs (repeat 0, orig order).
- **Swap-and-average agreement:** order-averaged verdict versus the human vote; a split counts 0.5.
- **Controls:** random verdicts and a length-only predictor (`reports/baselines.csv`).
- **H3:** best-of-n through a knockout bracket (n = 1, 2, 4), each match judged in both orders.
  - A split decision goes to the answer with more valid wins, then to a deterministic coin on (prompt, pair).
  - Format drift at n=4 = md(judge pick) − md rate of the gold pick (ties averaged).
  - Selection regret = best gold among the n candidates − gold of the pick.
  - Gold = mean of the 4 fine-grained GPT-4 ratings. A prompt with any N/A rating is excluded.
  - The analysis is repeated without the 111 prompts that contain a GPT-4 answer (389 prompts remain).

## 6. Hypotheses and analysis (unchanged from the proposal, thresholds now frozen)
- **H1:** for position and for format, the largest judge we run with thinking on (8B, or 4B if 8B thinking is
  dropped) has an excess flip rate whose 95% cluster-bootstrap CI lower bound is > 2 points.
- **H2:** effects of size (per doubling) and of thinking on excess flip rate. Smallest effect of interest: 3
  points, tested in both directions (equivalence via two one-sided tests at ±3 points).
  - Model: GEE logistic regression on flip indicators from edited and noise-floor comparisons, with condition ×
    log2(size) × thinking, clustered by prompt.
  - Effects are reported as average marginal effects in points.
  - Holm correction across {position, format} × {size, thinking}.
- **H3:** pooled format drift at n = 4 has a 95% cluster-bootstrap CI that excludes 0. Correlations of drift and
  regret with format bias across the 8 configurations are exploratory (|ρ| ≥ 0.74 needed).
- **Bootstrap:** 10,000 resamples of prompt clusters, seed 261, percentile intervals.
- Each pool reports its results by source (2023 / 2025).

## 7. Drop order if throughput or budget is short (unchanged)
1. Paraphrased-prompt check.
2. 8B thinking.
3. H3.

Never go below 1,000 pairs per cell for position and format.

## Change log
- v1.0 (2026-11-03): first freeze. Changes from the proposal:
  - Kaggle/vLLM is replaced by a hosted vLLM endpoint.
  - New context-fit filter (≤ 6,000 tokens).
  - One pair per prompt cluster.
  - The position pool and format pool are separate (16,912 requests per configuration instead of the proposal's
    ~13K; 151,296 in total).
  - The pilot ran locally on 0.6B and 1.7B (MLX, same checkpoints). A hosted pilot is still required
    before the main run; see `docs/api_comparison.md`.
