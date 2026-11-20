# Week 1 report: data, validation, hosted API selection (G3)

## Results
- **Pools are frozen.** All sha256 hashes are in `data/processed/freeze_manifest.json`.
  - Position pool: 1,000 pairs (2023 Arena).
  - Format pool: 1,000 pairs (167 from 2023, 833 from 2025).
  - 200 reserve pairs per pool; 100 pilot pairs; 200 check pairs.
- **Filter counts.**
  - 2023: 33,000 → 8,325 eligible. Only 170 had markdown in both answers, so the 2025 supplement was triggered.
  - 2025: 135,634 → 16,052 eligible, 11,308 of them strip cleanly with markdown in both answers.
  - Every step is in `reports/arena_filter_counts.csv`.
- **Stripping.** 2,400 of 2,400 stripped answers keep the exact word sequence. 160 candidates failed before the
  freeze and were excluded. In the 50-pair audit, meaning and wording were preserved in 100 of 100 answers; 2 pairs
  with deep nesting were replaced in the format pool from the reserve (they remain in the 100-pair midterm subset, drawn earlier).
- **Humans prefer markdown.** When only one answer used markdown, humans picked it 57% of the time (95% CI
  53.5–60.1% for 2023; 54.5–58.1% for 2025). The effect holds with length controlled (OR 1.21 and 1.15,
  p ≤ 0.009). Human agreement can therefore reward format bias.
- **Baselines.** Random agreement ≈ 0.50. Length-only agreement: 0.58 (Arena position pool), 0.57 (format pool),
  0.69 (MT-Bench, κ 0.38).
- **UltraFeedback.** 500 prompts × 4 answers (243 ShareGPT, 124 UltraChat, 133 Evol-Instruct). Our recomputed gold
  matches the provided fine-grained score for 100% of answers. 111 prompts contain a GPT-4 answer, leaving 389 in
  the sensitivity subset.
- **Manifest.** 151,296 judge calls, of which 16,912 per configuration. About 200M input tokens.
- **APIs.** None of the 4 checkpoints is on OpenRouter, Together, DeepInfra, Novita or HF Inference Providers.
  Featherless has all four, but 8B is FP8. **Choice: our own BF16 vLLM endpoint (Modal), with Featherless as the
  fallback.** Estimated cost: about $16 self-hosted (assumed throughput) or about $88 on Featherless.
- **Local pilot** (same checkpoints, 520 calls). 0 invalid verdicts; 3% of thinking runs truncated at 1.7B. Mean
  output was 106–633 tokens. Preview: 0.6B with thinking off picks the second slot 93% of the time.

## Still open (needs credentials or people)
1. Re-run notebook 01 with an `HF_TOKEN` that has accepted the official 2023 Arena terms (we used a mirror).
2. Each member confirms their 10 audit pairs in `reports/manual_audit.csv`.
3. Hosted compatibility check and pilot: `python scripts/run_pilot.py featherless`, or point it at the Modal vLLM
   URL. Then confirm the cost model and set the provider in contract v1.0.

## Deviations from the proposal (all logged in the contract change log)
- Hosted vLLM replaces Kaggle.
- New context-fit filter (≤ 6,000 tokens).
- One pair per prompt in each pool.
- Separate position and format pools (16,912 calls per configuration, not ~13K).
- The local pilot ran 0.6B and 1.7B instead of 0.6B and 8B.
- Only headings, bold and bullets are stripped; italics and numbered lists stay.

## Midterm results (100 Arena pairs × 8 configurations, 8,000 judge calls, 0 invalid)
4B ran as 8-bit (MLX) and 8B as Qwen's official 4-bit AWQ (vLLM, RTX 3070); the contract specifies BF16, so these rows are reported separately.
Source: `notebooks/06_midterm_results.ipynb`, `reports/midterm_results.csv`, `reports/midterm_excess_flip.png`.
Excess flip = flip rate − noise floor, in points, with a 95% bootstrap CI.

| Config | Position excess flip | Picks 1st slot | Format excess flip | Picks markdown answer | Human agreement |
|---|---|---|---|---|---|
| 0.6B no-think | **73.5** [63.5, 83.0] | 6.5% (prefers 2nd) | −0.5 [−4.8, 3.8] | 50.5% | 58% |
| 0.6B think | 6.5 [−5.5, 18.5] | 42.5% | 1.2 [−6.8, 9.0] | 52.2% | 61% |
| 1.7B no-think | **36.5** [25.0, 48.0] | 74.5% (prefers 1st) | −1.5 [−6.0, 3.0] | 54.0% [51.2, 56.8] | 59% |
| 1.7B think | **16.5** [5.5, 28.0] | 65.0% | −2.2 [−7.8, 3.0] | 51.2% | 51% |
| 4B 8-bit no-think | **24.5** [14.5, 35.0] | 66.0% | 2.2 [−2.2, 6.8] | 56.2% | 60% |
| 4B 8-bit think | **16.0** [7.0, 25.5] | 60.0% | 0.0 [−4.8, 4.5] | 58.0% | 64% |
| 8B 4-bit AWQ no-think | **41.5** [31.0, 52.5] | 75.5% | −1.5 [−5.8, 2.5] | 53.5% | 62% |
| 8B 4-bit AWQ think | **15.0** [5.0, 25.5] | 59.5% | −1.0 [−6.2, 4.2] | 52.0% | 66% |

Reading:
- Position bias is large and real in 3 of the 4 configurations. Thinking cuts it sharply: 73.5 → 6.5 points
  at 0.6B and 36.5 → 16.5 at 1.7B.
- The two sizes lean in opposite directions: 0.6B no-think picks the *second* answer, 1.7B the *first*.
- On these pairs, deleting headings, bold and bullets does not change the verdict beyond noise. Only 1.7B no-think
  shows a small markdown preference (54%, CI above 50%).
- Human agreement is 51–61%, about the level of the length-only baseline (57%).

Caveats: 100 pairs, 2 of the 4 sizes, run locally (MLX; part of the run used batched generation with batch-level
seeds). These are interim results, not the pre-registered H1–H3 tests.
