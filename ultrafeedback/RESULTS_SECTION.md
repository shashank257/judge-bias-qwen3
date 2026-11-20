# UltraFeedback midterm results - G3

**Analysis:** post-hoc parsing sensitivity analysis; 25-prompt pilot, not the full proposed study.

We evaluated Qwen3-0.6B, 1.7B, and 4B with thinking off/on on 25 frozen UltraFeedback prompts (12 ShareGPT, 7 Evol-Instruct, 6 UltraChat). Each prompt has four existing candidate answers. Quality is the mean of four GPT-4 fine-grained ratings, not human preference. A knockout bracket judges both answer orders; disagreements use a fixed seeded answer-identity priority. The n=1/2/4 settings reuse the same bracket.

| Model | Thinking | Complete prompts | Selected quality | Regret | Format drift (pp) |
|---|---|---:|---:|---:|---:|
| Qwen/Qwen3-0.6B | off | 25/25 | 4.390 | 0.520 | 12.00 |
| Qwen/Qwen3-0.6B | on | 16/25 | 4.578 | 0.328 | 31.25 |
| Qwen/Qwen3-1.7B | off | 25/25 | 4.620 | 0.290 | 12.00 |
| Qwen/Qwen3-1.7B | on | 25/25 | 4.660 | 0.250 | 12.00 |
| Qwen/Qwen3-4B | off | 25/25 | 4.640 | 0.270 | 12.00 |
| Qwen/Qwen3-4B | on | 25/25 | 4.720 | 0.190 | 12.00 |

Thinking increased mean quality on matched prompts by 0.250 for 0.6B (16 prompts; 95% CI 0.078 to 0.453), 0.040 for 1.7B (25; -0.080 to 0.200), and 0.080 for 4B (25; -0.010 to 0.230). The latter two intervals include zero. Different sampling settings by mode and post-hoc parsing prevent attribution solely to reasoning. Missing 0.6B-on prompts may bias its estimate.

All six modes exhausted their planned attempts. Five modes completed 25 prompts; 0.6B thinking-on completed 16. Eleven conflicting semifinal outputs blocked nine full brackets. These remain missing; no winner was invented.

## H3 exploratory pooled estimate
- all_available: 14.18 percentage points (95% prompt-cluster bootstrap CI -6.52 to 34.03); 25 prompts, 141 configuration-prompt rows.
- common_all_six: 27.08 percentage points (95% prompt-cluster bootstrap CI 0.00 to 53.12); 16 prompts, 96 configuration-prompt rows.
- without_gpt4_available: 7.14 percentage points (95% prompt-cluster bootstrap CI -22.10 to 36.14); 15 prompts, 84 configuration-prompt rows.

The all-available estimate equally weights observed configuration-prompt rows; prompt clusters retain within-prompt dependence. The common-all-six analysis gives each of the 16 complete prompts six configurations. These exploratory, post-hoc pilot results do not establish the proposal's confirmatory H3 claim. Positive drift is an association with Markdown, not a causal formatting effect.

## Protocol deviation and provenance
The strict parser required one bracketed label at the end. The recovery accepts consistent explicit A/B markers anywhere in the final answer, including duplicate identical labels; conflicting labels, ties, missing verdicts and truncated answers stay invalid. It applies to every configuration. Only the latest saved attempt is reused, with no selection among earlier attempts. There were 194 recovered judgments. Original strict outputs, recovery audit, and derived identities are included. The original parser influenced retries; therefore this is not equivalent to a fresh run under the relaxed rule.

The original strict snapshot was captured before the new finals and most 4B work. Its coverage is not a controlled full-run strict-versus-tolerant comparison. A strict reparse of later tolerant outputs would be another sensitivity analysis, not a completed strict-protocol rerun.

## Limits and remaining full-study work
This pilot uses 25 prompts, six modes, and Colab Transformers FP16 instead of the proposed 500 prompts, eight modes, and vLLM/Kaggle setup. 8B, sampling controls, full-scale H3, and cross-dataset correlations are absent. The no-GPT-4-candidate sensitivity retains 15 planned prompts (completion varies by mode). Teammate frozen inputs/configuration were not supplied. Do not claim exact replication, human accuracy, or causal format bias from this run.

AI assistance: Codex prepared code, data packaging, recovery diagnostics, analysis, figures, and draft report. Model judgments were generated in the user's Colab runs. Team review is required before submission.
