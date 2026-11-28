# UltraFeedback final-study analysis | COMPLETE

500 frozen prompts, 4 answers each, 3 Qwen3 sizes, thinking off/on. 2902/3000 usable four-answer tournaments; 412 prompts available in every mode. 0 parts have pending requests.

## Best-of-4 Markdown drift

| Cohort | Observed prompts | Usable tournaments | Drift (pp) | 95% interval (pp) |
|---|---:|---:|---:|---:|
| all_500 | 500 | 2902 | -0.76 | -3.24, +1.76 |
| new_475 | 475 | 2761 | -1.52 | -3.84, +0.87 |
| pilot_25 | 25 | 141 | +14.18 | -6.52, +34.03 |
| no_gpt4_candidates | 362 | 2100 | -0.95 | -4.02, +2.26 |
| new_no_gpt4 | 347 | 2016 | -1.29 | -4.33, +1.74 |

The primary interval includes zero or is unavailable; evidence for positive drift is inconclusive.

## Results by mode (four candidates)

| Mode | Usable / planned | Quality / 5 | Regret | Markdown drift (pp) | Best-rated selected (%) |
|---|---:|---:|---:|---:|---:|
| qwen3-0.6b-off | 493/500 | 4.033 | 0.737 | -8.519 | 35.294 |
| qwen3-0.6b-on | 429/500 | 4.309 | 0.455 | -0.699 | 50.583 |
| qwen3-1.7b-off | 496/500 | 4.375 | 0.393 | -0.605 | 55.847 |
| qwen3-1.7b-on | 487/500 | 4.489 | 0.278 | 0.821 | 63.039 |
| qwen3-4b-off | 499/500 | 4.525 | 0.247 | 2.204 | 64.729 |
| qwen3-4b-on | 498/500 | 4.554 | 0.216 | 2.209 | 68.675 |

## Paired thinking comparison (on minus off)

| Model | Matched prompts | Quality difference | 95% interval |
|---|---:|---:|---:|
| qwen3-0.6b | 426 | 0.291 | 0.205, 0.381 |
| qwen3-1.7b | 484 | 0.113 | 0.059, 0.168 |
| qwen3-4b | 497 | 0.034 | -0.003, 0.072 |

## Interpretation and handoff

- Primary estimate: selected Markdown indicator minus the proxy-best indicator at n=4, equally weighted over usable configuration-prompt tournaments. Bootstrap whole observed prompts with all their observed modes together (10,000 draws, percentile 95% intervals, seed 261). A one-prompt cohort has no interval.
- The full 500 include the 25 pilot prompts once. new_475 excludes all 25 prompts used to refine the parser. common_six requires a complete outcome in every mode. no_gpt4_candidates excludes prompts with a GPT-4 candidate. Per-mode n=1/2/4 comparisons use the same complete n=4 prompts within each mode. Paired contrasts use the intersection for the two modes in question.
- Quality is the mean of four GPT-4 fine-grained ratings (1–5). Regret is the best available rating minus the selected rating. A tied best rating counts as best selected. The seeded proxy-best answer determines Markdown drift in a tie. Zero n=1 regret follows from the single candidate.
- An AB/BA disagreement is resolved using the frozen seeded answer-identity rule. Order diagnostics use both valid presentations of a match. Tournament disagreement differs from the Arena controlled bias test. No human votes or counterfactual format edits were collected here.
- The 8B configuration was omitted from this study. Mode comparisons also change sampling parameters. The parsing rule was refined using the pilot. Missing outcomes may affect averages. Pairwise and source contrasts are exploratory, with unadjusted intervals. An Arena bias/drift correlation requires a separately matched Arena table.
- Recorded attempt and time totals include latest saved pilot attempts plus full-run attempts. Earlier pilot retries, loading, idle time and session overhead are excluded. Strict-format diagnostics reparse observed answers only and do not simulate an independent strict-protocol run.
- For merging, use prompt_outcomes.csv keyed by config_id, prompt_id and n within UltraFeedback. Missing outcome metrics are blank. Use pooled_h3.csv for primary and sensitivity estimates, per_mode.csv for curves, paired_comparisons.csv for matched differences, and pending_parts.csv for unfinished work. Raw responses remain in the source checkpoint folders and pilot bundle.
