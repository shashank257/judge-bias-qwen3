# Data dictionary

## Aggregate rows: tables/merge_summary.csv
`dataset`, `analysis_variant`: provenance labels; `config_id`: model/mode key; `model`: HF checkpoint; `thinking`: boolean; `n`: available candidates (1/2/4); `subset`: all or without_gpt4_candidates; `planned_prompts`: eligible source count; `complete_prompts`: matched complete n=4 cohort used for all n in that mode/subset.
`selected_quality`: mean GPT-4 proxy rating (1-5), higher is better. `regret`: best available proxy rating minus selected rating, lower is better. `format_drift`: selected Markdown indicator minus proxy-best Markdown indicator, mean in [-1,1]; multiply by 100 for percentage points. `best_quality_selected`: fraction selecting a candidate with proxy-best rating, ties allowed; this is not human agreement accuracy. Each `_ci_low/_ci_high` pair is a 95% percentile bootstrap interval (2,000 resamples, seed 261). `random_expected_quality`: uniform-random candidate baseline on the same cohort. `oracle_quality`: best available proxy rating. Missing means are blank.

## Prompt outcomes: tables/merge_prompt_outcomes.csv
`prompt_id`: frozen hashed identity; `source`: source dataset; `without_gpt4`: excludes prompts containing GPT-4 candidates; `selected_id`, `gold_id`: answer IDs within prompt; `missing`: unavailable bracket outcome. Missing outcomes must not be converted to zero. Candidate n=1 is fixed by the seeded subset, so its regret is necessarily zero. All n reuse a nested bracket; they are not independent experiments.

## Raw JSONL
selected_prompts.jsonl contains prompt/question, four answer texts and answer IDs, generating models, fine-grained ratings, derived quality, Markdown flag, and candidate subsets. judge_results.jsonl contains request identities, prompt messages, candidates, displayed order (AB/BA), attempt (zero-based), status, verdict mapped back to original candidate A/B, final content, reasoning, token counts, batch seed/timing, environment, and UTC time. Never remap verdict through order a second time.
`source_request_id`, `original_status`, `original_verdict`: recovery provenance. Recovered request/config hashes differ because the parser policy is explicit. `recovery_audit.jsonl` records migration changes. `joined_results.jsonl` contains latest requested bracket rows, not every attempt. Original strict logs include all original saved attempts.

## Paired and pooled tables
paired_thinking.csv: same-prompt on-minus-off differences; negative regret difference is an improvement. pooled_h3_cluster_bootstrap.csv: equally weighted available configuration-prompt rows with prompt-cluster bootstrap (10,000 resamples, seed 261); common_all_six uses only prompts complete in all six modes. Format drift is a fraction in CSV, percentage points in prose.

## Important distinction
GPT-4 ratings are a quality proxy, not human labels. Markdown drift is descriptive: this experiment does not strip formatting or hold content fixed. Preserve that distinction when integrating Arena and MT-Bench sections.
