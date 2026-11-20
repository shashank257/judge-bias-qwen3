> POST-HOC PARSING SENSITIVITY ANALYSIS. Original strict-parser retry selection retained. See RESULTS_SECTION.md for pooled H3 and limits.

# UltraFeedback midterm trial

Preliminary results from actual recorded judgments. GPT-4 ratings are a quality proxy, not human ground truth.

Sample: 25 prompts with four existing answers each; models: Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B, Qwen/Qwen3-4B; thinking off/on.
Data is independently sampled from this workspace, not verified against the teammate frozen dataset.
No new answers are generated: the model only judges the provided answers.

| Model | Mode | Candidates | Complete prompts | Selected quality (1-5) | Regret (lower is better) | Format drift (points) |
|---|---|---:|---:|---:|---:|---:|
| Qwen/Qwen3-0.6B | off | 1 | 25 | 4.030 | 0.000 | 0.000 |
| Qwen/Qwen3-0.6B | off | 2 | 25 | 4.420 | 0.160 | 12.000 |
| Qwen/Qwen3-0.6B | off | 4 | 25 | 4.390 | 0.520 | 12.000 |
| Qwen/Qwen3-0.6B | on | 1 | 16 | 3.750 | 0.000 | 0.000 |
| Qwen/Qwen3-0.6B | on | 2 | 16 | 4.438 | 0.156 | 12.500 |
| Qwen/Qwen3-0.6B | on | 4 | 16 | 4.578 | 0.328 | 31.250 |
| Qwen/Qwen3-1.7B | off | 1 | 25 | 4.030 | 0.000 | 0.000 |
| Qwen/Qwen3-1.7B | off | 2 | 25 | 4.450 | 0.130 | 12.000 |
| Qwen/Qwen3-1.7B | off | 4 | 25 | 4.620 | 0.290 | 12.000 |
| Qwen/Qwen3-1.7B | on | 1 | 25 | 4.030 | 0.000 | 0.000 |
| Qwen/Qwen3-1.7B | on | 2 | 25 | 4.510 | 0.070 | 4.000 |
| Qwen/Qwen3-1.7B | on | 4 | 25 | 4.660 | 0.250 | 12.000 |
| Qwen/Qwen3-4B | off | 1 | 25 | 4.030 | 0.000 | 0.000 |
| Qwen/Qwen3-4B | off | 2 | 25 | 4.490 | 0.090 | 4.000 |
| Qwen/Qwen3-4B | off | 4 | 25 | 4.640 | 0.270 | 12.000 |
| Qwen/Qwen3-4B | on | 1 | 25 | 4.030 | 0.000 | 0.000 |
| Qwen/Qwen3-4B | on | 2 | 25 | 4.560 | 0.020 | 0.000 |
| Qwen/Qwen3-4B | on | 4 | 25 | 4.720 | 0.190 | 12.000 |

Interpretation: regret is the best available GPT-4 proxy score minus the selected score.
Format drift is the Markdown indicator of the chosen answer minus that of the proxy-best answer. It is descriptive, not proof that formatting caused the selection.
n=1 is the seeded first answer; n=2 uses semifinal 1; n=4 uses the tournament final. No extra model calls for n=1/2.
Both orders are judged; disagreement is resolved by a fixed seeded answer-identity priority. Missing/invalid matches do not produce invented winners.
The same complete n=4 cohort is used for all n within each mode. Compare modes using paired_thinking_comparison.json.
95% bootstrap intervals (2,000 resamples, seed 261) are in summary.csv. At 25 prompts they are exploratory.
This trial cannot establish model-size effects, replicate Arena bias estimates, or prove a causal link between formatting and quality.
Transformers/Colab and recorded numerical precision differ from the teammate MLX run. Exact teammate inputs and prompt revision were unavailable.
Thinking uses temperature 0.6/top-p 0.95; off uses 0.7/0.8. A separate sampling control has not been run.
AI assistance: Codex prepared the trial code, data packaging and report template; results require actual GPU execution.

Sources: https://huggingface.co/datasets/openbmb/UltraFeedback ; https://huggingface.co/Qwen/Qwen3-0.6B ; https://huggingface.co/Qwen/Qwen3-1.7B ; https://huggingface.co/Qwen/Qwen3-4B
