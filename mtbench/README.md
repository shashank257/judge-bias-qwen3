# MT-Bench: Qwen3 position and formatting sensitivity (DSC 261)

This folder contains the first-turn MT-Bench human-agreement and format/position-sensitivity analysis, including the evaluation harness, 31-pair dataset, 8 configurations of judge evaluations, recovery logs, and report figures.

## Dataset and design

- **31 first-turn answer pairs**, with released expert preference labels; 310 planned jobs per judge configuration (31 × 10).
- **Formatting:** A plain/B plain, A markdown/B plain, A plain/B markdown, A markdown/B markdown.
- **Order:** AB and BA. The plain/plain condition is repeated once in each order to estimate repeat noise.
- **Judges:** Qwen3 0.6B, 1.7B, 4B and 8B, with thinking OFF/ON configurations.
- **Judge output:** `[[A]]` or `[[B]]`, no ties; original answer identity is reconstructed from display order.

## Repository layout

```text
mtbench/
├── README.md
├── EVALUATION_CONTRACT.md
├── requirements.txt
├── src/
│   ├── run_judges.py
│   ├── prepare_mtbench_pilot.py
│   ├── analyze_results.py
│   ├── consolidate_qwen3_4b.py
│   └── analyze_final.py
├── notebooks/                 # optional additional analysis notebooks
├── data/
│   └── mtbench_matched_31.csv
└── results/
    ├── logs/                 # original/recovery JSONL judge calls
    ├── analysis/             # validated judgments, summary, audit, CIs
    └── figures/              # report figures
```

## Reproducing the final analysis

From this `mtbench/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/analyze_final.py
```

`analyze_final.py` matches successful JSONL responses to canonical job IDs, excludes invalid records and jobs outside the 31-pair set, calculates per-pair metrics, and estimates 95% pair-bootstrap intervals (4,000 draws, seed 261). Results are saved to `results/analysis/`. It does **not** send model requests and does **not** reproduce the original GPU inference.

## Valid judgments (final analysis)

| Judge configuration | Valid / planned |
|---|---:|
| 0.6B thinking OFF | 310 / 310 |
| 0.6B thinking ON | 310 / 310 |
| 1.7B thinking OFF | 310 / 310 |
| 1.7B thinking ON | 306 / 310 |
| 4B thinking OFF | 310 / 310 |
| 4B thinking ON | 310 / 310 |
| 8B thinking OFF | 310 / 310 matched (500 total logged) |
| 8B thinking ON | 310 / 310 across three logs |

**Total: 2,476 / 2,480 valid matched judgments.** The 1.7B thinking-ON file includes four invalid attempts, which are not silently treated as successes.

## Selecting the source logs

- 1.7B thinking ON: only the `...seeded_v2_fp16.jsonl` run is used. Earlier pilot/partial runs are excluded.
- 4B thinking OFF: `qwen3_4b_instruct_matched31_off.jsonl` (not the earlier `qwen3_4b_matched31_off.jsonl` failed pilot).
- 4B thinking ON: `qwen3_4b_matched31_on_consolidated.jsonl` contains the 310 accepted records. The original retry attempts are not included in this package.
- 8B thinking OFF: the 500-record `qwen3_8b_50_native_off.jsonl` is restricted to matched canonical jobs, yielding 310.
- 8B thinking ON: the accepted records are drawn from the original Ollama log, the local recovery log, and the 25-job Colab NF4 recovery log. Duplicate successful job IDs use the first valid occurrence. `results/analysis/audit.json` records selection counts and overlap conflicts.

**Interpretation:** The inference settings were not identical across all configurations (including the 4B instruct-vs-thinking checkpoint distinction, sampling vs greedy decoding, and 8B Ollama/Colab quantization). Reported comparisons are observational comparisons of tested configurations, not isolated causal estimates of enabling thinking.

## Key outputs

- `results/analysis/metric_summary.csv`: per-configuration metrics and pair-bootstrap confidence intervals.
- `results/analysis/thinking_comparisons.csv`: matched per-pair ON-minus-OFF differences.
- `results/analysis/judgments_valid.csv`: canonical matched outcomes per model setup.
- `results/analysis/audit.json`: record counts, missing jobs, and overlap conflicts.
- `results/figures/`: report visualizations.

## Notes

The human-judgment sample was prepared using `src/prepare_mtbench_pilot.py` and a filtered selection of 31 first-turn pairs. The included CSV is the fixed input for this analysis. The first-turn matched experiment is the unit of reproduction; the broader 100-pair pilot code is provided for provenance.

This repo includes the human-answer text and model outputs. Confirm your team's repository visibility and any applicable upstream dataset use requirements before publishing publicly. Never commit tokens, API credentials, or local `.env` files.
