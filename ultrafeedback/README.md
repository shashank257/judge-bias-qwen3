# UltraFeedback midterm pilot (H3)

We tested best-of-4 selection on 25 prompts with Qwen3 0.6B, 1.7B and 4B in both thinking modes. Five modes completed all 25 prompts. The 0.6B thinking-on mode completed 16. Markdown drift averaged +14.18 percentage points across 141 completed tournaments (95% prompt-cluster interval -6.52 to +34.03). This pilot leaves H3 unresolved.

[Results pack](DSC261_G3_UltraFeedback_midterm_results_pack.pdf) · [Editable results](RESULTS_SECTION.md) · [Data dictionary](DATA_DICTIONARY.md)

## Reproduce locally

From this folder, install the analysis dependencies and run the summary from the frozen prompts and saved decisions. The analysis runs on CPU.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py
```

See `reproduced/summary.csv`, `reproduced/pooled.csv`, and `reproduced/ultrafeedback_midterm.png`. Timing from compact decisions is unavailable. Use the published recovery log statistics only with the limitations in the report.

## Team report

Use `tables/merge_summary.csv` with `subset=all` and `n=4` for headline results. Use `tables/paired_thinking.csv` for matched-prompt comparisons. UltraFeedback measures quality using GPT-4 ratings. Report its metrics separately from Arena and MT-Bench human agreement.

The analysis uses a post-hoc parser that accepts consistent explicit A/B labels. Original strict retries influenced the saved answers. Ambiguous verdicts remain missing. The trial and recovery Colab notebooks document GPU inference. The report-team ZIP distributed by the contributor contains the full original and recovered output logs. This folder includes compact decisions and the 25 frozen inputs for independent analysis.

Dataset: [openbmb/UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback). Dataset and model revisions are in `data/`.
