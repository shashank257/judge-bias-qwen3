# Reproduce the final UltraFeedback study

This folder contains the code and inputs for the 500-prompt study. The judges were Qwen3 0.6B, 1.7B and 4B, each with thinking on and off. The first 25 prompts carry forward the pilot judgments.

## Reproduce the published results on CPU

[Open the analysis notebook in Colab](https://colab.research.google.com/github/shashank257/judge-bias-qwen3/blob/main/ultrafeedback/final/REPRODUCE_RESULTS.ipynb) and select **Runtime → Run all**. A fresh Colab session downloads the tracked study files from GitHub. The computation runs on the saved judgments and needs no GPU, model downloads or Drive access.

For a local run, execute these commands from the repository root on macOS or Linux.

```sh
cd ultrafeedback/final
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py
```

[reproduce.py](reproduce.py) reads the frozen answers, scores, model decisions and attempts. It rebuilds the tournament for each prompt and mode, checks 9,000 outcome rows and eight tables against the published CSVs, and writes regenerated tables and two figures under `reproduced/`. The eight tables include the prompt outcomes. The bootstrap uses 10,000 prompt-level draws with seed 261.

## Run the judges on GPU

[Open the inference notebook in Colab](https://colab.research.google.com/github/shashank257/judge-bias-qwen3/blob/main/ultrafeedback/final/inference/RUN_SIX_MODES_COLAB.ipynb), select a GPU, mount Drive and run the setup. Each model and thinking mode has its own section with separately runnable parts. Run all to cover the six settings. The runner checks existing Drive checkpoints and resumes outstanding judgments. Model sampling can produce different decisions on a new run.

The notebook imports the [Python experiment code](code/src/) from the checkout and uses the [frozen judge prompt templates](code/config/judge_prompts.upstream.jsonl). The [checkpoint analysis script](inference/ANALYZE_SIX_MODES.py) exports results from the Drive checkpoints. For the original two-T4 assignment, use the [Kaggle notebook](inference/KAGGLE_1p7B_FROM_06_AND_4B_ALL_ON.ipynb) with the adjacent [frozen input](inference/uf_frozen_500_b95fe3a5d9b7.bin). That assignment covers 1.7B thinking-on parts 6–10 and 4B thinking-on parts 1–20.

## Files for the report team

- [Final PDF](DSC261_G3_UltraFeedback_Final.pdf) and [editable results](REPORT.md).
- [Prompt outcomes](results/prompt_outcomes.csv) for merging. Within this dataset the key is `(config_id, prompt_id, n)`. Preserve missing outcomes and cluster by prompt when estimating uncertainty.
- [Pooled estimates](results/pooled_h3.csv), [per-mode results](results/per_mode.csv) and [paired comparisons](results/paired_comparisons.csv). Multiply Markdown drift by 100 to convert it to percentage points.
- [Run manifest](results/manifest.json), [frozen inputs](results/frozen_inputs.jsonl), [final decisions](results/latest_decisions.csv) and [attempt records](results/recorded_attempts.csv) for auditing.
- [PDF rendering code](report/render.py). To regenerate the PDF locally after setup, run `.venv/bin/python -m pip install reportlab` and `.venv/bin/python report/render.py`. The PDF is written under `reproduced/`.
