# Running the 8B judge on another machine

You only need this folder (`dsc261_8b_kit.zip`): code + the 100 frozen pairs. ~2 MB.

## A. NVIDIA GPU with >= 24 GB (Colab L4/A100, lab server) — recommended, full precision (BF16)
```bash
unzip dsc261_8b_kit.zip && cd dsc261_8b_kit
pip install -r requirements-gpu.txt          # vllm, transformers, pandas, pyarrow, openai
# terminal 1 — start the model server (downloads ~16 GB once)
vllm serve Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
     --dtype bfloat16 --max-model-len 12288 --seed 261 --port 8000
# terminal 2 — run the 2,000 judge calls (resumable; rerun the same command if it stops)
python scripts/run_midterm_remote.py --url http://localhost:8000/v1 --size 8B
```
In Colab: run the server with `!nohup vllm serve ... > vllm.log 2>&1 &`, wait until `vllm.log` says
"Application startup complete", then run the second command in the next cell.
Expected time on an L4/A100: well under an hour.

## B. Another Mac (Apple silicon) with >= 24 GB memory — 8-bit (labelled "8B-q8")
```bash
unzip dsc261_8b_kit.zip && cd dsc261_8b_kit
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-8B', revision='b968826d9c46dd6066d109eabc6255188de91218'))"
.venv/bin/python -m mlx_lm convert --hf-path <path printed above> --mlx-path models/Qwen3-8B-q8 -q --q-bits 8 --q-group-size 64
MIDTERM_SIZES=8B-q8 MIDTERM_OUT=api/midterm_8bq8_stdout.txt caffeinate -ims bash scripts/run_midterm_supervised.sh
```

## Bring the results back
- Route A: copy `api/midterm_remote/8B_log.jsonl` into the main project's `api/midterm_remote/`.
- Route B: send `api/pilot_local/pilot_log.jsonl` (Claude merges it).
Then re-run `notebooks/06_midterm_results.ipynb` — the 8B rows appear in the table and the chart.
