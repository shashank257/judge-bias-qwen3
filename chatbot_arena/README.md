# Position and Format Bias in Qwen3 Judges (DSC 261, Group 3)

> **This part: Chatbot Arena** (data pipeline, judge harness, position/format experiments).
> Teammates add the MT-Bench and UltraFeedback parts in their own folders of this repository.

Do LLM judges change their verdict when only the **order** of two answers or their **markdown formatting**
changes? We test Qwen3 judges (0.6B, 1.7B, 4B, 8B; thinking off and on) on Chatbot Arena answer pairs with
content-preserving edits and a measured noise floor. Proposal: `../course/DSC261_Project_Proposal_G3.pdf`.

**Midterm status:** 100 frozen Arena pairs x 10 judge calls x 8 judge setups = 8,000 calls (0 invalid).
Results, caveats and progress-vs-plan: **`share/DSC261_G3_midterm_results_pack.pdf`**.

## Quick start: reproduce the midterm results (any OS, no GPU, ~3 min)
```bash
cd chatbot_arena
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash scripts/reproduce_midterm.sh
```
Rebuilds `report/results.csv`, `report/stats.json`, `report/extra.json`, the figures (`report/fig_*.pdf`) and,
if `pdflatex` is installed, the results-pack PDF, all from the committed raw judge logs in `api/`.

## Re-running the judges
| Hardware | Setup | Command |
|---|---|---|
| Apple-silicon Mac (MLX) | `pip install -r requirements-mac.txt` | `MIDTERM_SIZES=0.6B,1.7B bash scripts/run_midterm_supervised.sh` |
| NVIDIA GPU, >=24 GB (vLLM) | `pip install -r requirements-gpu.txt`, then `vllm serve Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --dtype bfloat16 --max-model-len 12288 --seed 261` | `python scripts/run_midterm_remote.py --url http://localhost:8000/v1 --size 8B` |

Runs are resumable (finished request ids are skipped) and log every call to `api/`. Details for other machines:
`remote_8b/RUN_8B_ELSEWHERE.md`.

## Full pipeline (data -> results)
Notebooks run in order, locally or in Colab (clone the repo to `MyDrive/dsc261`; this part runs from `chatbot_arena/`):
1. `01_data_prep`: download Arena 2023/2025 at pinned revisions, filter (counts per step), freeze pools (seed 261)
2. `02_presentation_validation`: markdown stripping, word-sequence check, audit, format variants
3. `03_eval_data_and_controls`: MT-Bench labels, baselines, human-markdown correlation, UltraFeedback brackets
4. `04_experiment_spec`: judge prompt, request manifest (151,296 calls for the full study)
5. `05_api_comparison_and_pilot`: provider catalogs, compatibility checks, pilot, cost
6. `06_midterm_results`: midterm tables and figures

`data/raw/` and two large intermediate files are not committed; notebook 01 re-downloads and rebuilds them.
The frozen pools, the 100 midterm pairs and all judge logs **are** committed.

**Arena 2023 text is not distributed.** The 2023 release is gated on Hugging Face, so the question/answer text
of its rows is blank in `data/processed/` and the judges' free-text outputs on those pairs are replaced by a
SHA-256 fingerprint in `api/`. IDs, human labels, lengths, verdicts and token counts are kept, so
`reproduce_midterm.sh` gives identical results. To re-run judges on 2023 pairs, accept the dataset terms and run
`HF_TOKEN=... python scripts/restore_arena2023_text.py` (restores the text exactly; verified).

## Repository layout
| Path | Contents |
|---|---|
| `src/dsc261/` | library: `markdown.py` (strip + validate), `arena.py` (filters, pools), `judge.py` (prompt, Qwen3 template, thinking budget, MLX/vLLM backends), `manifest.py`, `analysis.py`, `evaldata.py`, `common.py` |
| `scripts/` | judge runners, notebook builders, `report_stats.py` / `report_extra.py`, `reproduce_midterm.sh` |
| `notebooks/` | 01-06, executed with outputs |
| `data/processed/` | frozen pools, 100 midterm pairs, labels, UltraFeedback set, `freeze_manifest.json` (sha256) |
| `data/manifest/` | full-study request manifest |
| `api/` | raw judge logs (every call with its full output), provider catalog, cost projection |
| `report/` | midterm numbers (`results.csv`, `runs.csv`, `stats.json`, `extra.json`), figures, results-pack LaTeX |
| `reports/` | week-1 tables: filter counts, baselines, strip validation, audit |
| `docs/` | evaluation contract, week-1 report, API comparison, dataset records |
| `share/` | results pack PDF and figures for the team |

## Notes
- Judge setups: 0.6B/1.7B in BF16, 4B in 8-bit (MLX), 8B as Qwen's official 4-bit AWQ (vLLM, RTX 3070). The
  contract specifies BF16, so 4B/8B results are reported separately.
- Data: Chatbot Arena 2023 (lmsys/chatbot_arena_conversations, gated; text not redistributed here) and 2025
  (lmarena-ai/arena-human-preference-140k, CC-BY-4.0). The prepared MT-Bench and UltraFeedback inputs live with
  the teammates' parts.
- AI usage: Claude helped write the code and drafts; all numbers are computed by the scripts in this repo.
