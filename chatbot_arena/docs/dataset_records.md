# Dataset records

Machine-readable version, with the sha256 of every downloaded file: `data/dataset_records.json`.
Preprocessing version log: `data/version_log.jsonl`. Seed: **261** everywhere.

| Key | HF repo | Pinned revision | License | Rows | Downloaded | Role |
|---|---|---|---|---|---|---|
| arena2023 (used) | `agie-ai/lmsys-chatbot_arena_conversations` (public mirror) | `d45bf836ec5a09d8a98881b4ae16347ffb6aae9b` | same terms as the official set | 33,000 | 2026-11-03 | main pool |
| arena2023 (official) | `lmsys/chatbot_arena_conversations` (gated) | `1b6335d42a1d2c7e34870c905d03ab964f7f2bd8` | CC-BY-4.0 / CC-BY-NC-4.0 for outputs of some models; user agreement | 33,000 | needs HF token | notebook 01 uses it automatically when `HF_TOKEN` is set |
| arena2025 | `lmarena-ai/arena-human-preference-140k` | `6322995ab34d7c2693e3f47dd13fa5caa0789a74` | CC-BY-4.0 | 135,634 | 2026-11-03 | supplement for the format pool |
| mtbench_human | `lmsys/mt_bench_human_judgments` (human split) | `f7d2896d2cc5d80f8b55c2bbc722613555233c25` | CC-BY-4.0 | 3,355 votes | 2026-11-03 | human agreement |
| ultrafeedback | `openbmb/UltraFeedback` (sharegpt, ultrachat, evol_instruct files) | `40b436560ca83a8dba36114c22ab3c66e43f6d5e` | MIT | 39,878 prompts | 2026-11-03 | H3 best-of-n |

Notes:
- **Mirror check.** The mirror has the official shape (33,000 × 13) and the same column names and label values.
  It contains 20 repeated `question_id`s; we have not checked whether the official file has them too. Our
  `pair_id` gives repeats a `#k` suffix. Re-run notebook 01 with an accepted token to switch to the official file. The notebook records which
  source it used in `freeze_manifest.json → arena2023_source`.
- **2025 Arena.** "140K" is the release name; the release actually has 135,634 rows. The proposal's CC-BY-4.0
  license is confirmed.
- **UltraFeedback.** The TruthfulQA, FLAN and FalseQA files are never downloaded. Gold scores we recompute match the
  provided `fine-grained_score` for 100% of answers. The 8,248 "N/A" honesty ratings make their prompts
  ineligible.
