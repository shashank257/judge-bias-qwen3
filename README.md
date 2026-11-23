# Position and Format Bias in LLM Judges (DSC 261, Group 3)

Do LLM judges change their verdict when only the **order** of two answers or their **markdown formatting**
changes, and does that cost quality under best-of-n selection? We study Qwen3 judges (0.6B-8B, thinking off and
on). Proposal: [`course/DSC261_Project_Proposal_G3.pdf`](course/DSC261_Project_Proposal_G3.pdf).

## Parts of this repository
| Folder | Part | Status |
|---|---|---|
| [`chatbot_arena/`](chatbot_arena/) | Chatbot Arena: data pipeline, judge harness, position and format bias (100 pairs x 8 judge setups, 8,000 calls) | midterm results done; run `bash chatbot_arena/scripts/reproduce_midterm.sh` |
| [`mtbench/`](mtbench/) | MT-Bench: human agreement and sensitivity | Exploratory pilot with a different format intervention (bolding the first words) and prompt; not used in the midterm results. |
| [`ultrafeedback/`](ultrafeedback/) | UltraFeedback: best-of-n selection (H3) | Midterm pilot complete: 25 prompts, 6 Qwen3 judge configurations. [Results report](ultrafeedback/DSC261_G3_UltraFeedback_midterm_results_pack.pdf) · [Reproduce on CPU](ultrafeedback/README.md#reproduce-locally) |
| [`course/`](course/) | proposal | shared |

Each part is self-contained: open its README for setup, how to run, and results.
How to add your part: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Quick start (Chatbot Arena part, any OS, no GPU)
```bash
git clone https://github.com/shashank257/judge-bias-qwen3.git && cd judge-bias-qwen3/chatbot_arena
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash scripts/reproduce_midterm.sh
```

## Data and licences
Chatbot Arena 2023 (gated; text not redistributed, restore script provided), Chatbot Arena 2025 (CC-BY-4.0),
MT-Bench human judgments (CC-BY-4.0), UltraFeedback (MIT). Models: Qwen3 checkpoints from Hugging Face.
AI usage: AI assistants helped write code and drafts; all reported numbers come from the code in this repository.
