# How to add your part

1. **Clone** the repo: `git clone https://github.com/shashank257/judge-bias-qwen3.git` (accept the collaborator invite first).
2. **Work only inside your folder** (`mtbench/` or `ultrafeedback/`). Do not edit other parts' folders; if you need
   something from another part, copy it into your folder or ask its owner.
3. **Branch, then push:**
   ```bash
   git checkout -b mtbench            # or ultrafeedback
   git add mtbench/
   git commit -m "MT-Bench: <what you added>"
   git push -u origin mtbench
   ```
   Then open a Pull Request on GitHub into `main` (or, if the team agrees, push straight to `main` after
   `git pull --rebase`).
4. **Keep it executable:** your folder's README must say how to install (`requirements.txt`) and how to reproduce
   your tables/figures with one command or one notebook run.

## Do not commit
- Large raw data or model weights (GitHub rejects files > 100 MB; keep files < 50 MB). Document downloads instead.
- API keys / tokens (`HF_TOKEN`, OpenAI, etc.): use environment variables.
- Virtual environments, `__pycache__`, `.DS_Store` (already in `.gitignore`).
- Gated dataset text (e.g. Chatbot Arena 2023 conversations).
