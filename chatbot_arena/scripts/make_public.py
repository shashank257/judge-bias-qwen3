"""One-off: remove gated Chatbot Arena 2023 conversation text from files that go into the
public repository. Originals are copied to data/private/ (git-ignored) first.

- parquet files: question/answer/plain/slot text of rows from the 2023 release -> "" (ids, labels,
  lengths and token counts are kept)
- judge logs: for 2023 pairs, the judge's free-text output is replaced by
  "[redacted-arena2023 sha256=<hash of the stripped output>]" so verdicts, token counts and the
  repeat-identity check are unchanged
- manual audit sheet: 2023 rows' original/stripped text -> ""
Restore the text with scripts/restore_arena2023_text.py.
"""
import hashlib
import json
import os
import shutil
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PRIV = os.path.join(ROOT, "data", "private")
TEXT = ["question", "question_b", "answer_a", "answer_b", "plain_a", "plain_b", "slot1_text", "slot2_text"]
PARQUETS = ["data/processed/format_pool.parquet", "data/processed/format_reserve.parquet",
            "data/processed/pilot_pairs.parquet", "data/processed/position_pool.parquet",
            "data/processed/position_reserve.parquet", "data/processed/format_variants.parquet"]
LOGS = ["api/pilot_local/pilot_log.jsonl", "api/midterm_remote/8B-awq_log.jsonl", "api/compat_log.jsonl"]


def backup(rel):
    dst = os.path.join(PRIV, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):  # never overwrite the first (full-text) backup
        shutil.copy2(os.path.join(ROOT, rel), dst)


def is2023(pid):
    return str(pid).startswith("arena2023:")


def redact(text):
    return f"[redacted-arena2023 sha256={hashlib.sha256(text.strip().encode()).hexdigest()}]"


for rel in PARQUETS:
    backup(rel)
    d = pd.read_parquet(os.path.join(ROOT, rel))
    m = d.pair_id.map(is2023)
    for c in [c for c in TEXT if c in d.columns]:
        d.loc[m, c] = ""
    d.to_parquet(os.path.join(ROOT, rel), index=False)
    print(f"{rel}: blanked {int(m.sum())} rows")

for rel in LOGS:
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        continue
    backup(rel)
    out, n = [], 0
    for line in open(p):
        r = json.loads(line)
        if is2023(r.get("request_id", "").split("|")[0]):
            if r.get("output"):
                r["output"] = redact(r["output"]); n += 1
            for s in r.get("stages") or []:
                if s.get("text"):
                    s["text"] = redact(s["text"])
        out.append(json.dumps(r))
    open(p, "w").write("\n".join(out) + "\n")
    print(f"{rel}: redacted {n} outputs")

for rel in ["reports/manual_audit_sheet.csv", "reports/manual_audit_sheet.md"]:
    backup(rel)
a = pd.read_csv(os.path.join(ROOT, "reports/manual_audit_sheet.csv"))
m = a.pair_id.map(is2023)
a.loc[m, ["original", "stripped"]] = "[Arena-2023 text removed; restore with scripts/restore_arena2023_text.py]"
a.to_csv(os.path.join(ROOT, "reports/manual_audit_sheet.csv"), index=False)
with open(os.path.join(ROOT, "reports/manual_audit_sheet.md"), "w") as fh:
    for r in a.itertuples():
        fh.write(f"\n\n## [{r.audit_idx}{r.side}] {r.pair_id} ({r.assigned_to})\n\n### ORIGINAL\n\n````\n{r.original}\n````"
                 f"\n\n### STRIPPED\n\n````\n{r.stripped}\n````\n")
print(f"audit sheet: blanked {int(m.sum())} rows")
