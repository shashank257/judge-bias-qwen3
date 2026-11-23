"""Re-apply the original (pre-registered) verdict parser to the saved UltraFeedback outputs.

Original rule: exactly one [[...]] label in the answer, and it is [[A]] or [[B]] at the end.
Counts tournaments whose six calls (two semifinals and the final, both orders) all parse,
and gives pooled n=4 format drift on those tournaments with a prompt-cluster bootstrap.
Needs data/recovered/judge_results.jsonl from the UltraFeedback handoff package (full outputs)
and tables/merge_prompt_outcomes.csv. Run from that package's root folder.
"""
import json
import re

import numpy as np
import pandas as pd


def strict_verdict(content):
    if not isinstance(content, str):
        return None
    labels = re.findall(r"\[\[([^\]]+)\]\]", content)
    match = re.search(r"\[\[(A|B)\]\]\s*$", content)
    return match.group(1) if len(labels) == 1 and match else None


calls = pd.DataFrame(json.loads(line) for line in open("data/recovered/judge_results.jsonl"))
calls["strict"] = calls.content.map(strict_verdict)
latest = calls.drop_duplicates(["config_id", "prompt_id", "match_name", "order"], keep="last")
bracket = latest[latest.match_name.isin(["semi0", "semi1", "final"])]
complete = (bracket.groupby(["config_id", "prompt_id"])
            .apply(lambda g: len(g) == 6 and g.strict.notna().all())
            .rename("strict_complete").reset_index())
print("Strict-complete tournaments:", int(complete.strict_complete.sum()), "of", len(complete))
print(complete.groupby("config_id").strict_complete.sum().to_string())

outcomes = pd.read_csv("tables/merge_prompt_outcomes.csv")
n4 = outcomes[(outcomes.n == 4) & (~outcomes.missing)].merge(complete, on=["config_id", "prompt_id"])
sc = n4[n4.strict_complete]
rng = np.random.default_rng(261)
pids = sc.prompt_id.unique()
groups = {p: sc.loc[sc.prompt_id == p, "format_drift"].to_numpy() for p in pids}
boot = [np.concatenate([groups[p] for p in rng.choice(pids, len(pids))]).mean() for _ in range(10000)]
lo, hi = 100 * np.percentile(boot, [2.5, 97.5])
print(f"Drift on strict-complete tournaments: {100 * sc.format_drift.mean():+.1f} pp "
      f"[{lo:+.1f}, {hi:+.1f}], rows={len(sc)}, prompts={len(pids)}")
