"""Refill the Chatbot Arena 2023 conversation text that is not distributed in this repository
(the 2023 release is gated on Hugging Face; accept its terms first).

  HF_TOKEN=hf_... python scripts/restore_arena2023_text.py      # official lmsys repo
  python scripts/restore_arena2023_text.py --mirror              # public mirror used in our runs

Rebuilds question / answers / stripped variants for every 2023 row by pair_id, using the same
normalisation (src/dsc261/arena.py) and markdown stripper (src/dsc261/markdown.py) as the pipeline,
then checks the restored text against the word counts stored in the files.
Only needed to RE-RUN judges on 2023 pairs; reproducing the reported results does not need it.
"""
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dsc261 import arena as A, common as C, markdown as M  # noqa: E402

key = "arena2023_mirror" if "--mirror" in sys.argv or not os.environ.get("HF_TOKEN") else "arena2023_official"
files = C.download(key, token=os.environ.get("HF_TOKEN") or None)
n = A.normalise(pd.read_parquet(files[0]), "arena2023").set_index("pair_id")
print(f"source: {C.DATASETS[key][0]} ({len(n)} conversations)")

for rel in ["data/processed/format_pool.parquet", "data/processed/format_reserve.parquet",
            "data/processed/pilot_pairs.parquet", "data/processed/position_pool.parquet",
            "data/processed/position_reserve.parquet"]:
    p = os.path.join(ROOT, rel)
    d = pd.read_parquet(p)
    m = d.pair_id.str.startswith("arena2023:")
    for i in d.index[m]:
        src = n.loc[d.at[i, "pair_id"]]
        d.at[i, "question"] = src.question
        for s in ("a", "b"):
            d.at[i, f"answer_{s}"] = src[f"answer_{s}"]
            if f"plain_{s}" in d.columns:
                d.at[i, f"plain_{s}"] = M.strip_markdown(src[f"answer_{s}"])
    ok = all(M.word_count(d.at[i, f"answer_{s}"]) == d.at[i, f"len_{s}_words"] for i in d.index[m] for s in "ab")
    d.to_parquet(p, index=False)
    print(f"{rel}: restored {int(m.sum())} rows; word counts match stored lengths: {ok}")

# format variants: rebuild slot texts from the restored format pool
fp = pd.read_parquet(os.path.join(ROOT, "data/processed/format_pool.parquet")).set_index("pair_id")
vp = os.path.join(ROOT, "data/processed/format_variants.parquet")
v = pd.read_parquet(vp)
txt = lambda pid, ans, var: fp.at[pid, f"answer_{ans}" if var == "orig" else f"plain_{ans}"]  # noqa: E731
m = v.pair_id.str.startswith("arena2023:") & v.pair_id.isin(fp.index)
v.loc[m, "slot1_text"] = [txt(p, a, x) for p, a, x in zip(v.pair_id[m], v.slot1_answer[m], v.slot1_variant[m])]
v.loc[m, "slot2_text"] = [txt(p, a, x) for p, a, x in zip(v.pair_id[m], v.slot2_answer[m], v.slot2_variant[m])]
v.to_parquet(vp, index=False)
print(f"format_variants: restored {int(m.sum())} rows")
