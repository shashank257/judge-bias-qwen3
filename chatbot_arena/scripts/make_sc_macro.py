"""Writes report/numbers_sc.tex with the sampling-check sentence from report/extra.json."""
import json, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
d = json.load(open(os.path.join(ROOT, "report", "extra.json"))).get("sampling_check", {})
r = json.load(open(os.path.join(ROOT, "report", "stats.json")))
import pandas as pd
res = pd.read_csv(os.path.join(ROOT, "report", "results.csv"), index_col=0)
parts = []
for size in ("0.6B", "1.7B"):
    if size in d:
        s = d[size]
        off = res.loc[f"{size}-nothink", "pos_excess"] * 100
        on = res.loc[f"{size}-think", "pos_excess"] * 100
        parts.append(f"at {size} the excess is {s['pos_excess']*100:.1f} points (CI {s['pos_excess_lo']*100:.1f} to "
                     f"{s['pos_excess_hi']*100:.1f}), compared with {off:.1f} with thinking-off sampling and {on:.1f} "
                     f"with thinking on")
if parts:
    txt = "; ".join(parts) + ". Most of the reduction therefore comes from thinking itself rather than from the sampling settings." \
        if all(abs(d[s]["pos_excess"] * 100 - res.loc[f"{s}-nothink", "pos_excess"] * 100) <
               abs(d[s]["pos_excess"] * 100 - res.loc[f"{s}-think", "pos_excess"] * 100) for s in d) else \
        "; ".join(parts) + ". Part of the reduction is therefore attributable to the sampling settings."
else:
    txt = "this check is still running and will be reported in the final study."
open(os.path.join(ROOT, "report", "numbers_sc.tex"), "w").write("\\newcommand{\\SCtext}{" + txt + "}\n")
print(txt)
