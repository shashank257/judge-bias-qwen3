from nb_common import build

cells = [
r"""# 02 — Presentation changes and validation (Week 1)
Supported markdown = **ATX headings** (`#`…`######`), **bold** (`**x**`), **bullet markers** (`-`, `*`, `+` at line
start). Stripping is deterministic regex removal of those markers only (no rewriting model). Every stripped answer
is checked automatically; 50 pairs are audited by hand. Then all 4 format combinations × 2 orders are generated
with answer identities kept so each verdict maps back to the original answer.""",

r"""MD:## 1. Stripping rules and the word-sequence check
- Heading line `## Title` → `Title` (line kept). Bullet `- item` / `  * item` → `item` (marker + nesting indent
  removed, line break kept so list items stay on separate lines). `**x**` → `x` (single line only).
- Not touched: numbered lists (`1.` is a word), italics, links, inline code, horizontal rules, `__x__`.
- **Check (independent of the stripping regexes):** split the original into whitespace tokens line by line, drop a
  heading/bullet marker token only when it opens a line, delete every `**`; the result must equal the stripped
  text's tokens exactly. Any added, removed or reordered word fails. The stripped text must also contain no
  supported markdown and no leftover `**`.""",

r"""from dsc261 import markdown as M
tests = {
    "heading+bold+bullets": "## Plan\nUse **bold** here.\n- one\n  * two (**x**)\n1. numbered stays",
    "dash in prose is not a bullet": "Paris - the capital - is big.\n- real bullet",
    "unbalanced bold fails": "**oops never closed",
    "hr kept": "***\n**Title** text",
    "dunder kept": "call __init__ first",
}
rows = []
for name, t in tests.items():
    s = M.strip_markdown(t); v = M.validate_strip(t, s)
    rows.append({"case": name, "stripped": s.replace("\n", " ⏎ "), "strip_ok": v["strip_ok"], "reason": v["strip_fail_reason"]})
pd.DataFrame(rows)""",

r"""MD:## 2. Validate every stripped answer in the frozen pools""",

r"""fp = pd.read_parquet(C.path("data", "processed", "format_pool.parquet"))
fr = pd.read_parquet(C.path("data", "processed", "format_reserve.parquet"))
val = []
for name, d in (("pool", fp), ("reserve", fr)):
    for s in ("a", "b"):
        for pid, o, p in zip(d.pair_id, d[f"answer_{s}"], d[f"plain_{s}"]):
            v = M.validate_strip(o, p); f0 = M.features(o)
            val.append({"set": name, "pair_id": pid, "side": s, **v, "md_before": f0.has_supported_md,
                        "md_after": M.features(p).has_supported_md, "n_head": f0.n_headings, "n_bold": f0.n_bold,
                        "n_bullet": f0.n_bullets, "chars_removed": len(o) - len(p)})
val = pd.DataFrame(val)
val.to_csv(C.path("reports", "strip_validation.csv"), index=False)
print("answers validated:", len(val), "| pass:", int(val.strip_ok.sum()), "| fail:", int((~val.strip_ok).sum()))
assert val.strip_ok.all() and val.md_before.all() and not val.md_after.any()
val.groupby("set")[["n_head", "n_bold", "n_bullet", "chars_removed"]].describe().T.round(1)""",

r"""print("share of answers using each feature (pool):")
pv = val[val.set == "pool"]
print({"headings": round((pv.n_head > 0).mean(), 3), "bold": round((pv.n_bold > 0).mean(), 3), "bullets": round((pv.n_bullet > 0).mean(), 3)})
print("pre-freeze failures that were excluded (from 01):")
pd.read_csv(C.path("reports", "strip_failures.csv")).reason.value_counts()""",

r"""MD:## 3. Manual audit sheet (50 pairs = 100 answers)
Seeded sample (seed 261) of 50 format-pool pairs, 10 per team member as in the collaboration plan. Each answer is
rated on: **meaning preserved**, **wording preserved**, **list structure still interpretable** (yes/no) + notes.
The filled audit is `reports/manual_audit.csv`; a failed audit pair is replaced from the reserve.""",

r"""aud = fp.sample(50, random_state=C.SEED).reset_index(drop=True)
members = ["member1", "member2", "member3", "member4", "member5"]
sheet = []
for i, r in aud.iterrows():
    for s in ("a", "b"):
        sheet.append({"audit_idx": i, "pair_id": r.pair_id, "source": r.source, "side": s, "assigned_to": members[i // 10],
                      "original": r[f"answer_{s}"], "stripped": r[f"plain_{s}"],
                      "meaning_preserved": "", "wording_preserved": "", "list_structure_interpretable": "", "notes": ""})
sheet = pd.DataFrame(sheet)
sheet.to_csv(C.path("reports", "manual_audit_sheet.csv"), index=False)
with open(C.path("reports", "manual_audit_sheet.md"), "w") as fh:
    for r in sheet.itertuples():
        fh.write(f"\n\n## [{r.audit_idx}{r.side}] {r.pair_id} ({r.assigned_to})\n\n### ORIGINAL\n\n````\n{r.original}\n````\n\n### STRIPPED\n\n````\n{r.stripped}\n````\n")
print(len(sheet), "answers in the audit sheet;", sheet.assigned_to.value_counts().to_dict())""",

r"""import os
ap = C.path("reports", "manual_audit.csv")
if os.path.exists(ap):
    a = pd.read_csv(ap)
    cols = ["meaning_preserved", "wording_preserved", "list_structure_interpretable"]
    print("audited answers:", len(a), "| pairs:", a.pair_id.nunique())
    for c in cols:
        print(c, a[c].astype(str).str.lower().value_counts().to_dict())
    bad = a[a[cols].apply(lambda c: c.astype(str).str.lower().eq("no")).any(axis=1)]  # 'na' = no list, not a failure
    print("answers with any 'no':", len(bad)); display(bad[["pair_id", "side"] + cols + ["notes"]])
else:
    print("manual audit not filled yet")""",

r"""MD:### Audit outcome → replacements
Pre-specified rule: a pair with any "no" on meaning, wording or list structure is removed from the format pool and
replaced by the next reserve pair (seeded order). The pilot subset is not changed (it only measures throughput and
is not used for any hypothesis test).""",

r"""import json
if os.path.exists(ap):
    failed = sorted(bad.pair_id.unique())
    fp = pd.read_parquet(C.path("data", "processed", "format_pool.parquet"))  # re-read the frozen pool
    to_drop = [p for p in failed if p in set(fp.pair_id)]
    if to_drop:
        repl = fr[~fr.pair_id.isin(fp.pair_id)].sample(frac=1.0, random_state=C.SEED).head(len(to_drop))
        fp = pd.concat([fp[~fp.pair_id.isin(to_drop)], repl], ignore_index=True).sort_values("pair_id").reset_index(drop=True)
        assert len(fp) == 1000 and fp.pair_id.is_unique and fp.prompt_id.is_unique
        assert not set(fp.pair_id) & set(fr[~fr.pair_id.isin(repl.pair_id)].pair_id)
        fp.to_parquet(C.path("data", "processed", "format_pool.parquet"), index=False)
        fr = fr[~fr.pair_id.isin(repl.pair_id)].reset_index(drop=True)  # moved pairs leave the reserve
        fr.to_parquet(C.path("data", "processed", "format_reserve.parquet"), index=False)
        fz = json.load(open(C.path("data", "processed", "freeze_manifest.json")))
        fz["audit_replacements"] = {"removed": to_drop, "added_from_reserve": repl.pair_id.tolist()}
        fz["files"]["format_pool"]["sha256"] = C.sha256_file(C.path("data", "processed", "format_pool.parquet"))
        fz["files"]["format_reserve"] = {"rows": int(len(fr)), "sha256": C.sha256_file(C.path("data", "processed", "format_reserve.parquet"))}
        json.dump(fz, open(C.path("data", "processed", "freeze_manifest.json"), "w"), indent=2)
        C.log_version("02_audit_replacement", f"replaced {len(to_drop)} audit-failed pairs from reserve", [C.path("data", "processed", "format_pool.parquet")])
    print("removed:", to_drop, "\nadded:", json.load(open(C.path("data", "processed", "freeze_manifest.json"))).get("audit_replacements", {}).get("added_from_reserve"))
print("format pool:", len(fp), fp.source.value_counts().to_dict())""",

r"""MD:## 4. All four format combinations × both orders, with answer identities
`slot1_answer` / `slot2_answer` say which original answer (`a` or `b`) is shown first/second, and
`slot*_variant` whether it is the markdown original (`orig`) or stripped (`plain`). A verdict `[[A]]` means the
answer in slot 1 won, so `winner_answer = slot1_answer if verdict == 'A' else slot2_answer` maps every verdict back.""",

r"""import itertools
var = []
for r in fp.itertuples(index=False):
    for va, vb in itertools.product(("orig", "plain"), repeat=2):
        for first, second in (("a", "b"), ("b", "a")):
            fv, sv = (va, vb) if first == "a" else (vb, va)
            var.append({"pair_id": r.pair_id, "prompt_id": r.prompt_id, "cell": f"A={va}/B={vb}",
                        "order": "orig" if first == "a" else "swap",
                        "slot1_answer": first, "slot1_variant": fv, "slot2_answer": second, "slot2_variant": sv})
var = pd.DataFrame(var)
txt = fp.set_index("pair_id")
def text(pid, ans, v): return txt.at[pid, f"answer_{ans}" if v == "orig" else f"plain_{ans}"]
var["slot1_text"] = [text(p, a, v) for p, a, v in zip(var.pair_id, var.slot1_answer, var.slot1_variant)]
var["slot2_text"] = [text(p, a, v) for p, a, v in zip(var.pair_id, var.slot2_answer, var.slot2_variant)]
assert len(var) == 8 * len(fp)
# identity check: every slot text is exactly the original or stripped text of the answer it claims to be
for p, a, v, t in zip(var.pair_id, var.slot1_answer, var.slot1_variant, var.slot1_text):
    assert t == text(p, a, v)
assert (var.groupby("pair_id").size() == 8).all()
var.to_parquet(C.path("data", "processed", "format_variants.parquet"), index=False)
pp = pd.read_parquet(C.path("data", "processed", "position_pool.parquet"))
pos = pd.DataFrame([{"pair_id": r.pair_id, "prompt_id": r.prompt_id, "order": o, "slot1_answer": f, "slot2_answer": s}
                    for r in pp.itertuples() for o, f, s in (("orig", "a", "b"), ("swap", "b", "a"))])
pos.to_parquet(C.path("data", "processed", "position_variants.parquet"), index=False)
C.log_version("02_presentation", "strip validation, audit sheet, 4 cells x 2 orders",
              [C.path("data", "processed", "format_variants.parquet"), C.path("data", "processed", "position_variants.parquet")])
var.groupby(["cell", "order"]).size().unstack()""",
]

build(cells, "notebooks/02_presentation_validation.ipynb", "02_presentation_validation")
