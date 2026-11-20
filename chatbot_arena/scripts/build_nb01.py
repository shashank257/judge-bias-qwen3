from nb_common import build

cells = [
r"""# 01 — Data preparation: Arena pools (Week 1)
Loads the 2023 Chatbot Arena human-preference data (main pool) and the 2025 Arena release (supplement),
applies the proposal's filters with a count after every step, builds the **position-bias** and
**format-bias** pools, and freezes them (seed **261**). Every model configuration later receives exactly
these frozen pairs.

Outputs: `data/processed/*.parquet`, `reports/arena_filter_counts.csv`, `data/dataset_records.json`,
`data/version_log.jsonl`, `data/processed/freeze_manifest.json`.""",

r"""MD:## 1. Dataset records (source, license, pinned revision, download date)
The official `lmsys/chatbot_arena_conversations` repo is gated (click-through). If `HF_TOKEN` is set and the
terms are accepted we load it; otherwise we load a public, unmodified mirror and check it is structurally
identical (33,000 rows, 13 columns, same ids). Re-run with a token before the final freeze to confirm.""",

r"""from dsc261 import arena as A, markdown as M, judge as J
token = os.environ.get("HF_TOKEN") or None
try:
    files23 = C.download("arena2023_official", token=token); key23 = "arena2023_official"
except Exception as e:
    print("official 2023 Arena not accessible without accepted terms/token ->", type(e).__name__, "; using mirror")
    files23 = C.download("arena2023_mirror"); key23 = "arena2023_mirror"
raw23 = pd.read_parquet(files23[0])
assert raw23.shape == (33000, 13), raw23.shape
assert list(raw23.columns) == ["question_id","model_a","model_b","winner","judge","conversation_a","conversation_b",
                               "turn","anony","language","tstamp","openai_moderation","toxic_chat_tag"]
C.record_dataset(key23, files23, len(raw23), "2023 Chatbot Arena human-preference conversations (main pool)")
files25 = C.download("arena2025")
raw25 = pd.concat([pd.read_parquet(f) for f in files25], ignore_index=True)
C.record_dataset("arena2025", files25, len(raw25), "2025 Arena release (supplement for the format pool)")
print(key23, raw23.shape, "| arena2025", raw25.shape)
raw23.winner.value_counts()""",

r"""MD:## 2. Fields and vote labels
2023: `winner ∈ {model_a, model_b, tie, tie (bothbad)}`, `turn` = number of user turns, `language` from the
dataset. 2025: `winner ∈ {model_a, model_b, tie, both_bad}`, content is a list of typed parts (text/image),
`language` is an ISO code.""",

r"""print(raw25.winner.value_counts().to_dict())
print("2023 turn counts:", raw23.turn.value_counts().sort_index().head(6).to_dict())
print("2023 duplicate question_id rows:", raw23.question_id.duplicated().sum())
print("2025 duplicate id rows:", raw25.id.duplicated().sum())""",

r"""MD:## 3. Filters (counts after every step)
**Definitions.**
- *English*: dataset language label is English **and** ≥95% of alphabetic characters in the prompt and in both
  answers are Latin script (catches replies that drift into another language).
- *Single-turn*: exactly one user message and one assistant message on both sides, same prompt on both sides.
- *Non-tied*: `winner ∈ {model_a, model_b}`; ties, both-bad and anything else are dropped.
- *Answer length*: number of whitespace-separated words **after removing the supported markdown markers**
  (so the length is identical for the markdown and plain variant). Ratio = longer / shorter ≤ 2.
- *Tables / code blocks*: fenced (```/~~~) or indented code blocks, or markdown tables in either answer → excluded.
- *Duplicates*: exact (prompt, answer A, answer B) duplicates removed. Repeated prompts are **kept** and share a
  `prompt_id` (hash of the normalised prompt) for clustered statistics.""",

r"""log = A.CountLog()
f23 = A.apply_filters(A.normalise(raw23, "arena2023"), "arena2023", log)
f25 = A.apply_filters(A.normalise(raw25, "arena2025"), "arena2025", log)
counts = log.frame()
counts""",

r"""print("2023 eligible:", len(f23), "| unique prompts:", f23.prompt_id.nunique(),
      "| repeated-prompt pairs:", int(f23.prompt_id.duplicated(keep=False).sum()))
print("2025 eligible:", len(f25), "| unique prompts:", f25.prompt_id.nunique())
for name, d in (("2023", f23), ("2025", f25)):
    print(name, "markdown in both:", int((d.md_a & d.md_b).sum()), "| in exactly one:", int((d.md_a ^ d.md_b).sum()),
          "| in neither:", int((~d.md_a & ~d.md_b).sum()))""",

r"""MD:## 4. Context-fit filter (new, needed for hosted APIs)
The judge prompt (system + question + both answers, markdown variant = longest) must fit in **6,000 tokens**
(Qwen3 tokenizer) so that prompt + 2,048 thinking tokens + verdict fit comfortably in every provider's
context and cost stays predictable. Logged as its own step.""",

r"""MAX_PROMPT_TOKENS = 6000
def prompt_tokens(d):
    tok = J.get_tokenizer("8B")
    texts = [J.render_prompt(J.build_messages(q, a, b), "8B", False) for q, a, b in zip(d.question, d.answer_a, d.answer_b)]
    return [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]
f23["prompt_tokens_md"] = prompt_tokens(f23)
f25["prompt_tokens_md"] = prompt_tokens(f25)
f23 = log.step("arena2023", "context_fit", f23, f23[f23.prompt_tokens_md <= MAX_PROMPT_TOKENS], f"judge prompt <= {MAX_PROMPT_TOKENS} tokens")
f25 = log.step("arena2025", "context_fit", f25, f25[f25.prompt_tokens_md <= MAX_PROMPT_TOKENS], f"judge prompt <= {MAX_PROMPT_TOKENS} tokens")
print(f23.prompt_tokens_md.describe().round(0).to_dict()); print(f25.prompt_tokens_md.describe().round(0).to_dict())""",

r"""MD:## 5. Markdown stripping validated **before** freezing
Only pairs whose both answers strip cleanly (word sequence preserved, no supported markdown left) can enter the
format pool. Failures are saved to `reports/strip_failures.csv` and excluded.""",

r"""md23 = A.add_stripped(f23[f23.md_a & f23.md_b])
md25 = A.add_stripped(f25[f25.md_a & f25.md_b])
fails = []
for d in (md23, md25):
    for s in ("a", "b"):
        x = d[~d[f"strip_ok_{s}"]]
        fails.append(pd.DataFrame({"pair_id": x.pair_id, "side": s, "reason": x[f"strip_fail_{s}"], "first_diff": x[f"strip_diff_{s}"]}))
fails = pd.concat(fails, ignore_index=True)
fails.to_csv(C.path("reports", "strip_failures.csv"), index=False)
print("strip failures:", len(fails), fails.reason.value_counts().to_dict())
fmt23 = log.step("arena2023", "format:md_both_strip_ok", md23, md23[md23.strip_ok_a & md23.strip_ok_b], "markdown in both answers and both strip cleanly")
fmt25 = log.step("arena2025", "format:md_both_strip_ok", md25, md25[md25.strip_ok_a & md25.strip_ok_b], "markdown in both answers and both strip cleanly")""",

r"""MD:## 6. Build and freeze the pools
- **Position pool**: 1,000 pairs from the **2023** eligible set (main pool), one pair per prompt cluster, seed 261.
- **Format pool**: markdown in **both** answers. The 2023 set has far fewer than 1,000 such pairs, so (as the
  proposal pre-registers) we add 2025 pairs under the same filters and keep a `source` label: all 2023 format pairs
  + 2025 pairs up to 1,000.
- **Reserves** (200 each) replace pairs that fail the manual audit, so pool sizes never drop below 1,000.
- **Pilot subset**: 100 pairs drawn from the format pool (supports both position and format conditions).""",

r"""N_POOL, N_RESERVE, N_PILOT = 1000, 200, 100
pos_cand = log.step("arena2023", "position:one_per_prompt", f23, A.one_per_prompt(f23), "one pair per prompt cluster (seeded)")
pos_all = A.freeze_sample(pos_cand, N_POOL + N_RESERVE)
rng = np.random.RandomState(C.SEED)
pos_all["reserve"] = False
pos_all.loc[rng.choice(len(pos_all), N_RESERVE, replace=False), "reserve"] = True
position_pool, position_reserve = pos_all[~pos_all.reserve].reset_index(drop=True), pos_all[pos_all.reserve].reset_index(drop=True)

f23u = log.step("arena2023", "format:one_per_prompt", fmt23, A.one_per_prompt(fmt23), "one pair per prompt cluster (seeded)")
f25x = log.step("arena2025", "format:prompt_not_in_2023", fmt25, fmt25[~fmt25.prompt_id.isin(f23u.prompt_id)], "prompt not already used by a 2023 format pair")
f25u = log.step("arena2025", "format:one_per_prompt", f25x, A.one_per_prompt(f25x), "one pair per prompt cluster (seeded)")
n25 = max(0, N_POOL - len(f23u))
f25_take = A.freeze_sample(f25u, n25 + N_RESERVE)
f25_take["reserve"] = False
f25_take.loc[np.random.RandomState(C.SEED).choice(len(f25_take), N_RESERVE, replace=False), "reserve"] = True
f23u = f23u.assign(reserve=False)
fmt_all = pd.concat([f23u, f25_take], ignore_index=True).sort_values("pair_id").reset_index(drop=True)
format_pool, format_reserve = fmt_all[~fmt_all.reserve].reset_index(drop=True), fmt_all[fmt_all.reserve].reset_index(drop=True)
pilot = A.freeze_sample(format_pool, N_PILOT)
print("position pool:", len(position_pool), position_pool.source.value_counts().to_dict(), "| reserve", len(position_reserve))
print("format pool:", len(format_pool), format_pool.source.value_counts().to_dict(), "| reserve", len(format_reserve))
print("pilot:", len(pilot), pilot.source.value_counts().to_dict())
assert len(position_pool) >= 1000 and len(format_pool) >= 1000
assert position_pool.prompt_id.is_unique and format_pool.prompt_id.is_unique
assert fmt_all.pair_id.is_unique and pos_all.pair_id.is_unique""",

r"""MD:## 7. Save in the common record format and freeze
Common record: `prompt_id, pair_id, source, question, answer_a, answer_b, model_a, model_b, human_pref,
len_a_words, len_b_words, len_ratio, md_a, md_b` + per-feature counts. The format pool also stores
`plain_a, plain_b` (stripped variants).""",

r"""import json
FMT_COLS = A.RECORD_COLS + ["prompt_tokens_md", "plain_a", "plain_b"]
POS_COLS = A.RECORD_COLS + ["prompt_tokens_md"]
outs = {
    "position_pool": position_pool[POS_COLS], "position_reserve": position_reserve[POS_COLS],
    "format_pool": format_pool[FMT_COLS], "format_reserve": format_reserve[FMT_COLS], "pilot_pairs": pilot[FMT_COLS],
    "arena_eligible_2023": f23[POS_COLS], "arena_eligible_2025": f25[POS_COLS],
}
paths = {}
for k, d in outs.items():
    p = C.path("data", "processed", f"{k}.parquet"); d.to_parquet(p, index=False); paths[k] = p
counts = log.frame(); counts.to_csv(C.path("reports", "arena_filter_counts.csv"), index=False)
freeze = {"frozen_utc": pd.Timestamp.utcnow().isoformat(), "seed": C.SEED, "pipeline": C.PIPELINE_VERSION,
          "strip_version": M.STRIP_VERSION, "arena2023_source": key23,
          "files": {k: {"rows": int(len(outs[k])), "sha256": C.sha256_file(p)} for k, p in paths.items()}}
json.dump(freeze, open(C.path("data", "processed", "freeze_manifest.json"), "w"), indent=2)
C.log_version("01_data_prep", f"Arena filters + pools (2023 source={key23}); context fit <= {MAX_PROMPT_TOKENS} tok", list(paths.values()))
counts""",
]

build(cells, "notebooks/01_data_prep.ipynb", "01_data_prep")
