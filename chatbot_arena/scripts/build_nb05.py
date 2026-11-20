from nb_common import build

cells = [
r"""# 05 — Hosted API comparison, compatibility checks, pilot and cost (Week 1)
1. Live catalog check: which providers serve the **exact** original hybrid `Qwen/Qwen3-{0.6B,1.7B,4B,8B}`.
2. Compatibility calls (checkpoint, thinking on/off on the same weights, verdict parsing, response metadata).
   Uses the hosted API when a key is set (`FEATHERLESS_API_KEY`), otherwise the local MLX backend with the same
   pinned checkpoints so the harness logic is still verified end to end.
3. Pilot results (latency, tokens, failures, truncation) and the projected cost of the full manifest.""",

r"""MD:## 1. Live catalogs (snapshot saved to `api/catalog_snapshot.json`)""",

r"""import json, requests, re
from dsc261 import judge as J
TARGET = [f"Qwen/Qwen3-{s}" for s in ("0.6B", "1.7B", "4B", "8B")]
def get(url):
    try:
        r = requests.get(url, timeout=30); return r.status_code, (r.json() if "json" in r.headers.get("content-type", "") else r.text)
    except Exception as e:
        return None, repr(e)
cat = {}
code, d = get("https://openrouter.ai/api/v1/models")
ids = [m["id"] for m in d["data"]] if code == 200 else []
hf = {m["id"]: m.get("hugging_face_id") for m in d["data"]} if code == 200 else {}
cat["openrouter"] = {t: [i for i in ids if (hf.get(i) or "").lower() == t.lower()] for t in TARGET}
code, d = get("https://api.featherless.ai/v1/models")
fl = {m["id"]: m for m in (d.get("data", d) if isinstance(d, dict) else d)} if code == 200 else {}
cat["featherless"] = {t: ([t] if t in fl else []) for t in TARGET}
cat["featherless_pricing_per_M"] = {t: {"in": float(fl[t]["pricing"]["prompt"]) * 1e6, "out": float(fl[t]["pricing"]["completion"]) * 1e6,
                                        "context": fl[t]["context_length"]} for t in TARGET if t in fl}
code, d = get("https://router.huggingface.co/v1/models")
hfr = [m["id"] for m in d.get("data", [])] if code == 200 else []
cat["hf_inference_providers"] = {t: [i for i in hfr if i.lower() == t.lower()] for t in TARGET}
code, d = get("https://api.deepinfra.com/models/list")
di = [m.get("model_name") for m in d] if code == 200 and isinstance(d, list) else []
cat["deepinfra"] = {t: [i for i in di if i and i.lower() == t.lower()] for t in TARGET}
code, d = get("https://api.novita.ai/v3/openai/models")
nv = [m["id"] for m in d.get("data", [])] if code == 200 else []
cat["novita"] = {t: [i for i in nv if re.fullmatch(rf"qwen/{t.split('/')[1].lower()}(-fp8)?", i.lower())] for t in TARGET}
code, d = get("https://docs.together.ai/docs/serverless/models")
cat["together_serverless_docs"] = {t: ([t] if isinstance(d, str) and re.search(re.escape(t) + r"(?![\w.-])", d) else []) for t in TARGET}
cat["snapshot_utc"] = pd.Timestamp.utcnow().isoformat()
json.dump(cat, open(C.path("api", "catalog_snapshot.json"), "w"), indent=2)
avail = pd.DataFrame({p: {t.split("/")[1]: bool(v[t]) for t in TARGET} for p, v in cat.items() if isinstance(v, dict) and set(v) == set(TARGET)})
avail""",

r"""pd.DataFrame(cat.get("featherless_pricing_per_M", {})).T""",

r"""MD:## 2. Compatibility calls
For each size × thinking mode: one judgment through the production harness. Checks: the served checkpoint answers,
thinking **off** yields no reasoning text (template pre-fills an empty think block), thinking **on** yields a
`</think>`-closed block, the verdict parses, and token usage metadata is present.""",

r"""import os
key = os.environ.get("FEATHERLESS_API_KEY")
if key:
    backend = J.OpenAICompletionsBackend("https://api.featherless.ai/v1", key, {s: J.MODEL_REVISIONS[s][0] for s in J.MODEL_REVISIONS})
    sizes = ["0.6B", "1.7B", "4B", "8B"]
else:
    print("No FEATHERLESS_API_KEY -> running the same checks on the local MLX backend (pinned HF checkpoints).")
    backend = J.MLXBackend(); sizes = ["0.6B"]
pp = pd.read_parquet(C.path("data", "processed", "position_pool.parquet"))
r = pp.sort_values("prompt_tokens_md").iloc[10]  # a short pair keeps the check cheap
msgs = J.build_messages(r.question, r.answer_a, r.answer_b)
compat = []
for s in sizes:
    for th in (False, True):
        rec = J.judge_with_retry(backend, msgs, s, th, "think" if th else "nothink", f"compat|{s}|{th}",
                                 log_path=C.path("api", "compat_log.jsonl"))
        out = rec.get("output") or ""
        compat.append({"backend": backend.name, "size": s, "thinking": th, "status": rec["status"], "verdict": rec.get("verdict"),
                       "has_think_close": "</think>" in out, "reasoning_chars": len(out.split("</think>")[0]) if "</think>" in out else 0,
                       "prompt_tokens": rec.get("n_prompt_tokens"), "completion_tokens": rec.get("n_completion_tokens"),
                       "stages": len(rec.get("stages", [])), "latency_s": round(rec.get("latency_s", 0), 2)})
compat = pd.DataFrame(compat)
compat.to_csv(C.path("api", "compatibility_checks.csv"), index=False)
assert (compat[~compat.thinking].reasoning_chars == 0).all(), "thinking off must not produce reasoning"
compat""",

r"""MD:## 3. Pilot results
100 frozen pilot pairs, both orders, thinking off; first 30 pairs with thinking on (local hardware limit).
Sizes 0.6B and 1.7B. Measures latency, prompt/completion/reasoning tokens, invalid verdicts, retries, truncation
of the thinking budget, and — as a preview only — position-flip rate and first-slot preference.""",

r"""import glob
logs = []
for f in glob.glob(C.path("api", "pilot_*", "pilot_log.jsonl")):
    for line in open(f):
        x = json.loads(line); x["log"] = os.path.basename(os.path.dirname(f)); logs.append(x)
L = pd.DataFrame(logs)
L["config"] = L["size"] + "-" + np.where(L["thinking"].astype(bool), "think", "nothink")
final = L.sort_values("attempt").groupby("request_id").tail(1).copy()
final["status"] = np.where(final.verdict.notna(), "ok", "invalid")
attempts = L.groupby("request_id").size()
final["n_attempts"] = final.request_id.map(attempts)
summ = final.groupby(["log", "config"]).agg(
    requests=("request_id", "size"), invalid_rate=("status", lambda s: (s == "invalid").mean()),
    retry_rate=("n_attempts", lambda s: (s > 1).mean()), truncation_rate=("truncated_thinking", "mean"),
    latency_mean_s=("latency_s", "mean"), latency_p90_s=("latency_s", lambda s: s.quantile(0.9)),
    prompt_tok_mean=("n_prompt_tokens", "mean"), completion_tok_mean=("n_completion_tokens", "mean"),
    completion_tok_p90=("n_completion_tokens", lambda s: s.quantile(0.9)), think_tok_mean=("n_think_tokens", "mean"))
summ.round(3)""",

r"""final["pair_id"] = final.request_id.str.split("|").str[0]
final["first"] = final.request_id.str.split("|").str[1].str[0]
final["winner_answer"] = np.where(final.verdict == "A", final["first"], np.where(final.verdict == "B", np.where(final["first"] == "a", "b", "a"), None))
prev = []
for cfg, g in final.groupby("config"):
    w = g.pivot_table(index="pair_id", columns="first", values="winner_answer", aggfunc="first").dropna()
    slot1 = (g.verdict == "A").mean()
    prev.append({"config": cfg, "pairs_both_orders_valid": len(w), "position_flip_rate": (w["a"] != w["b"]).mean() if len(w) else np.nan,
                 "share_verdicts_slot1": slot1})
prev = pd.DataFrame(prev).round(3)
print("PREVIEW ONLY (no noise floor, small n, local backend) — not a test of H1/H2:")
prev""",

r"""MD:## 4. Projected cost and time for the full manifest
Input tokens per request are exact (notebook 04). Output tokens per request come from the pilot; 4B and 8B reuse
the 1.7B pilot numbers (no local pilot at those sizes; flagged as an assumption to be replaced by the hosted pilot).
Retries add the observed retry rate. Thinking-on requests that truncate need a second call (prompt re-sent).""",

r"""man = pd.read_parquet(C.path("data", "manifest", "request_manifest.parquet"))
tok_in = pd.read_parquet(C.path("data", "manifest", "request_input_tokens.parquet"))
man = man.merge(tok_in, on="request_id")
loc = summ.reset_index().set_index("config")
def out_tokens(cfg):
    size, mode = cfg.split("-")
    src = cfg if cfg in loc.index else ("1.7B-" + mode)
    return float(loc.loc[src, "completion_tok_mean"]), float(loc.loc[src, "retry_rate"]), float(np.nan_to_num(loc.loc[src, "truncation_rate"]))
rows = []
for cfg, g in man.groupby("config"):
    o, rr, tr = out_tokens(cfg)
    n = len(g) * (1 + rr)
    tin = g.in_tokens.sum() * (1 + rr) * (1 + (tr if cfg.endswith("think") else 0))
    rows.append({"config": cfg, "requests": len(g), "calls_incl_retries": n, "input_M": tin / 1e6, "output_M": n * o / 1e6,
                 "out_tok_per_req": o, "from_pilot_of": cfg if cfg in loc.index else "1.7B (assumed)"})
vol = pd.DataFrame(rows).set_index("config")
fl = cat.get("featherless_pricing_per_M", {})
price = {s: fl.get(f"Qwen/Qwen3-{s}", {}) for s in ("0.6B", "1.7B", "4B", "8B")}
vol["featherless_usd"] = [r.input_M * price[c.split("-")[0]].get("in", np.nan) + r.output_M * price[c.split("-")[0]].get("out", np.nan) for c, r in vol.iterrows()]
# Self-hosted vLLM (Modal/RunPod L4 ~ $0.80/h): ASSUMED throughput per GPU-hour, to be measured in week 2.
THROUGHPUT_OUT_TOK_S = {"0.6B": 2500, "1.7B": 1800, "4B": 1000, "8B": 600}   # batched decode, L4, bf16 (assumption)
PREFILL_TOK_S = {"0.6B": 40000, "1.7B": 20000, "4B": 9000, "8B": 5000}
vol["l4_gpu_hours"] = [(r.output_M * 1e6 / THROUGHPUT_OUT_TOK_S[c.split("-")[0]] + r.input_M * 1e6 / PREFILL_TOK_S[c.split("-")[0]]) / 3600 for c, r in vol.iterrows()]
vol["selfhost_l4_usd"] = vol.l4_gpu_hours * 0.80
tot = vol[["requests", "calls_incl_retries", "input_M", "output_M", "featherless_usd", "l4_gpu_hours", "selfhost_l4_usd"]].sum()
vol.loc["TOTAL"] = tot
vol.to_csv(C.path("api", "cost_projection.csv"))
vol.round(2)""",

r"""print("Featherless per-token total ≈ ${:,.0f}; self-hosted vLLM on L4 ≈ {:,.0f} GPU-h ≈ ${:,.0f} (throughput assumed)".format(
    tot.featherless_usd, tot.l4_gpu_hours, tot.selfhost_l4_usd))
C.log_version("05_api", "catalog snapshot, compatibility checks, pilot summary, cost projection",
              [C.path("api", "catalog_snapshot.json"), C.path("api", "cost_projection.csv")])
summ.round(3).to_csv(C.path("api", "pilot_summary.csv")); prev.to_csv(C.path("api", "pilot_position_preview.csv"), index=False)""",
]

build(cells, "notebooks/05_api_comparison_and_pilot.ipynb", "05_api_comparison_and_pilot")
