"""Batched midterm run (same design as run_midterm.py, ~several x faster).

Uses mlx_lm.batch_generate so the GPU decodes many judge calls at once.
Same rules as judge.judge_once: chat template rendered per thinking mode,
2048-token thinking budget with close-and-continue, one retry for an
unparseable verdict, same log file and record schema, resumable.
Difference: sampling is seeded per batch, not per request (logged as
`seed: null`, `backend: local-mlx-batched`).

  python scripts/run_midterm_batched.py        # env BATCH=8 (16 GB Mac), MIDTERM_SIZES=0.6B,1.7B
"""
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dsc261 import common as C, judge as J, manifest as MF  # noqa: E402

import mlx.core as mx  # noqa: E402
from mlx_lm import load, batch_generate  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

BATCH = int(os.environ.get("BATCH", "16"))  # upper bound on concurrent sequences
# KV-cache token budget across concurrent sequences (~0.11 MB/token for Qwen3 0.6B/1.7B
# in bf16 -> 36K tokens ~ 4 GB). Lowered by the supervisor after an OOM.
TOKEN_BUDGET = int(os.environ.get("TOKEN_BUDGET", "24000"))
CUR_SIZE = None
# bigger weights leave less room for the KV cache on a 16 GB Mac (measured: 36K swaps at 0.6B)
SIZE_SCALE = {"0.6B": 1.0, "1.7B": 0.7, "4B": 0.35, "4B-q8": 0.55, "8B-q8": 0.3}  # 4B: ~8 GB weights, 0.15 MB/token KV
CHUNK = 64  # requests per batch_generate call, length-sorted; log is flushed after each
# Keep MLX's buffer cache from growing between calls (measured: process grew to 12 GB
# and pushed the Mac into swap). Freed after every chunk as well.
mx.set_cache_limit(int(os.environ.get("MLX_CACHE_MB", "512")) * 1024 * 1024)
sizes = os.environ.get("MIDTERM_SIZES", "0.6B,1.7B").split(",")
pairs = pd.read_parquet(C.path("data", "processed", "pilot_pairs.parquet")).set_index("pair_id")
log_path = C.path("api", "pilot_local", "pilot_log.jsonl")


def load_prior():
    prior = {}
    if os.path.exists(log_path):
        for line in open(log_path):
            x = json.loads(line)
            if x.get("error") is None:
                prior.setdefault(x["request_id"], []).append(x)
    return prior


def gen(model, tok, prompts, max_tokens, sampler):
    """Batched generation; returns (text, n_prompt_tokens, n_out_tokens, finish, secs) per prompt.
    Concurrency is sized to the KV-cache token budget for the longest prompt in the call."""
    ids = [tok.encode(p, add_special_tokens=False) for p in prompts]
    longest = max(len(x) for x in ids) + max_tokens
    conc = max(1, min(BATCH, int(TOKEN_BUDGET * SIZE_SCALE.get(CUR_SIZE, 0.5)) // longest, len(ids)))
    t0 = time.time()
    r = batch_generate(model, tok, ids, max_tokens=max_tokens, sampler=sampler, return_token_ids=True,
                       completion_batch_size=conc, prefill_batch_size=min(conc, 4))
    dt = (time.time() - t0) / max(len(prompts), 1)  # amortised per request
    out = []
    for p_ids, text, t_ids in zip(ids, r.texts, r.token_ids):
        n = len(t_ids)
        out.append((text, len(p_ids), n, "length" if n >= max_tokens else "stop", dt))
    return out


def stage(text, n_in, n_out, finish, dt):
    return {"text": text, "n_prompt_tokens": n_in, "n_completion_tokens": n_out, "finish_reason": finish,
            "latency_s": dt, "raw_meta": {}}


def run_chunk(model, tok, size, thinking, reqs, attempt, fh):
    sp = J.SAMPLING[reqs[0]["sampling_profile"]]
    sampler = make_sampler(temp=sp["temperature"], top_p=sp["top_p"], top_k=sp["top_k"], min_p=sp["min_p"])
    mx.random.seed(J.request_seed(reqs[0]["request_id"], attempt))  # batch-level seed
    prompts = []
    for r in reqs:
        p = pairs.loc[r["pair_id"]]
        t = lambda a, v: p[f"answer_{a}"] if v == "orig" else p[f"plain_{a}"]  # noqa: E731
        msgs = J.build_messages(p.question, t(r["slot1_answer"], r["slot1_variant"]),
                                t(r["slot2_answer"], r["slot2_variant"]), r["prompt_version"])
        prompts.append(J.render_prompt(msgs, size, thinking))
    first = gen(model, tok, prompts, J.THINK_BUDGET if thinking else J.NOTHINK_MAX_TOKENS, sampler)
    recs = []
    cont_idx, cont_prompts = [], []
    for i, (r, g) in enumerate(zip(reqs, first)):
        rec = {"request_id": r["request_id"], "attempt": attempt, "seed": None, "size": size, "thinking": thinking,
               "sampling_profile": r["sampling_profile"], "backend": "local-mlx-batched",
               "truncated_thinking": False, "stages": [stage(*g)], "_full": g[0]}
        if thinking and "</think>" not in g[0]:
            rec["truncated_thinking"] = g[3] == "length"
            cont_idx.append(i); cont_prompts.append(prompts[i] + g[0] + J.EARLY_STOP_TEXT)
            rec["_full"] = g[0] + J.EARLY_STOP_TEXT
        elif thinking and g[3] == "length" and J.parse_verdict(g[0]) is None:
            cont_idx.append(i); cont_prompts.append(prompts[i] + g[0])
        recs.append(rec)
    if cont_prompts:
        for i, g in zip(cont_idx, gen(model, tok, cont_prompts, J.VERDICT_BUDGET, sampler)):
            recs[i]["stages"].append(stage(*g))
            recs[i]["_full"] += g[0]
    unresolved = []
    for r, rec in zip(reqs, recs):
        full = rec.pop("_full")
        rec["output"] = full
        rec["verdict"] = J.parse_verdict(full)
        rec["n_prompt_tokens"] = rec["stages"][0]["n_prompt_tokens"]
        rec["n_completion_tokens"] = sum(s["n_completion_tokens"] for s in rec["stages"])
        rec["n_think_tokens"] = J.count_tokens(full.split("</think>", 1)[0], size) if thinking and "</think>" in full else None
        rec["latency_s"] = sum(s["latency_s"] for s in rec["stages"])
        rec["error"] = None
        fh.write(json.dumps(rec) + "\n")
        if rec["verdict"] is None:
            unresolved.append(r)
    fh.flush()
    return unresolved


t_start = time.time()
for size in sizes:
    CUR_SIZE = size
    if size in J.QUANT_MODELS:
        model, tok = load(C.path(J.QUANT_MODELS[size][1]))
    else:
        repo, rev = J.MODEL_REVISIONS[size]
        model, tok = load(repo, revision=rev)
    # SAMPLING_CHECK=1: proposal's sampling check = thinking OFF with thinking-mode sampling
    # (separates the effect of thinking from the effect of the different sampling settings).
    check = os.environ.get("SAMPLING_CHECK") == "1"
    for thinking in ((False,) if check else (False, True)):
        prior = load_prior()
        todo = []
        reqs = (MF.format_requests(pairs.reset_index(), size, False, sampling="think", exp="sampling_check")
                if check else MF.format_requests(pairs.reset_index(), size, thinking, exp="midterm"))
        for r in reqs:
            rs = prior.get(r["request_id"], [])
            if any(x.get("verdict") for x in rs) or len(rs) >= 2:
                continue
            todo.append((r, len(rs)))  # attempt number continues from the log
        print(f"{MF.config_name(size, thinking)}: {len(todo)} to run (max conc {BATCH}, token budget {TOKEN_BUDGET})", flush=True)
        with open(log_path, "a") as fh:
            for attempt in (0, 1):
                batch = sorted([r for r, a in todo if a == attempt],
                               key=lambda r: pairs.at[r["pair_id"], "prompt_tokens_md"])
                retry = []
                for k in range(0, len(batch), CHUNK):
                    chunk = batch[k:k + CHUNK]
                    retry += run_chunk(model, tok, size, thinking, chunk, attempt, fh)
                    mx.clear_cache()
                    print(f"  attempt {attempt}: {min(k + CHUNK, len(batch))}/{len(batch)} "
                          f"({time.time() - t_start:.0f}s elapsed)", flush=True)
                todo = [(r, 1) for r in retry] + [(r, a) for r, a in todo if a == 1 and attempt == 0]
    del model
    mx.clear_cache()
print("midterm done")
