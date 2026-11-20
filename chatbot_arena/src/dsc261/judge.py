"""Pairwise judge: prompt, Qwen3 chat template, verdict parsing, thinking budget,
and backends (OpenAI-compatible hosted API, or local MLX for smoke tests).

Design: we render the Qwen3 chat template *locally* from the pinned checkpoint
revision and send raw text to a `/v1/completions` endpoint. That gives the same
`enable_thinking` switch as the model card (the template pre-fills an empty
`<think>\n\n</think>\n\n` block when thinking is off, so reasoning is disabled,
not merely hidden) and lets us enforce the 2048-token thinking budget with the
close-the-block-and-continue procedure from the proposal.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict

JUDGE_VERSION = "judge-v1.0"

MODEL_REVISIONS = {  # original hybrid checkpoints, pinned 2026-11-03
    "0.6B": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "1.7B": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    "4B": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
    "8B": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
}

# 8-bit MLX conversions made locally from the pinned revisions above (mlx_lm convert
# -q --q-bits 8 --q-group-size 64). Deviation from the BF16 contract: reported as
# separate configurations ("4B-q8", "8B-q8"), never pooled with BF16 results.
QUANT_MODELS = {"4B-q8": ("4B", "models/Qwen3-4B-q8"), "8B-q8": ("8B", "models/Qwen3-8B-q8"),
                # Qwen's official 4-bit AWQ of Qwen3-8B, served by vLLM 0.8.5.post1 on an 8 GB RTX 3070
                # Laptop (BF16 does not fit). Same tokenizer/chat template as the pinned 8B revision.
                "8B-awq": ("8B", "Qwen/Qwen3-8B-AWQ@4da05a8edb55c6046cce958586c33b61da07bb79")}

# Model-card sampling settings (Qwen3 cards): never greedy for thinking mode.
SAMPLING = {
    "think": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
    "nothink": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
}

THINK_BUDGET = 2048        # max reasoning tokens before the block is closed
VERDICT_BUDGET = 512       # tokens allowed after </think> (explanation + verdict)
NOTHINK_MAX_TOKENS = 1024  # explanation + verdict when thinking is off
# Qwen's documented phrase for stopping thinking early (budget forcing).
EARLY_STOP_TEXT = ("\n\nConsidering the limited time by the user, I have to give the "
                   "solution based on the thinking directly now.\n</think>\n\n")

# --- prompts -----------------------------------------------------------------
# MT-Bench "pair-v2" (FastChat judge_prompts.jsonl) with the tie option removed.
SYSTEM_PROMPTS = {
    "mtbench_pair_v2_notie": (
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI "
        "assistants to the user question displayed below. You should choose the assistant that follows the "
        "user's instructions and answers the user's question better. Your evaluation should consider factors "
        "such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their "
        "responses. Begin your evaluation by comparing the two responses and provide a short explanation. "
        "Avoid any position biases and ensure that the order in which the responses were presented does not "
        "influence your decision. Do not allow the length of the responses to influence your evaluation. Do "
        "not favor certain names of the assistants. Be as objective as possible. After providing your "
        "explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A "
        "is better, \"[[B]]\" if assistant B is better."
    ),
    # Paraphrase used for the prompt-sensitivity check (same content, new wording/order).
    "paraphrase_v1": (
        "You are an impartial evaluator. Below is a user question followed by two AI assistant responses. "
        "Decide which response better follows the user's instructions and better answers the question, "
        "weighing helpfulness, relevance, accuracy, depth, creativity and level of detail. The order in which "
        "the responses appear, their length, and the assistants' names must not affect your judgment. Start "
        "with a brief comparison of the two responses, then give your final verdict in exactly this format: "
        "\"[[A]]\" if Assistant A's response is better, or \"[[B]]\" if Assistant B's response is better."
    ),
}
USER_TEMPLATE = (
    "[User Question]\n{question}\n\n"
    "[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n"
    "[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"
)


def build_messages(question: str, first: str, second: str, prompt_version: str = "mtbench_pair_v2_notie"):
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[prompt_version]},
        {"role": "user", "content": USER_TEMPLATE.format(question=question, answer_a=first, answer_b=second)},
    ]


_TOKENIZERS = {}


def get_tokenizer(size: str):
    if size not in _TOKENIZERS:
        from transformers import AutoTokenizer
        repo, rev = MODEL_REVISIONS[QUANT_MODELS.get(size, (size,))[0]]
        _TOKENIZERS[size] = AutoTokenizer.from_pretrained(repo, revision=rev)
    return _TOKENIZERS[size]


def render_prompt(messages, size: str, thinking: bool) -> str:
    tok = get_tokenizer(size)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=thinking)


def count_tokens(text: str, size: str = "8B") -> int:
    return len(get_tokenizer(size).encode(text, add_special_tokens=False))


# --- verdict parsing -----------------------------------------------------------
VERDICT_RE = re.compile(r"\[\[\s*([AB])\s*\]\]")


def parse_verdict(output: str) -> str | None:
    """Fixed rule: look only after the last </think>; the verdict is the LAST
    [[A]]/[[B]] token. Anything else (no token, or only [[C]]) is invalid."""
    text = output.rsplit("</think>", 1)[-1]
    hits = VERDICT_RE.findall(text)
    return hits[-1] if hits else None


def request_seed(request_id: str, attempt: int) -> int:
    """Distinct per request id *and* attempt, so intentional repeats (different
    repeat index -> different request id) and retries get fresh samples."""
    h = hashlib.sha256(f"{request_id}|{attempt}|261".encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


# --- backends ----------------------------------------------------------------
@dataclass
class GenResult:
    text: str
    n_prompt_tokens: int | None
    n_completion_tokens: int | None
    finish_reason: str | None
    latency_s: float
    raw_meta: dict = field(default_factory=dict)


class OpenAICompletionsBackend:
    """Any OpenAI-compatible /v1/completions endpoint (Featherless, vLLM on
    Modal/RunPod/HF Inference Endpoints, ...). `model_map` maps size -> served id."""

    def __init__(self, base_url: str, api_key: str, model_map: dict, extra_body: dict | None = None,
                 timeout: float = 300.0, max_retries_http: int = 5):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries_http)
        self.model_map = model_map
        self.extra_body = extra_body or {}
        self.name = base_url

    def generate(self, prompt: str, size: str, max_tokens: int, sampling: dict, seed: int, stop=None) -> GenResult:
        t0 = time.time()
        extra = {"top_k": sampling["top_k"], "min_p": sampling["min_p"], **self.extra_body}
        r = self.client.completions.create(
            model=self.model_map[size], prompt=prompt, max_tokens=max_tokens,
            temperature=sampling["temperature"], top_p=sampling["top_p"], seed=seed,
            stop=stop, extra_body=extra)
        c = r.choices[0]
        u = getattr(r, "usage", None)
        return GenResult(c.text, getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None),
                         c.finish_reason, time.time() - t0,
                         {"id": r.id, "model": r.model, "system_fingerprint": getattr(r, "system_fingerprint", None)})


class MLXBackend:
    """Local Apple-silicon backend for harness smoke tests and the local pilot.
    Loads the same pinned HF checkpoints (bf16 weights)."""

    def __init__(self):
        self.models = {}
        self.name = "local-mlx"

    def _load(self, size):
        if size not in self.models:
            from mlx_lm import load
            repo, rev = MODEL_REVISIONS[size]
            self.models[size] = load(repo, revision=rev) if "revision" in load.__code__.co_varnames else load(repo)
        return self.models[size]

    def generate(self, prompt: str, size: str, max_tokens: int, sampling: dict, seed: int, stop=None) -> GenResult:
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        model, tok = self._load(size)
        mx.random.seed(seed)
        sampler = make_sampler(temp=sampling["temperature"], top_p=sampling["top_p"],
                               top_k=sampling["top_k"], min_p=sampling["min_p"])
        t0 = time.time()
        ids = tok.encode(prompt, add_special_tokens=False)
        out = generate(model, tok, prompt=ids, max_tokens=max_tokens, sampler=sampler, verbose=False)
        n_out = len(tok.encode(out, add_special_tokens=False))
        finish = "length" if n_out >= max_tokens else "stop"
        return GenResult(out, len(ids), n_out, finish, time.time() - t0, {})


# --- one judgment ---------------------------------------------------------------
def judge_once(backend, messages, size: str, thinking: bool, sampling_profile: str, request_id: str,
               attempt: int = 0) -> dict:
    """Run one judgment with the thinking budget. Returns a log record."""
    sampling = SAMPLING[sampling_profile]
    seed = request_seed(request_id, attempt)
    prompt = render_prompt(messages, size, thinking)
    rec = {"request_id": request_id, "attempt": attempt, "seed": seed, "size": size, "thinking": thinking,
           "sampling_profile": sampling_profile, "backend": backend.name, "truncated_thinking": False,
           "stages": []}
    if not thinking:
        g = backend.generate(prompt, size, NOTHINK_MAX_TOKENS, sampling, seed)
        rec["stages"].append(asdict(g))
        full = g.text
    else:
        g1 = backend.generate(prompt, size, THINK_BUDGET, sampling, seed)
        rec["stages"].append(asdict(g1))
        full = g1.text
        if "</think>" not in g1.text:
            # Budget exhausted inside the reasoning block: close it and ask for the verdict.
            rec["truncated_thinking"] = g1.finish_reason == "length"
            cont = prompt + g1.text + EARLY_STOP_TEXT
            g2 = backend.generate(cont, size, VERDICT_BUDGET, sampling, seed + 1)
            rec["stages"].append(asdict(g2))
            full = g1.text + EARLY_STOP_TEXT + g2.text
        elif g1.finish_reason == "length" and parse_verdict(g1.text) is None:
            # Thinking finished within budget but the answer was cut: continue it.
            g2 = backend.generate(prompt + g1.text, size, VERDICT_BUDGET, sampling, seed + 1)
            rec["stages"].append(asdict(g2))
            full = g1.text + g2.text
    rec["output"] = full
    rec["verdict"] = parse_verdict(full)
    rec["n_prompt_tokens"] = rec["stages"][0]["n_prompt_tokens"]
    rec["n_completion_tokens"] = sum((s["n_completion_tokens"] or 0) for s in rec["stages"])
    rec["n_think_tokens"] = None
    if thinking and "</think>" in full:
        rec["n_think_tokens"] = count_tokens(full.split("</think>", 1)[0], size)
    rec["latency_s"] = sum(s["latency_s"] for s in rec["stages"])
    return rec


def _prior_attempts(log_path, request_id) -> list:
    if not log_path or not os.path.exists(log_path):
        return []
    out = []
    with open(log_path) as fh:
        for line in fh:
            r = json.loads(line)
            if r["request_id"] == request_id:
                out.append(r)
    return out


def judge_with_retry(backend, messages, size, thinking, sampling_profile, request_id, log_path=None,
                     prior_attempts=None, max_errors=3) -> dict:
    """One retry on an unparseable verdict; still unparseable -> invalid.
    Provider/HTTP errors are NOT verdict attempts: they are logged with
    status 'error', do not consume the retry, and leave the request open for a
    later re-queue. On resume, attempt numbering continues from the log so a
    retry never reuses the seed of an earlier attempt."""
    prior = prior_attempts if prior_attempts is not None else _prior_attempts(log_path, request_id)
    verdict_attempts = [r for r in prior if r.get("error") is None]
    if any(r.get("verdict") for r in verdict_attempts):
        return {**[r for r in verdict_attempts if r.get("verdict")][-1], "status": "ok", "resumed": True}
    attempt = len(verdict_attempts)
    n_err = 0
    last = None
    while attempt < 2 and n_err < max_errors:
        try:
            r = judge_once(backend, messages, size, thinking, sampling_profile, request_id, attempt)
            r["error"] = None
        except Exception as e:
            r = {"request_id": request_id, "attempt": attempt, "error": repr(e)[:500], "verdict": None, "status": "error"}
        if log_path:
            with open(log_path, "a") as fh:
                fh.write(json.dumps(r) + "\n")
        last = r
        if r["error"] is not None:
            n_err += 1
            continue
        attempt += 1
        if r.get("verdict") is not None:
            break
    final = dict(last or verdict_attempts[-1])
    final["n_attempts"] = attempt
    if final.get("error") is not None:
        final["status"] = "error"  # re-queue later; not an invalid verdict
    else:
        final["status"] = "ok" if final.get("verdict") else "invalid"
    return final


def done_request_ids(log_path: str) -> set:
    """Resumability: request ids with a parsed verdict, or with two
    non-error (unparseable) attempts. Error-only requests stay open."""
    if not os.path.exists(log_path):
        return set()
    attempts = {}
    with open(log_path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("error") is None:
                attempts.setdefault(r["request_id"], []).append(r)
    return {rid for rid, rs in attempts.items() if any(x.get("verdict") for x in rs) or len(rs) >= 2}
