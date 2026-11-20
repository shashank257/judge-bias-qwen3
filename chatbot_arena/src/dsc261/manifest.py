"""Request manifest: one row per judge call, for every experiment and configuration.

A request is identified by *what the judge sees and how it is sampled*:
(pair, answer shown first + its variant, answer shown second + its variant,
size, thinking, sampling profile, prompt version, repeat). Two experiments
that need the same request share it (counted once, `reused_by` lists both).
Repeats have different `repeat` values -> different request ids -> different
seeds -> fresh generations; nothing is ever served from a cache keyed on text.
"""
from __future__ import annotations

import itertools

import pandas as pd

SIZES = ["0.6B", "1.7B", "4B", "8B"]
MODES = [False, True]  # thinking off / on
MAIN_PROMPT = "mtbench_pair_v2_notie"
ALT_PROMPT = "paraphrase_v1"


def config_name(size, thinking):
    return f"{size}-{'think' if thinking else 'nothink'}"


def request_id(pair_id, first, first_var, second, second_var, size, thinking, sampling, prompt, repeat):
    return (f"{pair_id}|{first}:{first_var}>{second}:{second_var}|{config_name(size, thinking)}"
            f"|s={sampling}|p={prompt}|r{repeat}")


def _row(exp, pair_id, prompt_id, source, first, fv, second, sv, size, thinking, sampling, prompt, repeat, stage="single"):
    return {
        "request_id": request_id(pair_id, first, fv, second, sv, size, thinking, sampling, prompt, repeat),
        "experiment": exp, "pair_id": pair_id, "prompt_id": prompt_id, "source": source,
        "slot1_answer": first, "slot1_variant": fv, "slot2_answer": second, "slot2_variant": sv,
        "order": "orig" if first < second else "swap",
        "size": size, "thinking": thinking, "config": config_name(size, thinking),
        "sampling_profile": sampling, "prompt_version": prompt, "repeat": repeat, "stage": stage,
    }


def position_requests(pool: pd.DataFrame, size, thinking, sampling=None, prompt=MAIN_PROMPT, repeats=(0, 1), exp="position"):
    """Original answers (as written) in both orders; repeat 1 = noise floor."""
    sampling = sampling or ("think" if thinking else "nothink")
    rows = []
    for r in pool.itertuples(index=False):
        for rep in repeats:
            for first, second in (("a", "b"), ("b", "a")):
                rows.append(_row(exp, r.pair_id, r.prompt_id, r.source, first, "orig", second, "orig",
                                 size, thinking, sampling, prompt, rep))
    return rows


def format_requests(pool: pd.DataFrame, size, thinking, sampling=None, prompt=MAIN_PROMPT, noise_repeat=True, exp="format"):
    """2x2 (A,B) in {md, plain}^2, both orders, repeat 0; plus md/md repeated in
    both orders (noise floor). 10 requests per pair. The markdown variant *is*
    the original text, so it is labelled 'orig' and md/md requests coincide
    with position-experiment requests for any pair in both pools."""
    sampling = sampling or ("think" if thinking else "nothink")
    rows = []
    for r in pool.itertuples(index=False):
        for va, vb in itertools.product(("orig", "plain"), repeat=2):
            for first, second in (("a", "b"), ("b", "a")):
                fv, sv = (va, vb) if first == "a" else (vb, va)
                rows.append(_row(exp, r.pair_id, r.prompt_id, r.source, first, fv, second, sv,
                                 size, thinking, sampling, prompt, 0))
        if noise_repeat:
            for first, second in (("a", "b"), ("b", "a")):
                rows.append(_row(exp, r.pair_id, r.prompt_id, r.source, first, "orig", second, "orig",
                                 size, thinking, sampling, prompt, 1))
    return rows


def h3_round1_requests(brackets: pd.DataFrame, size, thinking, sampling=None, prompt=MAIN_PROMPT):
    """Best-of-4 knockout, round 1 (both semifinals, both orders). The final is
    built adaptively from round-1 winners with `h3_final_requests`."""
    sampling = sampling or ("think" if thinking else "nothink")
    rows = []
    for b in brackets.itertuples(index=False):
        for x, y in ((b.slot0, b.slot1), (b.slot2, b.slot3)):
            for first, second in ((x, y), (y, x)):
                rows.append(_row("h3", b.uf_id, b.uf_id, b.source, first, "orig", second, "orig",
                                 size, thinking, sampling, prompt, 0, stage="semi"))
    return rows


def h3_final_requests(uf_id, source, w1, w2, size, thinking, sampling=None, prompt=MAIN_PROMPT):
    sampling = sampling or ("think" if thinking else "nothink")
    return [_row("h3", uf_id, uf_id, source, f, "orig", s, "orig", size, thinking, sampling, prompt, 0, stage="final")
            for f, s in ((w1, w2), (w2, w1))]


def build_manifest(position_pool, format_pool, brackets, check_pairs):
    """Full manifest. Final-round H3 rows are placeholders (2 per prompt and
    config) because their contents depend on semifinal verdicts."""
    rows = []
    for size, th in itertools.product(SIZES, MODES):
        rows += position_requests(position_pool, size, th)
        rows += format_requests(format_pool, size, th)
        rows += h3_round1_requests(brackets, size, th)
        for b in brackets.itertuples(index=False):
            for k in range(2):
                rows.append({**_row("h3", b.uf_id, b.uf_id, b.source, "W1" if k == 0 else "W2", "orig",
                                    "W2" if k == 0 else "W1", "orig", size, th, "think" if th else "nothink",
                                    MAIN_PROMPT, 0, stage="final_placeholder")})
    # Sampling check: thinking off with thinking-mode sampling, 200 pairs per size (format cells + md/md repeats)
    for size in SIZES:
        rows += format_requests(check_pairs, size, False, sampling="think", noise_repeat=True, exp="sampling_check")
    # Prompt check: paraphrased prompt, 200 pairs at 1.7B and 8B, both modes
    for size, th in itertools.product(["1.7B", "8B"], MODES):
        rows += format_requests(check_pairs, size, th, prompt=ALT_PROMPT, noise_repeat=True, exp="prompt_check")
    m = pd.DataFrame(rows)
    # Reuse accounting: identical requests needed by several experiments are run once.
    reuse = m.groupby("request_id")["experiment"].agg(lambda s: ",".join(sorted(set(s))))
    m = m.drop_duplicates("request_id").copy()
    m["reused_by"] = m.request_id.map(reuse)
    return m.reset_index(drop=True)
