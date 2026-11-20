"""MT-Bench human votes, UltraFeedback best-of-n candidates, baselines."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from . import markdown as md
from .arena import latin_share, CountLog

SEED = 261
UF_SOURCES = ["sharegpt", "ultrachat", "evol_instruct"]  # TruthfulQA / FLAN / FalseQA files never loaded
ASPECTS = ["helpfulness", "honesty", "instruction_following", "truthfulness"]
GPT4_MODELS = {"gpt-4"}


# ---- MT-Bench --------------------------------------------------------------------
def load_mtbench(path: str, log: CountLog) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_parquet(path)
    checks = {}
    d = log.step("mtbench", "loaded", raw, raw, "human judgments (all turns)")
    qa = [list(x)[0]["content"] for x in d.conversation_a]
    qb = [list(x)[0]["content"] for x in d.conversation_b]
    checks["user_turn1_identical_a_b"] = bool(all(a == b for a, b in zip(qa, qb)))
    d = d.assign(question=qa)
    checks["one_question_text_per_question_id"] = bool((d.groupby("question_id").question.nunique() == 1).all())
    checks["n_question_ids"] = int(d.question_id.nunique())
    lens = d.conversation_a.map(len)
    checks["conversation_lengths"] = lens.value_counts().to_dict()
    roles_ok = d.conversation_a.map(lambda c: [m["role"] for m in c] == ["user", "assistant"] * (len(c) // 2))
    checks["roles_alternate_user_assistant"] = bool(roles_ok.all())
    checks["turn_values"] = d.turn.value_counts().to_dict()
    checks["winner_values"] = d.winner.value_counts().to_dict()
    d = log.step("mtbench", "turn_1_only", d, d[d.turn == 1], "single-turn: verdicts about the first answer only")
    d = d.assign(answer_a=[list(x)[1]["content"] for x in d.conversation_a],
                 answer_b=[list(x)[1]["content"] for x in d.conversation_b])
    d = log.step("mtbench", "non_tie", d, d[d.winner.isin(["model_a", "model_b"])], "ties dropped (judge has no tie option)")
    d = d.assign(human_pref=d.winner.map({"model_a": "A", "model_b": "B"}),
                 pair_key=[f"q{q}|{'|'.join(sorted([a, b]))}" for q, a, b in zip(d.question_id, d.model_a, d.model_b)],
                 len_a_words=d.answer_a.map(md.word_count), len_b_words=d.answer_b.map(md.word_count),
                 md_a=d.answer_a.map(lambda t: md.features(t).has_supported_md),
                 md_b=d.answer_b.map(lambda t: md.features(t).has_supported_md))
    d["prompt_id"] = "mtq" + d.question_id.astype(str)
    d["vote_id"] = [f"mtbench:{i}" for i in range(len(d))]
    return d.reset_index(drop=True), checks


# ---- baselines --------------------------------------------------------------------
def random_verdicts(n: int, seed: int = SEED) -> np.ndarray:
    return np.where(np.random.RandomState(seed).rand(n) < 0.5, "A", "B")


def length_verdicts(len_a, len_b, seed: int = SEED) -> np.ndarray:
    """Longer answer (words) wins; exact length ties broken by a seeded coin."""
    la, lb = np.asarray(len_a), np.asarray(len_b)
    coin = random_verdicts(len(la), seed + 1)
    return np.where(la > lb, "A", np.where(lb > la, "B", coin))


def agreement(pred, human) -> dict:
    pred, human = np.asarray(pred), np.asarray(human)
    return {"n": int(len(pred)), "accuracy": float((pred == human).mean()),
            "cohen_kappa": float(cohen_kappa_score(human, pred))}


# ---- UltraFeedback -------------------------------------------------------------------
def load_ultrafeedback(paths: dict, log: CountLog) -> pd.DataFrame:
    rows = []
    for src, p in paths.items():
        with open(p) as fh:
            for i, line in enumerate(fh):
                r = json.loads(line)
                rows.append({"uf_id": f"uf:{src}:{i}", "source": r["source"], "file": src,
                             "instruction": r["instruction"], "completions": r["completions"]})
    d = pd.DataFrame(rows)
    d = log.step("ultrafeedback", "loaded", d, d, "sharegpt + ultrachat + evol_instruct files only")
    d = log.step("ultrafeedback", "source_allowed", d, d[d.source.isin(UF_SOURCES)],
                 "source field in {sharegpt, ultrachat, evol_instruct}; no TruthfulQA-derived prompts")
    d = log.step("ultrafeedback", "four_completions", d, d[d.completions.map(len) == 4], "exactly four answers")

    def usable(cs):
        resp = [(c.get("response") or "").strip() for c in cs]
        return all(resp) and len(set(resp)) == 4
    d = log.step("ultrafeedback", "four_usable_answers", d, d[d.completions.map(usable)], "four non-empty, distinct responses")

    def ratings(cs):
        out = []
        for c in cs:
            rs = []
            for a in ASPECTS:
                v = c["annotations"].get(a, {})
                v = v.get("Rating") if isinstance(v, dict) else None
                rs.append(int(v) if isinstance(v, str) and v.isdigit() and 1 <= int(v) <= 5 else None)
            out.append(rs)
        return out
    d = d.assign(ratings=d.completions.map(ratings))
    full = d.ratings.map(lambda rs: all(x is not None for r in rs for x in r))
    d = log.step("ultrafeedback", "all_16_ratings", d, d[full],
                 "all four fine-grained ratings present (1-5) for all four answers; 'N/A' -> prompt excluded")
    eng = d.apply(lambda r: latin_share(r.instruction) >= 0.95 and
                  all(latin_share(c["response"]) >= 0.95 for c in r.completions), axis=1)
    d = log.step("ultrafeedback", "english_script", d, d[eng], ">= 95% Latin letters in prompt and all answers")
    return d.reset_index(drop=True)


def expand_answers(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in d.itertuples(index=False):
        for k, (c, rs) in enumerate(zip(r.completions, r.ratings)):
            f = md.features(c["response"])
            rows.append({"uf_id": r.uf_id, "source": r.source, "answer_idx": k, "model": c["model"],
                         "response": c["response"], **{f"r_{a}": v for a, v in zip(ASPECTS, rs)},
                         "gold": float(np.mean(rs)), "fg_score_provided": c.get("fine-grained_score"),
                         "overall_score_provided": c.get("overall_score"),
                         "has_md": f.has_supported_md, "has_code_or_table": f.excluded,
                         "n_words": md.word_count(c["response"]), "is_gpt4": c["model"] in GPT4_MODELS})
    return pd.DataFrame(rows)


def make_brackets(selected: pd.DataFrame, answers: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Seeded slot assignment. n=1: slot0. n=2: winner(slot0, slot1).
    n=4: winner(winner(slot0, slot1), winner(slot2, slot3)). Every match is judged in both orders."""
    rng = np.random.RandomState(seed)
    a = answers.set_index(["uf_id", "answer_idx"])
    rows = []
    for r in selected.sort_values("uf_id").itertuples(index=False):
        perm = rng.permutation(4)
        row = {"uf_id": r.uf_id, "source": r.source}
        for s, k in enumerate(perm):
            x = a.loc[(r.uf_id, int(k))]
            row.update({f"slot{s}": f"ans{k}", f"slot{s}_model": x.model, f"slot{s}_gold": x.gold, f"slot{s}_md": bool(x.has_md)})
        golds = [row[f"slot{s}_gold"] for s in range(4)]
        mds = [row[f"slot{s}_md"] for s in range(4)]
        best = max(golds)
        tied = [s for s in range(4) if golds[s] == best]
        row.update({"gold_best_n4": best, "gold_best_n2": max(golds[:2]), "gold_best_n1": golds[0],
                    "gold_pick_md_rate_n4": float(np.mean([mds[s] for s in tied])),  # ties: average over tied picks
                    "n_gold_tied_n4": len(tied),
                    "any_gpt4": any(row[f"slot{s}_model"] in GPT4_MODELS for s in range(4))})
        rows.append(row)
    return pd.DataFrame(rows)


def resolve_match(v_xy: str | None, v_yx: str | None, uf_id: str, x: str, y: str) -> tuple[str, bool]:
    """Winner of one bracket match from the two orders.
    v_xy: verdict with x in slot 1; v_yx: verdict with y in slot 1.
    Both orders agree -> that answer. Split (position-inconsistent) or an invalid
    verdict -> answer with more valid wins; still tied -> deterministic coin from
    (uf_id, sorted pair), identical across judge configurations."""
    wins = {x: 0, y: 0}
    if v_xy in ("A", "B"):
        wins[x if v_xy == "A" else y] += 1
    if v_yx in ("A", "B"):
        wins[y if v_yx == "A" else x] += 1
    if wins[x] != wins[y]:
        return (x if wins[x] > wins[y] else y), False
    lo, hi = sorted([x, y])
    coin = int(hashlib.sha256(f"{uf_id}|{lo}|{hi}|261".encode()).hexdigest(), 16) % 2
    return (lo if coin == 0 else hi), True
