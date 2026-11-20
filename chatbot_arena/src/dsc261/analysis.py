"""Measures from the evaluation contract (section 5), computed from the judge log.

Verdicts are mapped back to the *original answer* that won ('a' or 'b'); an
invalid verdict is None and counts as a flip in both edited and noise-floor
comparisons (contract section 4).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

SEED = 261


def load_verdicts(log_path: str) -> pd.DataFrame:
    """Final state per request id: the parsed verdict if any non-error attempt
    has one, else invalid (two unparseable attempts). Error-only ids are dropped."""
    recs = {}
    with open(log_path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("error") is not None:
                continue
            recs.setdefault(r["request_id"], []).append(r)
    rows = []
    for rid, rs in recs.items():
        ok = [r for r in rs if r.get("verdict")]
        if not ok and len(rs) < 2:
            continue  # unfinished: one unparseable attempt, retry pending
        r = ok[-1] if ok else rs[-1]
        pair, cell, cfg, samp, prompt, rep = rid.split("|")
        s1, s2 = cell.split(">")
        a1, v1 = s1.split(":")
        a2, v2 = s2.split(":")
        v = r.get("verdict")
        rows.append({"request_id": rid, "pair_id": pair, "config": cfg, "sampling": samp[2:], "prompt": prompt[2:],
                     "repeat": int(rep[1:]), "slot1": a1, "slot1_var": v1, "slot2": a2, "slot2_var": v2,
                     "verdict": v, "winner": (a1 if v == "A" else a2) if v else None,
                     "n_attempts": len(rs), "truncated": bool(r.get("truncated_thinking")),
                     "out_tokens": r.get("n_completion_tokens"), "think_tokens": r.get("n_think_tokens"),
                     "latency_s": r.get("latency_s")})
    return pd.DataFrame(rows)


def main_runs(v: pd.DataFrame) -> pd.DataFrame:
    """Drop sampling-check runs (thinking off with thinking-mode sampling); they share
    the config name with the main thinking-off runs and must never be pooled with them."""
    return v[~(v.config.str.endswith("-nothink") & (v.sampling == "think"))]


def _flip(x, y):
    """Different winner, or either side invalid -> flip (1.0)."""
    return 1.0 if (x is None or y is None or x != y) else 0.0


def per_pair_table(v: pd.DataFrame, config: str, sampling: str | None = None) -> pd.DataFrame:
    """One row per pair with complete position and (if present) format cells.
    `sampling` defaults to the mode's own profile, so sampling-check runs
    (thinking off with thinking-mode sampling) are never mixed into main results."""
    sampling = sampling or ("think" if config.endswith("-think") else "nothink")
    d = v[(v.config == config) & (v.sampling == sampling)]
    key = lambda r: (r.slot1, r.slot1_var, r.slot2, r.slot2_var, r.repeat)  # noqa: E731
    out = []
    for pid, g in d.groupby("pair_id"):
        w = {key(r): r.winner for r in g.itertuples()}
        verd = {key(r): r.verdict for r in g.itertuples()}
        need = [("a", "orig", "b", "orig", 0), ("b", "orig", "a", "orig", 0),
                ("a", "orig", "b", "orig", 1), ("b", "orig", "a", "orig", 1)]
        if not all(k in w for k in need):
            continue
        ab0, ba0, ab1, ba1 = (w[k] for k in need)
        row = {"pair_id": pid,
               "pos_flip": _flip(ab0, ba0),
               "noise_flip": (_flip(ab0, ab1) + _flip(ba0, ba1)) / 2,
               "slot1_share": np.mean([verd[k] == "A" for k in need[:2] if verd[k]]) if any(verd[k] for k in need[:2]) else np.nan,
               "winner_ab0": ab0, "winner_ba0": ba0,
               "invalid_any": any(verd[k] is None for k in need)}
        fmt_keys = {"ab": [("a", "plain", "b", "orig", 0), ("a", "orig", "b", "plain", 0)],
                    "ba": [("b", "orig", "a", "plain", 0), ("b", "plain", "a", "orig", 0)]}
        if all(k in w for ks in fmt_keys.values() for k in ks):
            flips, md_pick = [], []
            for order, base in (("ab", ab0), ("ba", ba0)):
                for k in fmt_keys[order]:
                    flips.append(_flip(base, w[k]))
                    plain_ans = k[0] if k[1] == "plain" else k[2]
                    if w[k] is not None:
                        md_pick.append(w[k] != plain_ans)  # winner is the markdown-kept answer
            row.update({"fmt_flip": float(np.mean(flips)), "md_pref": float(np.mean(md_pick)) if md_pick else np.nan,
                        "invalid_any": row["invalid_any"] or any(w[k] is None for ks in fmt_keys.values() for k in ks)})
        out.append(row)
    return pd.DataFrame(out)


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = SEED, stat=np.mean):
    """Percentile CI, resampling pairs (each pair is its own prompt cluster:
    pools keep one pair per prompt)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    s = stat(values[idx], axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def summarize(t: pd.DataFrame) -> dict:
    out = {"pairs": len(t)}
    if not len(t):
        return out
    pe = (t.pos_flip - t.noise_flip).to_numpy()
    out.update({"pos_flip_rate": t.pos_flip.mean(), "noise_floor": t.noise_flip.mean(),
                "pos_excess": pe.mean(), "pos_excess_lo": bootstrap_ci(pe)[0], "pos_excess_hi": bootstrap_ci(pe)[1],
                "slot1_pref": t.slot1_share.mean(), "slot1_pref_lo": bootstrap_ci(t.slot1_share)[0],
                "slot1_pref_hi": bootstrap_ci(t.slot1_share)[1]})
    if "fmt_flip" in t and t.fmt_flip.notna().any():
        f = t[t.fmt_flip.notna()]
        fe = (f.fmt_flip - f.noise_flip).to_numpy()
        out.update({"fmt_pairs": len(f), "fmt_flip_rate": f.fmt_flip.mean(), "fmt_excess": fe.mean(),
                    "fmt_excess_lo": bootstrap_ci(fe)[0], "fmt_excess_hi": bootstrap_ci(fe)[1],
                    "md_pref": f.md_pref.mean(), "md_pref_lo": bootstrap_ci(f.md_pref)[0], "md_pref_hi": bootstrap_ci(f.md_pref)[1]})
    return out


def human_agreement(t: pd.DataFrame, human: pd.Series) -> dict:
    """Accuracy and Cohen's kappa of the original-order verdict vs the human
    vote; swap-and-average credits 0.5 for an order-split pair."""
    h = human.reindex(t.pair_id).map({"A": "a", "B": "b"}).to_numpy()
    w = t.winner_ab0.to_numpy()
    valid = np.array([x is not None for x in w])
    acc = float(np.mean(w[valid] == h[valid])) if valid.any() else np.nan
    kappa = float(cohen_kappa_score(h[valid], w[valid].astype(str))) if valid.sum() > 1 else np.nan
    swap = np.mean([(0.5 * ((x == hh) + (y == hh))) for x, y, hh in zip(t.winner_ab0, t.winner_ba0, h)])
    return {"human_acc": acc, "human_kappa": kappa, "swap_avg_agreement": float(swap)}
