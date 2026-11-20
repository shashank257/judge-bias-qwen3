"""Chatbot Arena loading, normalisation, filtering and pool freezing.

Every filter step appends a row to a count log so the notebook can report
retained / excluded counts after each stage.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np
import pandas as pd

from . import markdown as md

SEED = 261
MAX_LEN_RATIO = 2.0
LATIN_SHARE_MIN = 0.95  # share of alphabetic chars that are Latin, per answer


def prompt_id(text: str) -> str:
    """Cluster id for repeated prompts: hash of whitespace/case-normalised text."""
    norm = re.sub(r"\s+", " ", (text or "").strip().lower())
    return "p_" + hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def _msg_text(content) -> str | None:
    """2023 content is a str; 2025 content is a list of {type,text,...} parts."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    parts = list(content)
    if any(p.get("type") != "text" for p in parts):
        return None  # image or other non-text part -> not usable
    return "\n".join(p.get("text") or "" for p in parts)


def normalise(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map a raw Arena dump to the common record format (pre-filter)."""
    rows = []
    for r in df.itertuples(index=False):
        ca, cb = list(r.conversation_a), list(r.conversation_b)
        rows.append({
            "source": source,
            "raw_id": r.question_id if source == "arena2023" else r.id,
            "model_a": r.model_a, "model_b": r.model_b,
            "winner_raw": r.winner,
            "language_raw": r.language,
            "n_msgs_a": len(ca), "n_msgs_b": len(cb),
            "roles_a": "".join(m["role"][0] for m in ca),
            "roles_b": "".join(m["role"][0] for m in cb),
            "question": _msg_text(ca[0]["content"]) if ca else None,
            "question_b": _msg_text(cb[0]["content"]) if cb else None,
            "answer_a": _msg_text(ca[1]["content"]) if len(ca) > 1 else None,
            "answer_b": _msg_text(cb[1]["content"]) if len(cb) > 1 else None,
            "judge": getattr(r, "judge", None),
        })
    out = pd.DataFrame(rows)
    # raw ids are not always unique (20 repeated question_ids in 2023): suffix repeats
    k = out.groupby("raw_id").cumcount()
    out["pair_id"] = [f"{source}:{x}" + (f"#{i}" if i else "") for x, i in zip(out["raw_id"], k)]
    assert out["pair_id"].is_unique
    return out


def latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    latin = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
    return latin / len(letters)


class CountLog:
    def __init__(self):
        self.rows = []

    def step(self, source, name, before, after, note=""):
        self.rows.append({"source": source, "step": name, "before": len(before),
                          "retained": len(after), "excluded": len(before) - len(after), "rule": note})
        return after

    def frame(self):
        return pd.DataFrame(self.rows)


def apply_filters(df: pd.DataFrame, source: str, log: CountLog) -> pd.DataFrame:
    """Base filters shared by the position and format pools."""
    d = df
    d = log.step(source, "loaded", d, d, "raw rows")
    english_labels = {"English", "en"}
    d = log.step(source, "english_label", d, d[d.language_raw.isin(english_labels)],
                 "dataset language label is English ('English' 2023 / 'en' 2025)")
    single = (d.n_msgs_a == 2) & (d.n_msgs_b == 2) & (d.roles_a == "ua") & (d.roles_b == "ua")
    d = log.step(source, "single_turn", d, d[single], "exactly one user turn + one assistant reply on both sides")
    ok_text = d.question.notna() & d.answer_a.notna() & d.answer_b.notna() & (d.question == d.question_b)
    d = log.step(source, "text_only_aligned", d, d[ok_text], "text-only content; same user prompt on both sides")
    nonempty = (d.question.str.strip().str.len() > 0) & (d.answer_a.str.strip().str.len() > 0) & (d.answer_b.str.strip().str.len() > 0)
    d = log.step(source, "non_missing", d, d[nonempty], "no empty prompt/answer")
    d = log.step(source, "valid_label_non_tie", d, d[d.winner_raw.isin(["model_a", "model_b"])],
                 "winner in {model_a, model_b}; ties / both-bad / invalid labels dropped")
    d = log.step(source, "different_answers", d, d[d.answer_a.str.strip() != d.answer_b.str.strip()],
                 "identical answers carry no preference signal")
    d = d.copy()
    d["latin_a"] = d.answer_a.map(latin_share)
    d["latin_b"] = d.answer_b.map(latin_share)
    d["latin_q"] = d.question.map(latin_share)
    eng = (d.latin_a >= LATIN_SHARE_MIN) & (d.latin_b >= LATIN_SHARE_MIN) & (d.latin_q >= LATIN_SHARE_MIN)
    d = log.step(source, "english_script_check", d, d[eng],
                 f">= {LATIN_SHARE_MIN:.0%} of letters Latin in prompt and both answers (catches mixed-language replies)")
    d["len_a_words"] = d.answer_a.map(md.word_count)
    d["len_b_words"] = d.answer_b.map(md.word_count)
    d = log.step(source, "nonzero_words", d, d[(d.len_a_words > 0) & (d.len_b_words > 0)],
                 "both answers have >= 1 word after marker removal")
    d["len_ratio"] = d[["len_a_words", "len_b_words"]].max(axis=1) / d[["len_a_words", "len_b_words"]].min(axis=1)
    d = log.step(source, "length_ratio_le_2", d, d[d.len_ratio <= MAX_LEN_RATIO],
                 "max/min answer length in words (markdown markers not counted) <= 2")
    fa = d.answer_a.map(md.features)
    fb = d.answer_b.map(md.features)
    d["excl_a"] = [f.excluded for f in fa]
    d["excl_b"] = [f.excluded for f in fb]
    d = log.step(source, "no_tables_or_code", d, d[~d.excl_a & ~d.excl_b],
                 "neither answer has a table, fenced or indented code block")
    fa = d.answer_a.map(md.features)
    fb = d.answer_b.map(md.features)
    d["md_a"] = [f.has_supported_md for f in fa]
    d["md_b"] = [f.has_supported_md for f in fb]
    for side, fs in (("a", fa), ("b", fb)):
        d[f"n_head_{side}"] = [f.n_headings for f in fs]
        d[f"n_bold_{side}"] = [f.n_bold for f in fs]
        d[f"n_bullet_{side}"] = [f.n_bullets for f in fs]
        d[f"other_md_{side}"] = [(f.n_italic + f.n_links + f.n_inline_code) > 0 or f.has_hr for f in fs]
    d["prompt_id"] = d.question.map(prompt_id)
    dup = d.duplicated(subset=["question", "answer_a", "answer_b"], keep="first")
    d = log.step(source, "dedup_exact", d, d[~dup], "exact duplicate (prompt, answer_a, answer_b) records removed")
    d["human_pref"] = d.winner_raw.map({"model_a": "A", "model_b": "B"})
    return d


RECORD_COLS = ["prompt_id", "pair_id", "source", "question", "answer_a", "answer_b",
               "model_a", "model_b", "human_pref", "len_a_words", "len_b_words", "len_ratio",
               "md_a", "md_b", "n_head_a", "n_bold_a", "n_bullet_a", "n_head_b", "n_bold_b",
               "n_bullet_b", "other_md_a", "other_md_b"]


def add_stripped(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for side in ("a", "b"):
        d[f"plain_{side}"] = d[f"answer_{side}"].map(md.strip_markdown)
        v = [md.validate_strip(o, s) for o, s in zip(d[f"answer_{side}"], d[f"plain_{side}"])]
        d[f"strip_ok_{side}"] = [x["strip_ok"] for x in v]
        d[f"strip_fail_{side}"] = [x["strip_fail_reason"] for x in v]
        d[f"strip_diff_{side}"] = [x["first_diff"] for x in v]
    return d


def one_per_prompt(d: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Keep one pair per prompt cluster (seeded) so a few very common prompts
    ('hi', 'test') cannot dominate a pool."""
    return (d.sample(frac=1.0, random_state=seed)
             .drop_duplicates("prompt_id", keep="first"))


def freeze_sample(d: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    if len(d) <= n:
        out = d.copy()
    else:
        out = d.sample(n=n, random_state=seed)
    return out.sort_values("pair_id").reset_index(drop=True)
