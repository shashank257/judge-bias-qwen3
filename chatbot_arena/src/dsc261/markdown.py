"""Markdown detection, deterministic stripping, and word-sequence validation.

Supported (manipulated) features, per the proposal: ATX headings, bold
(``**x**`` only; ``__x__`` is left alone because it collides with Python
dunder names), and bullet markers (-, *, +). Tables and code blocks are *excluded* (stripping them would
change content). Everything else (numbered lists, italics, links, ...) is left
untouched in both variants and only recorded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

STRIP_VERSION = "strip-v1.0"

# ---- supported features -----------------------------------------------------
HEADING_RE = re.compile(r"^[ ]{0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$", re.M)
BULLET_RE = re.compile(r"^([ \t]*)([-*+])[ \t]+(?=\S)", re.M)
BOLD_STAR_RE = re.compile(r"\*\*(?=[^\s*])([^\n]+?)(?<=[^\s*])\*\*")  # line-local

# ---- excluded features ------------------------------------------------------
FENCE_RE = re.compile(r"^[ ]{0,3}(```|~~~)", re.M)
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.M)
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", re.M)
HR_RE = re.compile(r"^[ ]{0,3}([-*_])([ \t]*\1){2,}[ \t]*$", re.M)

# ---- other markdown (recorded, not manipulated) -----------------------------
ITALIC_RE = re.compile(r"(?<![\*\w])\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*(?![\*\w])")
LINK_RE = re.compile(r"\[[^\]\n]+\]\([^)\s]+\)")
NUMBERED_RE = re.compile(r"^[ \t]*\d{1,3}[.)][ \t]+\S", re.M)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def _indented_code_block(text: str) -> bool:
    """CommonMark indented code: >=4-space/tab line after a blank line that is
    not a list item (list continuations are indented too)."""
    lines = text.split("\n")
    for i in range(1, len(lines)):
        ln = lines[i]
        if (ln.startswith("    ") or ln.startswith("\t")) and ln.strip():
            if lines[i - 1].strip() == "":
                s = ln.strip()
                # list items / continuation of list paragraphs are allowed
                if re.match(r"^([-*+]|\d{1,3}[.)])\s", s):
                    continue
                prev_nonblank = next((lines[j] for j in range(i - 1, -1, -1) if lines[j].strip()), "")
                if re.match(r"^\s*([-*+]|\d{1,3}[.)])\s", prev_nonblank):
                    continue
                return True
    return False


@dataclass
class MdFeatures:
    n_headings: int
    n_bold: int
    n_bullets: int
    has_code_block: bool
    has_table: bool
    n_italic: int
    n_links: int
    n_numbered: int
    n_inline_code: int
    has_hr: bool

    @property
    def has_supported_md(self) -> bool:
        return (self.n_headings + self.n_bold + self.n_bullets) > 0

    @property
    def excluded(self) -> bool:
        return self.has_code_block or self.has_table

    def as_dict(self) -> dict:
        d = asdict(self)
        d["has_supported_md"] = self.has_supported_md
        d["excluded_table_or_code"] = self.excluded
        return d


def _without_hr(text: str) -> str:
    return HR_RE.sub("", text)


def features(text: str) -> MdFeatures:
    t = text or ""
    body = _without_hr(t)  # '***' / '---' rules must not count as bullets/bold
    return MdFeatures(
        n_headings=len(HEADING_RE.findall(body)),
        n_bold=len(BOLD_STAR_RE.findall(body)),
        n_bullets=len(BULLET_RE.findall(body)),
        has_code_block=bool(FENCE_RE.search(t)) or _indented_code_block(t),
        has_table=bool(TABLE_SEP_RE.search(t)) or len(TABLE_ROW_RE.findall(t)) >= 2,
        n_italic=len(ITALIC_RE.findall(body)),
        n_links=len(LINK_RE.findall(body)),
        n_numbered=len(NUMBERED_RE.findall(body)),
        n_inline_code=len(INLINE_CODE_RE.findall(body)),
        has_hr=bool(HR_RE.search(t)),
    )


def strip_markdown(text: str) -> str:
    """Remove headings / bold / bullet markers. Line breaks are kept so list
    items stay on their own lines; no word is added, dropped or reordered."""
    lines = []
    for line in text.split("\n"):
        if HR_RE.match(line):  # horizontal rules are not manipulated
            lines.append(line)
            continue
        line = HEADING_RE.sub(lambda m: m.group(2), line)
        # bullets: drop marker and the indentation that only encoded nesting
        line = BULLET_RE.sub("", line)
        line = BOLD_STAR_RE.sub(lambda m: m.group(1), line)
        lines.append(line)
    return "\n".join(lines)


# ---- validation (independent of the stripping regexes) ----------------------
_MARKER_TOKENS = {"#", "##", "###", "####", "#####", "######", "-", "*", "+"}


def canonical_words(text: str) -> list[str]:
    """Token-level view used for the word-sequence check. Independent of the
    regex stripper: whitespace tokens, drop standalone heading/bullet marker
    tokens, then remove every '**' except a lone '**' between digits (arithmetic like 2**3 is kept,
    so a stripper that deletes it fails; unbalanced pairs also fail)."""
    words = []
    for line in (text or "").split("\n"):
        toks = line.split()
        # a marker only counts as markup when it opens the line
        if toks and toks[0] in _MARKER_TOKENS:
            is_heading = toks[0].startswith("#")
            toks = toks[1:]
            if is_heading:
                while toks and set(toks[-1]) == {"#"}:
                    toks = toks[:-1]
        for tok in toks:
            if set(tok) != {"*"}:  # leave '***' rules alone
                # a single '**' between digits is arithmetic (2**3), not markup: keep it, so a
                # stripper that deletes it fails the check. Paired '**' (bold, even 9**.1**1) is removed.
                if not (tok.count("**") == 1 and re.search(r"\d\*\*\d", tok)):
                    tok = tok.replace("**", "")
            if tok:
                words.append(tok)
    return words


def plain_words(text: str) -> list[str]:
    return (text or "").split()


def word_count(text: str) -> int:
    """Answer length in words, invariant to supported markdown markers."""
    return len(canonical_words(text))


def validate_strip(original: str, stripped: str) -> dict:
    """Pass iff (1) the stripped text's words equal the original's words with
    only supported markers removed, and (2) no supported markdown remains.
    Also reports the first divergence (added/removed/reordered words)."""
    a = canonical_words(original)
    b = plain_words(stripped)
    same = a == b
    first_diff = None
    if not same:
        import difflib
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
        if ops:
            tag, i1, i2, j1, j2 = ops[0]
            first_diff = f"{tag}: orig={a[i1:i2][:8]} stripped={b[j1:j2][:8]}"
    residual = features(stripped)
    residual_md = residual.has_supported_md
    # residual leftover '**' indicates unbalanced bold
    leftover = "**" in _without_hr(stripped)
    ok = same and not residual_md and not leftover
    reason = None
    if not same:
        reason = "word_sequence_changed"
    elif residual_md:
        reason = "residual_supported_markdown"
    elif leftover:
        reason = "unbalanced_bold_marker"
    return {"strip_ok": ok, "strip_fail_reason": reason, "first_diff": first_diff,
            "n_words_orig": len(a), "n_words_stripped": len(b)}
