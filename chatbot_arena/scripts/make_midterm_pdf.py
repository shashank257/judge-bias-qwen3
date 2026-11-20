"""Build reports/DSC261_G3_midterm_results.pdf from the midterm outputs."""
import os
import sys

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle, KeepTogether)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "reports", "DSC261_G3_midterm_results.pdf")

res = pd.read_csv(os.path.join(ROOT, "reports", "midterm_results.csv"), index_col=0)
pairs = pd.read_parquet(os.path.join(ROOT, "data", "processed", "pilot_pairs.parquet"))

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=17, leading=21, spaceAfter=4)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=9.5, textColor=colors.HexColor("#555555"),
                     alignment=TA_CENTER, spaceAfter=10)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=4,
                    textColor=colors.HexColor("#1f3b63"))
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=9.5, leading=13)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8, leading=10.5, textColor=colors.HexColor("#444444"))
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8, leading=10)
MONO = ParagraphStyle("MONO", parent=BODY, fontName="Courier", fontSize=7.6, leading=9.6)


def bullets(items, style=BODY):
    return [Paragraph(f"&bull;&nbsp;{t}", style) for t in items]


def tbl(data, widths, header_bg="#1f3b63", zebra=True, font=8):
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
          ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
          ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
          ("FONTSIZE", (0, 0), (-1, -1), font),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if zebra:
        for i in range(1, len(data)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f2f5f9")))
    t.setStyle(TableStyle(st))
    return t


def pct(x):
    return f"{x * 100:.1f}"


def ci(m, lo, hi):
    return f"{m * 100:.1f} [{lo * 100:.1f}, {hi * 100:.1f}]"


LABEL = {"0.6B-nothink": "0.6B, thinking off", "0.6B-think": "0.6B, thinking on",
         "1.7B-nothink": "1.7B, thinking off", "1.7B-think": "1.7B, thinking on",
         "4B-q8-nothink": "4B (8-bit), thinking off", "4B-q8-think": "4B (8-bit), thinking on",
         "8B-awq-nothink": "8B (4-bit AWQ), thinking off", "8B-awq-think": "8B (4-bit AWQ), thinking on"}
PREC = {"0.6B": "BF16", "1.7B": "BF16", "4B-q8": "8-bit (MLX)", "8B-awq": "4-bit AWQ"}

story = []
story.append(Paragraph("Position and Format Bias in Qwen3 Judges", H1))
story.append(Paragraph("DSC261 Group 3 &middot; Midterm results on 100 Chatbot Arena pairs &middot; "
                       "interim analysis, not the pre-registered H1&ndash;H3 tests", SUB))

story.append(Paragraph("1. Question", H2))
story.append(Paragraph(
    "An LLM judge reads a user question and two answers and replies <b>[[A]]</b> or <b>[[B]]</b>. "
    "We test whether its verdict changes when only the <b>order</b> of the answers changes (position bias) "
    "or when only the <b>markdown</b> (headings, bold, bullet markers) is removed with the words kept identical "
    "(format bias), for Qwen3 judges of increasing size with thinking switched off and on.", BODY))

story.append(Paragraph("2. Setup", H2))
setup = [["Item", "Value"],
         ["Data", "100 frozen pairs from Chatbot Arena (84 from the 2025 release, 16 from 2023); "
                  "markdown in both answers; English, single-turn, non-tied human vote, length ratio &le; 2"],
         ["Judge prompt", "MT-Bench pair-v2 without the tie option; Qwen3 chat template, thinking via enable_thinking"],
         ["Calls per pair", "10 = (A, B) &isin; {markdown, plain}&sup2; &times; both orders (8) + markdown pair repeated in "
                            "both orders (2, noise floor)"],
         ["Decoding", "Model-card sampling (thinking T 0.6 / top-p 0.95; non-thinking T 0.7 / top-p 0.8; top-k 20); "
                      "2,048-token thinking budget"],
         ["Judges", "Qwen3 0.6B &amp; 1.7B (BF16) and 4B (8-bit) on an Apple M2 Pro (MLX); 8B as Qwen's official 4-bit AWQ build on an RTX 3070 (vLLM); thinking off and on"],
         ["Volume", f"{int(res.pairs.count()) * 1000:,} judge calls, 0 invalid verdicts"]]
story.append(tbl([[Paragraph(c, CELL) for c in r] for r in setup], [1.25 * inch, 5.75 * inch]))

story.append(Paragraph("3. Measures", H2))
story += bullets([
    "<b>Flip rate</b>: share of pairs whose winning answer changes when only the order (position) or only the "
    "markdown (format) changes. <b>Noise floor</b>: share that changes between two runs on identical input.",
    "<b>Excess flip rate</b> = flip rate &minus; noise floor, in points, with a 95% bootstrap CI over pairs "
    "(10,000 resamples, seed 261). The proposal's H1 threshold is a CI lower bound above 2 points.",
    "<b>Picks 1st</b>: share of verdicts for the answer shown first (50% = no position preference). "
    "<b>Picks markdown</b>: when exactly one answer keeps its markdown, share of verdicts for it (50% = none).",
    "<b>Human agreement</b>: accuracy of the original-order verdict against the Arena voter's choice."])

story.append(Spacer(1, 4))
story.append(Paragraph("4. Results", H2))
story.append(Image(os.path.join(ROOT, "reports", "midterm_excess_flip.png"), width=7.0 * inch, height=2.52 * inch))
story.append(Paragraph("Figure 1. Excess flip rate (flip &minus; noise floor) with 95% CIs. Light bars: thinking off; "
                       "dark bars: thinking on. Dashed line: H1 threshold (2 points).", SMALL))
story.append(Spacer(1, 8))

hdr = ["Judge", "Precision", "Position excess flip", "Picks 1st", "Format excess flip", "Picks markdown",
       "Human agr."]
rows = [hdr]
for cfg, r in res.iterrows():
    size = cfg.rsplit("-", 1)[0]
    rows.append([LABEL.get(cfg, cfg), PREC.get(size, "?"),
                 ci(r.pos_excess, r.pos_excess_lo, r.pos_excess_hi), pct(r.slot1_pref) + "%",
                 ci(r.fmt_excess, r.fmt_excess_lo, r.fmt_excess_hi),
                 ci(r.md_pref, r.md_pref_lo, r.md_pref_hi) + "%", pct(r.human_acc) + "%"])
rows = [[Paragraph(str(c), CELL) for c in r] for r in rows]
story.append(tbl(rows, [1.45 * inch, 0.75 * inch, 1.2 * inch, 0.6 * inch, 1.15 * inch, 1.15 * inch, 0.7 * inch]))
story.append(Paragraph("Table 1. All values in percent / points with 95% bootstrap CIs in brackets. "
                       "Controls on the same 100 pairs: random verdicts 58% human agreement, length-only (longer answer "
                       "wins) 57%.", SMALL))

story.append(Paragraph("5. Findings", H2))
story += bullets([
    "<b>Position bias is large and real in every configuration but one.</b> 0.6B with thinking off flips "
    f"{pct(res.loc['0.6B-nothink','pos_excess'])} points beyond noise and almost always picks the <i>second</i> answer "
    f"({pct(res.loc['0.6B-nothink','slot1_pref'])}% first). Larger judges lean toward the <i>first</i> answer.",
    "<b>Thinking reduces position bias at every size</b> (0.6B 73.5 &rarr; 6.5, no longer significant; 1.7B 36.5 &rarr; 16.5; "
    "4B 24.5 &rarr; 16.0; 8B 41.5 &rarr; 15.0) <b>but does not remove it</b> from 1.7B upward.",
    "<b>No clean size trend with thinking off</b> (0.6B 73.5, 1.7B 36.5, 4B 24.5, 8B 41.5 points); "
    "<b>with thinking on, 1.7B, 4B and 8B all sit at 15&ndash;16.5 points</b>, every CI above the 2-point H1 threshold.",
    "<b>Removing markdown rarely flips a verdict</b> (format excess flip within &plusmn;2.2 points, every CI includes 0). "
    "<b>A small preference for the markdown version</b> appears from 1.7B up (52&ndash;58%; CI above 50% for 1.7B and 8B "
    "thinking off and for 4B in both modes), strongest at 4B, with no clean size trend.",
    "<b>Agreement with human voters is weak</b> (51&ndash;66%), close to the length-only control (57%); the best is "
    "8B with thinking on (66%)."])

story.append(Paragraph("6. Example (one pair, 1.7B, thinking off)", H2))
ex = pairs[pairs.pair_id == "arena2025:3f42c387-0a15-4cdb-b975-245b164f1a04"]
q = ex.question.iloc[0] if len(ex) else "Create a sarcastic, meta-style one-liner to promote Deadpool."
story.append(Paragraph(f"<b>User question:</b> {q}  <b>Human vote:</b> answer A (DeepSeek-R1) over B (DeepSeek-V3).", BODY))
exrows = [["Shown first", "Shown second", "Judge", "Winner"],
          ["A (markdown)", "B (markdown)", "[[A]]", "A"], ["B (markdown)", "A (markdown)", "[[A]]", "B"],
          ["A (plain)", "B (markdown)", "[[A]]", "A"], ["B (plain)", "A (plain)", "[[A]]", "B"]]
story.append(tbl(exrows, [1.4 * inch, 1.4 * inch, 0.8 * inch, 0.8 * inch], zebra=False))
story.append(Paragraph("The judge answers [[A]] (\"the first one\") in all 10 calls, so the winner is decided by order, "
                       "not content; removing markdown changes nothing. Four of the ten calls shown.", SMALL))

story.append(Paragraph("7. Caveats", H2))
story += bullets([
    "100 of the planned 1,000 pairs per pool; intervals are wide.",
    "Precision differs by size (BF16 / 8-bit / 4-bit) because of hardware limits; the contract specifies BF16, "
    "so size trends mix model size with quantization.",
    "Run locally (Apple M2 Pro, MLX) for 0.6B&ndash;4B; part of the run used batched generation with batch-level seeds.",
    "Format pairs have markdown in both answers; the plain variant removes headings, bold and bullets only "
    "(italics and numbered lists remain).",
    "8B ran on different hardware and software (RTX 3070, vLLM, 4-bit AWQ) from the smaller sizes (M2 Pro, MLX)."], SMALL)

doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                        topMargin=0.55 * inch, bottomMargin=0.55 * inch,
                        title="DSC261 G3 Midterm Results", author="DSC261 Group 3")
doc.build(story)
print(OUT)
