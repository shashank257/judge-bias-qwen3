# Proposal update: Colab + hosted APIs (replaces the Kaggle / local-vLLM plan)

These are drop-in replacements for the paragraphs of the proposal that assumed Kaggle T4s. Everything else in
the proposal is unchanged. Where a detail is now fixed, see `docs/evaluation_contract.md` v1.0.

## §3 Judges (replace the first paragraph)
> **Judges.** The original hybrid Qwen3 checkpoints at 0.6B, 1.7B, 4B and 8B (not the later split releases),
> pinned to their Hugging Face revisions, each with `enable_thinking` on and off: 8 configurations. None of the
> four is served by OpenRouter or Together AI (catalogs checked 2026-11-03), so we serve them ourselves with
> vLLM (BF16, one pinned vLLM version) behind an OpenAI-compatible endpoint on rented GPUs. Featherless.ai,
> the only serverless provider with all four checkpoints, is the fallback; it serves 8B in FP8 and the others in
> BF16, which we would report as a confound. Experiments are driven from Google Colab. We render each
> checkpoint's own chat template and send raw prompts to `/v1/completions`, so thinking-off pre-fills an empty
> think block and reasoning is disabled rather than hidden. We use each mode's model-card sampling settings.
> Since these differ by mode, on 200 pairs per size we also run thinking off with the thinking-mode settings.
> Thinking is capped at 2,048 tokens. On truncation we append Qwen's early-stop sentence and `</think>`, ask for
> the verdict with up to 512 more tokens, and report the truncation rate. The judge uses the MT-Bench pairwise
> prompt without its tie option. Verdicts are parsed by a fixed rule (last `[[A]]` or `[[B]]` after the
> reasoning block). An output that is still unparseable after one retry is invalid and counts as a flip in both
> the edited and the noise-floor comparisons. Provider, revision, vLLM version, GPU type, seed, token usage and
> raw outputs are logged per call.

## §3 Data (append)
> The 2023 pool has only 170 pairs with markdown in both answers (167 after one-per-prompt and strip
> validation), so the format pool adds 833 pairs from the 2025 Arena release under the same filters, and results
> are reported by source. A new context-fit filter keeps judge prompts at or below 6,000 tokens (it removed 89
> of 16,141 eligible 2025 pairs). Each pool keeps one pair per prompt cluster.

## §3 Feasibility (replace)
> **Feasibility.** The request manifest has 151,296 judge calls: 16,912 per configuration (4 per pair for the
> 1,000-pair position pool, 10 per pair for the 1,000-pair format pool, 6 per prompt for 500 H3 brackets), plus
> 16,000 for the sampling and prompt checks. That is ≈200M input tokens. On a self-hosted BF16 vLLM endpoint this is about 20 L4 GPU-hours (≈ $16, within Modal's free credit). Featherless per-token pricing would cost ≈ $88. Both estimates come from the week-1 pilot token counts. If budget is short, we drop
> the second-prompt check, then 8B thinking, then H3, and we never go below 1,000 pairs per cell for position
> and format.

## §3 Threats to validity (append)
> Hosted serving cannot guarantee the bit-identical backend of the original plan. We pin the revision, dtype and
> vLLM version, and we log them. Week-1 data shows human voters preferred the markdown answer in 56–57% of
> pairs where only one answer used it, including after controlling for length (odds ratio 1.15–1.21,
> p < 0.01). Human agreement can therefore reward format bias, and we report it alongside that correlation.

## §4 Timeline (replace)
> **Week 1 (done):** data pools frozen, stripping validated and audited, request manifest, API comparison, local
> pilot, evaluation contract v1.0. **Week 2:** endpoint deployment and a hosted 100-pair pilot at 0.6B and 8B
> thinking (cheapest and most expensive cells); confirm the cost model; start the 0.6B and 1.7B runs.
> **Week 7:** preliminary estimates on two sizes. **Weeks 9–10:** full study, H1–H3, reproducibility package.
