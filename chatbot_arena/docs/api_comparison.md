# Hosted API comparison for the original hybrid Qwen3 0.6B / 1.7B / 4B / 8B

Checked 2026-11-03. Live catalog snapshot: `api/catalog_snapshot.json` (notebook 05). Each claim below cites its
source page. **UNVERIFIED** marks anything we could not confirm from a primary source.

## 1. Which providers serve the exact checkpoints?
| Provider | 0.6B | 1.7B | 4B | 8B | What they have instead |
|---|---|---|---|---|---|
| **OpenRouter** | ✗ | ✗ | ✗ | ✗ | Qwen3-14B/32B/30B-A3B, 2507 splits, Qwen3.5+ (453 models listed) |
| **Together AI** (serverless) | ✗ | ✗ | ✗ | ✗ | Qwen3.5-9B, Qwen3.6+/3.7/3.8 |
| DeepInfra · Novita · HF Inference Providers | ✗ | ✗ | ✗ | ✗ | Qwen3-14B/32B and newer; Novita only has a DeepSeek-R1 *distill* on Qwen3-8B |
| **Featherless.ai** | ✓ `Qwen/Qwen3-0.6B` | ✓ `Qwen/Qwen3-1.7B` | ✓ `Qwen/Qwen3-4B` | ✓ `Qwen/Qwen3-8B` | — |
| Self-hosted vLLM (Modal, RunPod, HF Inference Endpoints) | ✓ any pinned revision | ✓ | ✓ | ✓ | — |

**Consequence.** Together and OpenRouter, the two providers the checklist names, cannot run this study
without substituting models, and the proposal forbids silent substitution. The real choice is between
**Featherless**, the only serverless provider that has all four, and **our own vLLM endpoint** on rented GPUs.

## 2. Requirement-by-requirement comparison
| Requirement | Featherless (serverless) | Own vLLM endpoint (Modal / RunPod / HF Endpoints) | OpenRouter / Together |
|---|---|---|---|
| Exact checkpoints | ✓ same HF ids; **revision not disclosed** | ✓ pinned `--revision <sha>` | ✗ not served |
| Thinking on/off, same weights | ✓ `chat_template_kwargs.enable_thinking` (chat) or a locally rendered template sent to `/v1/completions` | ✓ same | n/a (OpenRouter's `reasoning.enabled/effort` would apply) |
| Thinking off really disables reasoning | ✓ with the template we render: the empty `<think></think>` is pre-filled, verified in notebooks 04 and 05 | ✓ same | OpenRouter `reasoning.exclude` only **hides** reasoning, which is not acceptable |
| 2,048-token thinking budget + "close and ask for verdict" | ✓ two `/v1/completions` calls (our harness) | ✓ same, or vLLM's reasoning-budget options | OpenRouter `reasoning.max_tokens` is a provider-mapped budget: no control over the closing text |
| Sampling: T, top-p, top-k, min-p, seed | ✓ all accepted; seed "not reliable across servers" | ✓ all; seed honoured per request | varies by provider; OpenRouter `require_parameters: true` needed |
| Reasoning-token visibility | ✓ text is in `text` (completions); a separate `reasoning_content` field is UNVERIFIED | ✓ raw text | OpenRouter: `message.reasoning` |
| Precision | **0.6B / 1.7B / 4B BF16, 8B FP8** (model pages): differs along the size ladder | ✓ BF16 for all four (`--dtype bfloat16`; T4 lacks bf16 support → use an L4 or A10G) | — |
| Backend fixed across cells | probably vLLM (parameter set), UNVERIFIED; may change without notice | ✓ one pinned vLLM version | ✗ routing may change |
| Provider pinning / no fallbacks | single provider (no routing) | n/a | OpenRouter: `provider.order=[...]`, `allow_fallbacks:false`, `require_parameters:true`, `quantizations:["bf16"]` |
| Context / output limits | 32,768 tokens | set `--max-model-len 12288` (prompt ≤ 6K + 2K think + verdict) | — |
| Rate limits | concurrency budget: Developer plan 100 units, each model costs 1 unit; HTTP 429 when exceeded | GPU-bound (batching) | — |
| Pricing | per token, per 1M in/out: 0.6B $0.04/$0.08 · **1.7B $0.32/$1.60** · 4B $0.40/$0.80 · 8B $0.117/$0.455. Developer plan = $50/month of credits. The $25 "Chat" plan forbids API automation | GPU time: Modal L4 $0.80/h (+$30/month free credit per account), RunPod 24 GB serverless $0.75/h, RunPod L4 pod $0.44–0.59/h, HF Endpoint L4 $0.80/h; scales to zero | — |
| Availability | all four listed "Warm" | we control it | — |

Sources: featherless.ai/pricing, /docs/concurrency-limits, /docs/completions, /docs/chat-template-kwargs, model pages;
openrouter.ai/docs/guides/routing/provider-selection and /best-practices/reasoning-tokens;
docs.together.ai/docs/serverless/models and /dedicated-endpoints; huggingface.co/docs/inference-endpoints/pricing;
modal.com/pricing; runpod.io/pricing; huggingface.co/Qwen/Qwen3-8B (model card).

## 3. OpenRouter settings we would need (documented for completeness)
```json
{"provider": {"order": ["<slug>"], "allow_fallbacks": false, "require_parameters": true,
              "quantizations": ["bf16"], "data_collection": "deny"},
 "reasoning": {"max_tokens": 2048}}
```
Reasoning tokens are billed as output tokens, even when the budget runs out before a visible answer. `exclude: true`
only hides reasoning. Because none of the four checkpoints is on OpenRouter, none of this can be used today.

## 4. Compatibility calls and pilot
See notebook 05 (`api/compatibility_checks.csv`, `api/pilot_summary.csv`). No API key was available in week 1, so
the compatibility checks and the pilot ran through the **same harness** on the **same pinned checkpoints**
locally (Apple-silicon MLX, bf16). That verifies template rendering, thinking on/off, the two-stage
budget logic, verdict parsing, retries, resumability and token accounting. A provider-specific check is still
needed: run `python scripts/run_pilot.py featherless` (or `openai-compatible <url>`) with a key. Per call, the
harness logs the response id, model, finish reason and token usage.

## 5. Pilot results (local, same checkpoints) and projected cost
| Config | Calls | Invalid | Thinking truncated | Mean output tokens | Mean thinking tokens |
|---|---|---|---|---|---|
| 0.6B no-think | 200 | 0% | — | 106 | — |
| 0.6B think | 60 | 0% | 0% | 471 | 388 |
| 1.7B no-think | 200 | 0% | — | 164 | — |
| 1.7B think | 60 | 0% | 3.3% | 633 | 529 |

Preview only (no noise floor, small n): position flip rates were 87% for 0.6B no-think and 33–53% for the others.
0.6B no-think chose the *second* slot 93% of the time. Large position effects look likely.

Full manifest of 151,296 requests: ≈200M input tokens and ≈55M output tokens. The output estimate for 4B and 8B
reuses the 1.7B pilot numbers. Costs (`api/cost_projection.csv`):
- **Featherless per-token: ≈ $88.** At list prices, 1.7B is the most expensive size.
- **Self-hosted vLLM on L4: ≈ 20 GPU-hours ≈ $16.** This uses *assumed* throughput and must be measured in week 2.
  Modal's $30/month free credit per account covers it.

## 6. Decision
**Selected: our own vLLM endpoint serving the four pinned revisions in BF16** (Modal first, RunPod or HF Inference
Endpoints as alternatives). It is the only option that keeps the exact revision, the same precision and the same
backend across all 8 cells, as the proposal requires. It also gives full control of the thinking budget, and it is
the cheapest.
**Fallback: Featherless** (Developer plan, $50 of credit covers the study). It has zero setup, but 8B runs in FP8, so
the precision confound must be reported, and the served revision is undisclosed.
**Rejected: OpenRouter, Together, DeepInfra, Novita, HF Inference Providers.** None serves the exact checkpoints, and
we do not substitute models.
Before the main run, the week-2 hosted pilot (0.6B and 8B thinking, 100 pairs) must confirm throughput and cost.
