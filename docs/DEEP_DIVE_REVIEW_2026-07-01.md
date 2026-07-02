# ProjectAI — Deep-Dive Architecture, Performance & Goal Review

**Date:** 2026-07-01
**Method:** Multi-agent review (10 parallel deep-readers over every subsystem and design doc,
followed by adversarial verification of all 40 critical/major findings — each claim independently
re-checked against the code and refuted, corrected, or confirmed). Ground truth: full build clean,
`dotnet test` → **301 passed / 0 failed / 0 skipped** (~4s), all on real libtorch.
**Goal under review (owner's statement):** *"an app that can handle highly complex tasks ranging
all topics where I can plug and play models for the most accuracy that can run on the smallest
computers."*

---

## 1. Verdict

The architecture is genuinely excellent and the documentation is trustworthy — every one of the
boldest CLAUDE.md status claims (KV-cache decode parity, DisposeScope-per-op, gradient
checkpointing, BF16 threading, conformance suite, checkpoint v2) reconciles with committed,
tested code. The `IComputeBackend` seam is enforced by project structure (Core has literally zero
references), not convention, and the oracle → conformance-suite → backend pattern is exactly the
right mechanism for trustworthy plug-and-play hardware.

The gap is between the **codebase's excellent engineering** and the **stated goal's three pillars**,
each of which is currently unmet — and, crucially, **the goal as stated does not match the goal the
repo is planned against.** `BUILD_PLAN.md` §1 targets an RTX 4090 and an Apple M4 — high-end
hardware — and never mentions small machines. "Smallest computers," "most accuracy," and broad
"plug and play" appear nowhere in the repo's own documents. The single highest-leverage action
from this review is to **write the real goal into BUILD_PLAN.md and re-sequence Stage 2/3 around
it** (quantization and converter breadth before Vulkan).

Scorecard against the stated goal:

| Pillar | Today | Verdict |
|---|---|---|
| Plug and play models | Effectively one family loads (SmolLM/SmolLM2). Llama-3.x, Qwen, Mistral, Gemma, Phi all rejected (loudly, by design) | Narrow whitelist; each unlock is small and model-layer-only |
| Most accuracy | No evaluation instrument exists anywhere — no perplexity/bpb/task harness ticket. Accuracy is unmeasurable | Cannot be navigated until an eval harness exists |
| Smallest computers | No quantization (S3-3 unstarted), CPU backend is a deliberate slow oracle, BF16 is Torch-only, Vulkan is a stub. Real floor today: libtorch + CUDA/MPS | The least-developed pillar; staged but unscheduled |
| Complex tasks, all topics | Web search (Tavily) + memory M0 shipped; no doc RAG, no routing, no compaction, no tool loop | Right direction; the multipliers are designs, not code |

None of this requires re-architecting. The seams are correct; every gap is an implementation
behind an existing interface.

---

## 2. What is genuinely strong (verified)

These survived adversarial verification and are worth protecting as the project grows:

- **The dependency rule is structural, not aspirational.** `ProjectAI.Core.csproj` is one line with
  zero references; concrete backends are named in exactly one place above the seam
  (`ProjectAI/Backends.cs`); Models/Training/Formats reference Core only. Verified in the csproj graph.
- **Single-tape-builder autograd.** `Autograd.Record` is the only place GradNodes are created; raw
  `Tensor` view ops are deliberately detached with differentiable variants on the facade —
  eliminating the classic "view silently drops gradient" trap by design. `ReduceGradToShape`
  centralizes broadcast-aware grad folding (what makes GQA's shared-KV gradients correct).
- **The conformance suite is real Liskov enforcement.** One shared 33-case op catalog runs
  identically against the oracle and Torch (including transposed/sliced-offset/stride-0/GQA-broadcast
  views); Torch-specific tests cover what values can't — lifetime regressions under DisposeScope,
  checkpointed-vs-standard grad equality, BF16-tracks-F32.
- **The Torch bridge is economical and correct**: a contiguous base tensor + per-op
  `as_strided(shape, strides, offset)` reconstruction reuses libtorch's own striding — the whole
  backend is 277 lines for the full seam.
- **Checkpoints are self-describing and tokenizer-agnostic** (config + tokenizer + ComputeDType in
  metadata; bit-identical logits after reload, asserted with exact float equality).
- **Loaders and converter fail loudly, never silently wrong** — full-coverage/bounds/overflow checks
  in safetensors/GGUF; every unsupported architecture rejected with a named error.
- **Test quality in the numeric core is exemplary**: independent double-precision oracles,
  finite-difference gradient checks on every op with curvature-aware tolerances, seeded determinism
  throughout, adversarial malformed-input tests on every parser.
- **Docs err toward understatement.** Test count claim was 274; reality is 301. Committed features
  (WS /chat, /tokenize, dataset packing) exceed what CLAUDE.md records.
- **The memory design docs are exceptional artifacts** — explicit rejected alternatives, honest
  costing, a trust boundary that is actually enforced at the read path and tested (untrusted content
  never auto-injected), atomic writes, traversal-proof per-user partitioning.

---

## 3. Confirmed correctness traps (fix before they fire)

Ordered by (cost of fix) vs (cost of the day it fires). All are small changes.

### 3.1 Latent sampler misalignment on padded-vocab models — CONFIRMED
`Inference.GenerateText` uses `tokenizer.VocabSize` as the logits row width, but the logits' true
last axis is `config.VocabSize` (`Inference.cs:27, 63-68` vs `Modules.cs:260`). HF models routinely
pad `vocab_size` past the tokenizer's max id. The moment converter breadth widens, sampling reads a
garbage window with no error. **Fix:** use `config.VocabSize` in `SampleLast`; assert
`tokenizer.VocabSize <= config.VocabSize` at convert/load.

### 3.2 `Autograd.Checkpoint` silently drops parameter gradients if the segment input stops requiring grad — CONFIRMED
`Record` gates on the *activation* input's `RequiresGrad`; parameters are hidden inputs
(`Autograd.cs:133, 319`). Freeze the embedding for fine-tuning and every checkpointed block trains
nothing — silently (AdamW skips null grads; with the tied LM head, only FinalNorm would train while
loss still creeps down). Latent today; fires the day partial-freeze fine-tuning (LoRA prep!) lands.
**Fix:** gate on `input.RequiresGrad || parameters.Any(p => p.RequiresGrad)`.

### 3.3 CPU backend silently ignores requested DType — a BF16 checkpoint "loads" as F32 — CONFIRMED
`CpuComputeBackend.Allocate/FromHost` stamp whatever dtype is requested onto an F32 buffer
(`CpuComputeBackend.cs:41-49`); `Checkpointing.LoadModel` rebuilds at `meta.ComputeDType` with no
backend guard. A `convert --bf16` checkpoint served on `cpu` reports BF16 while occupying F32 memory
— a Liskov violation the F32-only conformance suite structurally cannot see. **Fix:** throw
`NotSupportedException` in the CPU backend for non-F32 dtypes (or auto-fallback with a logged
downgrade at load), and add at least one non-F32 conformance case.

### 3.4 Training/inference mutual exclusion has holes — CONFIRMED
`/generate` is gated on `training.IsTraining`, but WS `/chat` never checks it, and
`TrainingService.Run` never takes `InferenceLock` (`Server.cs:143` vs `Server.cs:409-536`,
`TrainingService.cs:43`). A chat turn and a training job can share the *same cached
`TorchComputeBackend` instance* — whose thread-safety the code explicitly disclaims — and contend
for VRAM (the defining failure mode of small machines). The `/generate` gate is also TOCTOU.
**Fix:** one `GpuGate` honored by chat + train + generate; check it under the lock.

### 3.5 Memory recall injects irrelevant context when nothing matches — CONFIRMED
On a no-keyword-match query, `FileMemoryStore.Search` falls back to *all* nodes and the additive
scorer guarantees ~1.2 base score, so up to 3 unrelated memory bodies (~400 tokens) are prepended
(`FileMemoryStore.cs:98-101, 314-331`; no threshold in `RenderRecall`). On 360M–1.7B models,
irrelevant context measurably hurts answers — the opposite of the feature's purpose. **Fix:**
require ≥1 lexical hit (drop the fallback) or add a minimum-score floor.

### 3.6 No auth + CORS `*` + unchecked WS origin — CONFIRMED (design decision to revisit)
Any web page in the user's browser can `POST /train` (start a GPU job), hit `/generate`, and attach
any user's memory store (`Server.cs:102-112, 225, 549-551`). Localhost binding does not mitigate
browser-mediated cross-origin calls, and the planned web client + secrets/memory endpoints
(CLIENT_DESIGN.md §3.3) raise the stakes: a malicious origin could overwrite the Tavily key and read
memory content. **Fix:** a generated bearer token the clients send (Godot + web harness both control
their headers), or at minimum an Origin allowlist; do this *before* the secrets/memory endpoints land.

Also confirmed, lower urgency: memory supersession ignores trust/confidence (a chat-trust draft can
tombstone a curated long-tier fact — M2 work, but the doc's own top risk;
`FileMemoryStore.cs:232-260`), and the WS receive loop has no message-size cap while every HTTP
endpoint is carefully capped.

---

## 4. Performance & computational-ability improvements

The section you asked for. Split into (a) inference performance on current hardware, (b) memory/
allocation efficiency, (c) computational *capability* — what lets the runtime do more per byte and
per machine. Ordered by leverage within each group.

### 4.1 Inference performance — quick wins (days each, mostly one-liners)

1. **Slice to the last position before the LM head when generating.** `LlamaModel.Forward` always
   computes full-vocab logits for *every* position, and `Inference.SampleLast` downloads the entire
   `[1, seq, vocab]` tensor to read one row (`Modules.cs:286`, `Inference.cs:63-68`). For a
   2048-token SmolLM2 prompt that is **~402 MB of device→host transfer (plus matching device + host
   allocations) to use 196 KB**, and 6–16% of prefill FLOPs wasted in the head matmul. Fix with the
   existing `Ag.Slice` before the head (generation path only) or at minimum download only the last
   row. This is the single best latency-per-line-changed in the codebase.

2. **Scope the decode loop.** The trainer scopes every micro-batch, but `Inference.GenerateText`
   runs unscoped and `KvCache.ConcatSequence` drops the superseded cache tensor without `Release`
   (`Inference.cs:40-53`, `Modules.cs:361-370`). On Torch, every generated token strands
   `2 × layers` superseded K/V tensors plus all per-op intermediates — **O(n²) transient device
   garbage per generation**, reclaimed only by GC finalizers. This is the *unacknowledged* sibling
   of the S2-3 training fix (the docs' "residual" note covered training only) and a likely cause of
   invisible OOMs on the 8 GB GPU during long generations. Fix: `BeginScope`/`KeepAlive` per decode
   step + `Release` the superseded cache tensor in `ConcatSequence`.

3. **Skip-init on model load.** `Module.Param` always runs the Gaussian init, then `Converter.Load`
   / `Checkpointing.LoadModel` overwrite every weight (`Module.cs:59-67`, `Init.cs:35-40`). Loading
   SmolLM2-1.7B pays **~1.7 billion sequential Box–Muller draws** for nothing, linear in model size
   — a direct tax on exactly the plug-and-play load path. Fix: a `SkipInit`/`Init.Deferred` flag on
   `ParameterContext` used by the two load paths.

4. **Cache the causal mask.** `Attention.CausalMask` rebuilds an `s × keyLen` host array and
   re-uploads it **per layer per forward** (`Modules.cs:190, 211-218`) — at s=1024 that's ~4 MB ×
   layers of identical host→device traffic every training step. Cache per (s, keyLen) on the model,
   or generate device-side.

5. **Share one RoPE table across layers.** Every `Attention` builds its own `RotaryEmbedding`
   (`Modules.cs:138`) — N identical `[maxSeq, headDim]` cos/sin tables. One shared instance; also cap
   converted models' `MaxSequenceLength` (Converter maps `max_position_embeddings` uncapped, so a
   131k-context model would build a giant table).

6. **Incremental streaming decode.** `ChatSession` calls `_tok.Decode(generated)` on the *whole
   reply so far for every token* — O(n²) decode work over a long answer. Decode only the new suffix
   with the existing UTF-8 holdback machinery.

7. **Stop uploading token ids as F32 and round-tripping them.** Ids go up as float tensors, then
   `Embedding.ToIds`/`Loss.ToIds` download them again inside the same forward (`Training.cs:140`,
   `Embedding.cs:38-45`, `Loss.cs:26-33`) — a hidden device→host sync every step. Short-term: carry
   host-side ids alongside the tensor. Long-term: an integer dtype through the seam (see 4.3.4).

8. **Remove per-step `.item<float>()` syncs in the Torch loss path** (`TorchComputeBackend.cs:175,
   186`) — each forces a GPU sync; return device scalars or batch the reads.

9. **Per-backend inference locks.** The single static `InferenceLock` serializes a slow CPU
   generation against an unrelated `torch:cuda` chat turn (`Server.cs`). `ComputeRegistry` already
   isolates instances per backend id — key the lock the same way.

10. **Paged KV cache (S3-4, keep it scheduled).** `ConcatSequence` re-concatenates the full history
    every token — O(n) copy per token, O(n²) per generation. Preallocate pages/capacity and write in
    place; this compounds with #2.

11. **Add a repetition penalty sampler.** Small models loop badly; this is the cheapest quality knob
    missing from the pipeline (`Sampling.cs`). Also lets `TopKTopP` avoid full-vocab work when k ≪ V.

### 4.2 Memory-frugality (the "smallest computers" enablers already half-built)

- The S2-3/S3-2 machinery (per-step scoping, deterministic `Release`, gradient checkpointing) is
  excellent and **training-only today**. Extending it to inference (#2 above) completes the story.
- The memory-lifetime contract (`KeepAlive` one-level promotion, `Release` double-free precondition)
  is correct but backend-dependent: mistakes are silent on CPU and use-after-free on Torch. The three
  Torch lifetime-regression tests cover the known choreographies — keep adding one per new scope
  user, and consider a debug-mode scope-tracking CPU backend so lifetime bugs fail on any machine.
- `FileMemoryStore` rewrites *every* derived posting file on every mutation but never reads them
  back (`FileMemoryStore.cs:368-386`) — O(all keys) file churn per Encode for zero runtime benefit.
  Either read them at startup or stop writing them.

### 4.3 Computational ability — what unlocks more capability per machine

1. **Quantization (S3-3) is the single biggest lever — do it before Vulkan.** Q4 turns the 6.85 GB
   F32 SmolLM2-1.7B checkpoint into ~1 GB and puts 7–8B-class models (the accuracy tier) inside an
   8 GB machine. It is one summary line in the plan today. **Design note discovered by this review:**
   the seam's host transfer speaks only `Span<float>` (`IComputeBackend.cs:16-17`) and the Torch
   bridge's core invariant is element-strided views — block-quantized tensors (Q4_K/Q8_0) will not
   fit through either. Extend the seam (raw-byte transfer + block-aware storage or dequant-on-load)
   **before** writing any Vulkan kernel, or the seam gets redesigned twice and the third backend
   inherits the churn.

2. **Converter breadth — each unlock is small and model-layer-only:**
   - *untied `lm_head`* → one optional `Linear` replacing the hard-coded tied matmul
     (`Modules.cs:286`). Unlocks Llama-2/3-8B-class, Mistral, TinyLlama.
   - *`rope_scaling` (llama3 type)* → a deterministic per-frequency rescale in
     `RotaryEmbedding.BuildTables`. Unlocks **Llama-3.2-1B/3B — the canonical accuracy-per-GB
     targets, which are weight-tied and would otherwise convert today.** Requires the cl100k
     pre-tokenizer too (next bullet).
   - *`pre_tokenizer` parsing* → read the split regex from tokenizer.json instead of assuming GPT-2
     (`HfTokenizer.cs:93`, `Converter.cs:63`). Without it, a converted Llama-3/Qwen silently
     tokenizes wrong — a *silent accuracy* bug, inconsistent with the fail-loud weights policy.
   - *attention bias* → bias on the QKV Linears. Unlocks **Qwen2.5, the strongest small-model
     family**.
   - *QK-norm* → an RmsNorm application. Unlocks Qwen3.
   None require new backend ops. Suggested order: untied head + rope_scaling + pre_tokenizer
   (Llama-3.2), then attention bias (Qwen2.5).

3. **An evaluation harness — "most accuracy" is unmeasurable today.** No perplexity, bpb, or task
   eval exists anywhere in code or plan; the core decision the goal implies ("on this 8 GB machine,
   is Q4-7B better than BF16-3B?") cannot be answered. Cheap to build: `projectai eval --data
   <file>` reusing the existing loss path for held-out perplexity/bpb; CLIENT_DESIGN.md's benchmark
   spec (bpb + tok/s per backend) and MEMORY_LINEAGE_DESIGN's L5 eval gate both already *design*
   this instrument — build it once, share it. **Ship it in the same stage as quantization**, so Q4
   degradation is measured, not guessed.

4. **A fast CPU tier — decide, don't drift.** The managed CPU backend is a deliberate oracle
   (single-threaded scalar GEMM; ~1 tok/s for a 360M F32 model). The plan's answer for CPU-only
   machines is currently "torch:cpu with a several-hundred-MB native bundle." Either bless that
   as the floor explicitly, or add a `Backends.CpuFast` project (Parallel.For + `Vector<T>`/AVX
   tiled GEMM + eventually the Q4/Q8 dequant kernels from S3-3) validated by the existing
   conformance suite — the seam makes this purely additive. A hand-written runtime whose only fast
   path is libtorch undercuts the project's own thesis; this is also where quantized CPU kernels
   (the true "smallest computer" story) would live.

5. **LoRA (S3-5) + the lineage/consolidation loop is the genuine moat.** Per-user LoRA deltas +
   eval-gated consolidation of file memories into weights is the one capability the free baseline
   (llama.cpp/ollama) structurally cannot offer, because ProjectAI owns its trainer. Sequence it
   after quantized serving of a stronger base (Qwen2.5/Llama-3.2-class) so there is a base worth
   consolidating into. Note S3-1 mixed-precision training is a prerequisite for LoRA on multi-B
   models — the current "next" queue is right about that.

6. **App-layer capability multipliers** (what makes small models handle complex tasks):
   - *Context compaction* — today a full chat context is a dead session (`ChatSession.cs:76-77`);
     M3's compaction design exists, unbuilt. This is the highest-impact chat-quality item.
   - *Document RAG* — memory recall is lexical-only; there is no "chat with your files" path at all.
     Even lexical file-RAG over the memory store's index machinery would move the needle.
   - *Model routing* — the server already supports per-request model+backend; nothing routes easy
     turns to a small model and hard turns to a big one. This architecture is unusually well
     positioned for it (it's the classic accuracy-per-watt play).
   - *Chat template metadata in the checkpoint* — the hardcoded ChatML probe (`ChatSession.cs:40-42,
     118-125`) means every future non-ChatML instruct model silently degrades to base-model
     behavior. `tokenizer_config.json` carries `chat_template` at convert time; store a template id
     in checkpoint v2 metadata. Do this *with* converter breadth, not after.

---

## 5. Architecture & process recommendations

1. **Re-sequence: S3-3 quantization (+ GGUF→ModelConfig convert) and the eval harness before the
   S2-4/5 Vulkan path.** Vulkan is the most expensive, lowest-goal-leverage item queued: it serves
   machines that already have a Torch path or no GPU at all, and hand-rolled SPIR-V GEMM parity is a
   multi-month effort. It also gets *cheaper* after quantization (the seam extension lands first,
   and Q8 kernels are what a Vulkan backend should implement anyway). Keep Vulkan as the craft
   track; don't put it on the goal's critical path. Stage 4 (image/mesh) is orthogonal to the
   stated goal entirely — sequence it last, consciously.

2. **Write the goal down and make it falsifiable.** BUILD_PLAN.md names two high-end machines and
   no minimum target. Add one line — e.g. "a 4-year-old 8 GB laptop with no discrete GPU must run a
   useful instruct model at ≥N tok/s" — and the Vulkan/quantization/fast-CPU priority argument
   settles itself.

3. **Restore roadmap governance.** CLAUDE.md's "BUILD_PLAN.md is the single source of the roadmap"
   is now false: the server, clients, Research, Trainer, Memory (M0–M4), and Lineage (L1–L9)
   workstreams live outside it, §12's "next actions" still points at Stage 0, and the promised
   Stage 2–4 expansion never happened. Either fold the M*/L*/client milestones into BUILD_PLAN.md
   or amend CLAUDE.md to name the four roadmap documents. Also restructure CLAUDE.md's ~150-line
   Status prose into a per-ticket table — it already lags the code (274 vs 301 tests; WS /chat,
   /tokenize, dataset packing, Research absent).

4. **Unify the two generation paths and the three training entry points.** `/generate` and `/chat`
   differ in templating, stop handling, and memory framing (raw prepend vs system turn); prompt
   assembly (template + memory + research framing) should be one component. `ModelTrainer.TrainOnText`
   exists to be the shared training core but only the HTTP path uses it — the CLI and Trainer exe
   hand-roll drifted variants (different warmup formulas; the CUDA GC mitigation exists only in the
   exe, so the identical job can OOM via `POST /train` but succeed via the exe).

5. **Close the test-coverage inversion.** The numeric core is superbly tested; the layers users
   actually touch have zero tests: Server routing/validation (the path-traversal guard, body caps,
   the train gate — all hardening claims, none regression-tested), `Inference.GenerateText` (the
   shared decode core: stop reasons, truncation, continuation split), ChatSession, registries, CLI
   parsing. Also: add `Copy` + a non-F32 case + `Softmax[axis=1]` to the conformance catalog, a
   real-vocab HfTokenizer parity test (the Ġ byte-map region has no coverage; skip-if-absent against
   `hf-models/` like the Torch tests do), and **stand up the CI leg with libtorch-cpu that
   BUILD_PLAN §11/§12 already calls for** — today conformance enforcement exists only on the one
   dev machine.

6. **Pick the client architecture's security posture before the web client grows.** The permissive
   CORS exists *for* the web-client direction, and the planned secrets/memory endpoints would be
   readable/writable cross-origin (§3.6 above). A bearer token is a day of work now and a breaking
   change later. Watch the dual-client drift too — the prototype already disagrees with the wire
   protocol in two places the Godot client gets right (`/train/status` nesting; `sources` vs `items`).

7. **Memory subsystem, before M1 wires ENCODE live:** add the recall threshold (§3.5), make
   supersession respect trust/status (a low-trust draft must not tombstone a curated fact), and
   revisit the model-name default store id (memory silently fragments per model the moment the
   client exposes the toggle — "the user is Ben" is user state, not model state; the design doc
   itself argues this).

---

## 6. Suggested order of work

| # | Item | Size | Serves |
|---|---|---|---|
| 1 | Correctness traps: vocab-width fix, Checkpoint grad gate, CPU-BF16 guard, train/chat GPU gate | days | accuracy, stability |
| 2 | Inference perf pack: last-position logits, decode scoping + KV Release, skip-init, mask/RoPE caching, incremental stream decode | days | speed on all hardware |
| 3 | Eval harness (`projectai eval`, bpb/perplexity; benchmark endpoint per CLIENT_DESIGN) | ~1 wk | "most accuracy" measurable |
| 4 | Converter breadth: untied head → rope_scaling + cl100k pre-tokenizer → attention bias; chat-template metadata in checkpoint v2 | ~1-2 wks | plug-and-play (Llama-3.2, Qwen2.5) |
| 5 | S3-3 quantization incl. seam extension (raw-byte transfer) + GGUF→ModelConfig convert | the big one | smallest computers |
| 6 | Fast-CPU decision: bless torch:cpu floor or build `Backends.CpuFast` (SIMD+parallel, later Q4/Q8 kernels) | decision + wks | smallest computers |
| 7 | Memory M1–M3 with the three behavioral fixes; context compaction | per design docs | complex tasks |
| 8 | S3-1 mixed-precision training → S3-5 LoRA → lineage L1–L3 | staged | the moat |
| 9 | CI leg (libtorch-cpu), app-layer tests, auth token, roadmap-doc consolidation | ongoing | trust & velocity |
| 10 | Vulkan S2-4/5 (post-quantization, kernels target Q8/Q4 from day 1); Stage 4 last | craft track | breadth |

---

## 7. Verified finding index (by severity)

**Critical (all verified):** converter whitelist ≈ SmolLM2-only (documented deferral, but collides
with Stage 3's own exit criteria) · no eval instrument anywhere · no credible small-computer path
yet (quantization unstarted, oracle-only CPU, Vulkan stub, libtorch floor) · Vulkan queued ahead of
quantization (partially mitigated: status already sequences S3-3 after S3-1).

**Major, confirmed:** decode path unscoped (O(n²) device garbage — unacknowledged in docs) ·
full-logits prefill + full-tensor download · eager-init tax on load · padded-vocab sampler trap ·
Checkpoint frozen-input grad drop · CPU backend ignores DType · train/chat mutual-exclusion holes ·
three drifted training entry points · no auth + CORS `*` + WS origin unchecked · memory recall
no-match fallback injects noise · supersession ignores trust · app layer has zero tests ·
`Copy` absent from conformance catalog · `GenerateText` untested · HfTokenizer tested on ASCII toy
vocab only · CLAUDE.md project map drift · prototype↔server wire drift (2 live mismatches) ·
plug-and-play has no client path (import/convert is CLI-only; memory dark in Godot client) ·
lineage moat is design-only.

**Notable minors:** WS message size uncapped · global inference lock across backends · id
F32-round-trip per forward · causal-mask/RoPE rebuild churn · `uses` counter never persisted ·
chars/4 token budgets · derived memory indexes written-never-read · repetition penalty missing ·
per-step `.item()` syncs · trainer resume replays LR schedule from step 1 · checkpoint F32-on-disk
2× size · design tokens hand-mirrored in three places.

Full per-finding evidence (file:line) lives in the review transcript; each item above names its
primary locations in the sections that discuss it.
