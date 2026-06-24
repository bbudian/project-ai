# CLAUDE.md — ProjectAI

Orientation for anyone (human or AI) writing code here. Keep it short and true.
The staged roadmap and per-ticket detail live in **`docs/BUILD_PLAN.md`** — link to it, don't
restate it here.

## What this is
A local, self-hosted, general-purpose AI runtime written **by hand** in C# / .NET 10. One numeric
core powers three swappable modules: an LLM (first), image generation, and 3D-mesh generation.
We write the ML logic ourselves; libraries are allowed only for math, SIMD, IO, and device bindings
— never for model, autograd, or training logic.

## The one rule that keeps this simple
**Dependencies point inward to `ProjectAI.Core`. Nothing above the `IComputeBackend` seam may
reference a concrete backend.**

```
  Tokenizers   Models   Training   Formats   CLI
       \          \        |         /        /
        \          \       |        /        /        (all depend only on Core's
         ─────────────►  ProjectAI.Core  ◄────────     abstractions, never on a
                         (Tensor, Module,             concrete backend)
                          autograd, AdamW,
                          IComputeBackend) ◄── THE SEAM
                                ▲
              implemented by ───┼───────────────┐
              Backends.Cpu   Backends.Torch   Backends.Vulkan
              (oracle)       (CUDA / MPS)     (SPIR-V; MoltenVK→Metal)
```

Respect that single rule and models/training/tokenizers never know or care whether they run on
CPU, CUDA, or Vulkan. Backend selection happens **once**, in the CLI composition root.

## Project map
One concern per project. Every arrow in the table is a *compile-time* `ProjectReference`.

| Project | Single responsibility | References |
|---|---|---|
| `ProjectAI.Core` | Tensor/Shape/DType/Device, `Module`, autograd (`GradNode`), `AdamW`, **`IComputeBackend`** | none |
| `ProjectAI.Backends.Cpu` | Managed reference backend (`TensorPrimitives`); the numerical oracle | Core |
| `ProjectAI.Backends.Torch` | TorchSharp/libtorch backend (CUDA/MPS) | Core |
| `ProjectAI.Backends.Vulkan` | Hand-written Vulkan compute (Silk.NET) | Core |
| `ProjectAI.Tokenizers` | Byte-level BPE | none |
| `ProjectAI.Formats` | safetensors / GGUF loaders, `StateDict` | Core |
| `ProjectAI.Models` | Llama-style transformer (RoPE/GQA/RMSNorm/SwiGLU), KV cache, samplers | Core |
| `ProjectAI.Training` | Training loop, datasets, checkpointing | Core, Models, Formats, Tokenizers |
| `ProjectAI` (CLI) | Composition root: picks a backend, wires `generate`/`train`/`convert` | all of the above |
| `ProjectAI.Tests` | xUnit v3: contract tests (live); gradient-check (S0-6) and backend-conformance (S2-1) suites land with those stages | Core, Backends.Cpu, Models |

## SOLID, concretely
- **S — Single responsibility:** one concern per project and per class. A file that mixes
  "what to compute" with "how to compute it on hardware" is a smell — push the *how* below the seam.
- **O / D — Open-closed & dependency inversion:** code against `IComputeBackend`, `ITokenizer`,
  `IWeightLoader`, `IOptimizer`, `ISampler` — never a concrete type. Add a backend or a loader
  without editing a single call site.
- **L — Liskov:** every `IComputeBackend` must be substitutable. The **conformance suite** (each
  backend must match `Backends.Cpu` within tolerance) is the planned enforcement mechanism — it lands
  in Stage 2 (ticket S2-1); a backend that can't pass it isn't done. Today the only enforcement is that
  every backend implements the interface so the solution compiles.
- **I — Interface segregation:** keep `IComputeBackend` cohesive. If a method only one backend needs
  starts creeping in, split the interface instead of widening it.

## DRY, concretely
- **`Directory.Build.props`** is the single home for shared MSBuild config (`net10.0`, nullable,
  implicit usings, doc-gen). **Don't repeat these in individual `.csproj` files.**
- **`Backends.Cpu` is the single source of numerical truth.** Don't re-derive reference math
  elsewhere — every other backend is validated against it.
- **`docs/BUILD_PLAN.md`** is the single source of the roadmap. Reference ticket IDs (e.g. `S1-7`);
  don't copy the plan into code comments beyond the short `NotImplementedException` pointers.

## Where does my code go?
- **New tensor/compute op** → add to `IComputeBackend`, implement in `Backends.Cpu` first (the
  oracle), then the others. Ship a reference comparison + gradient check.
- **New backend** → new `ProjectAI.Backends.*` project implementing `IComputeBackend`; wire it only
  in the CLI composition root.
- **New model layer** → `ProjectAI.Models`, deriving `Module`, built from `IComputeBackend` ops.
- **New checkpoint format** → `ProjectAI.Formats`, implementing `IWeightLoader`.
- **Shared build setting** → `Directory.Build.props` (never copy-pasted per project).

## Conventions
- Data-oriented inner loops: `Span<T>`, SIMD, `TensorPrimitives`. Allocate off the hot path.
- Nullable enabled; analyzers on. Prefer immutable `record` configs (`ModelConfig`, `TrainingConfig`).
- Determinism: fixed seeds, centralized tolerances, a reference comparison for every numeric op.

## Build & test
```
dotnet build
dotnet test                         # contracts now; gradient-check (S0-6) + conformance (S2-1) later
dotnet run --project ProjectAI -- help
```

## Status
**Stage 0 complete (S0-1 … S0-6); Stage 1 in progress.** Stage 0 gave a CPU oracle (elementwise incl.
`Sub`/`Div`/`Sqrt`/`Sigmoid` with broadcasting + strided support, batched `MatMul`, axis reductions, all
vs a double-precision reference), reverse-mode autograd (`Autograd` facade + `Tensor.Backward`:
topo-order accumulation, broadcast-aware grad reduction, differentiable view ops, `no_grad`), and `AdamW`
(decoupled weight decay, per-parameter timestep). **S1-2 done**: numerically-stable `Softmax`, `RmsNorm`,
`SiLU`, rotate-half `RoPE` on the oracle, each with a differentiable `Autograd` wrapper (closed-form
backward) that passes finite-difference gradient checks. **S1-1 done**: byte-level BPE tokenizer
(`BpeTokenizer` + `BpeTrainer`) — deterministic training, lossless roundtrip on well-formed UTF-8
(fail-fast on ill-formed UTF-16), special tokens, JSON save/load. `dotnet run --project ProjectAI --
demo` trains `y = Wx + b` to ~0 loss; clean build. **S0-7/S0-8/S0-9 + S1-6 done**: seedable `PcgRng` +
`Init` (Xavier/Kaiming/normal/…) + centralized `Tolerances`; the `Module` contract (`ParameterContext`,
`Param`, `NamedParameters`, `Forward(input, ForwardContext)`, `IKvCache`); and the transformer modules
`Linear`/`RmsNorm`/`RotaryEmbedding`/`SwiGluFeedForward` built on the autograd facade, each gradient-checked.
**S1-3 done**: token `Embedding` (gather + scatter-add backward; gradient flows only to used rows), a
numerically-stable fused `CrossEntropy` with ignore-index + bounds-checked targets, and tied-LM-head support
(gradient-checked to sum both paths). **S1-7 done** (full-sequence/training path): batched (rank≥2) `MatMul`
autograd; the GQA `Attention` module (Q/K/V/O projections, RoPE on Q/K, grouped-query head sharing via a
size-1 broadcast group axis, scaled dot-product, causal mask, softmax, output projection) — validated against a
hand-rolled reference (1e-4) for MHA/GQA/MQA, gradient-checked, causal, with config validation. **S1-8 done —
the model is runnable end-to-end**: `TransformerBlock` (pre-norm residual) + `LlamaModel` (embedding → N blocks →
final norm → tied LM head) → logits `[batch, seq, vocab]`. An overfit test trains the full model to ~0 loss and
greedily reproduces a sequence, and `dotnet run --project ProjectAI -- train` trains a tiny byte-level LLaMA from
scratch (loss 5.6→0.003 in ~15s) and generates the corpus from a prompt. **S1-9 done** — `ISampler` with
`GreedySampler` (argmax) and `TopKTopPSampler` (temperature → top-k → top-p nucleus, seeded `PcgRng` for
reproducibility); wired into the CLI (`train [prompt] [--temp T] [--topk K] [--topp P] [--seed S]`, greedy by
default; non-finite logits fall back to argmax, and the CLI flag parser fails fast on bad/unknown/out-of-range
flags). **S1-10 done** — the real training loop: `Trainer` (forward → cross-entropy → backward → AdamW, with
gradient accumulation via leaf-grad accumulation and a warmup+cosine LR schedule), `TextDataset` (tokenize + pack
into next-token blocks), and a self-contained little-endian `Checkpoint` (model weights + AdamW moments + step) —
training reload reproduces **bit-identical logits**, resume restores step+moments, and grad-accum matches a full
batch. Wired into the CLI: `train` saves `checkpoints/model.ckpt`; `generate --load <ckpt>` reloads and decodes
with no retraining. (Hardened after an adversarial review: the checkpoint loader validates payload length/rank/
dims against the declared shape; the grad-accumulation equal-token invariant and the resume timestep assumption
are documented; the LR warmup is clamped below the step total.) **S1-11 done** — the usable training CLI:
checkpoints now carry their `ModelConfig` + tokenizer (checkpoint format v2 with a metadata string;
`Checkpointing.SaveModel`/`LoadModel`), so `generate --load` rebuilds the model from the file — no hardcoded
config, and an architecture mismatch is a clear error. `train --data <file> [--steps/--batch/--seqlen/--lr]`
trains on your own text via the real `Trainer` (with a per-step progress callback); default `train` stays the
quick built-in demo. (Hardened after review: the checkpoint loader length-checks the metadata/name regions too,
not just the payload; `--seqlen` is capped to avoid RoPE-table blow-up.) **S1-4 done** — `SafetensorsLoader`
reads a `.safetensors` file into a `StateDict`, materializing every dtype as F32 (F32 bit-exact, BF16 exact
upper-16-bits widening, F16 via `Half`, F64/ints cast); defensive against malformed input (bounds, overflow,
duplicate names, full-coverage check, non-numeric offsets, little-endian-host guard), reads tensors by seeking.
**S1-7b done** — KV-cache incremental decode: `Attention` projects/RoPEs only the new tokens at the cached
position offset, appends this layer's K/V to a `KvCache`, and attends over the full history (per-layer
`layerIndex`; inference-only via `GradMode`); generation prefills then decodes one token per step instead of
re-running the whole sequence. Decode reproduces the full-forward logits (tested to 1e-3, incl. chunked prefill);
the cache guards layer/batch/length bounds and documents its single-stream / uniform-batch assumption.
`dotnet test` is green (218 passing, 0 skipped).
**HTTP API + UI client**: `projectai serve [--models <dir>] [--port N]` serves a local JSON API over
`HttpListener` (`GET /health`, `GET /models`, `POST /generate`; request body capped, inputs validated, no web
framework dependency; `Server.cs`/`ModelRegistry.cs`/`Inference.cs`) over a directory of trained `.ckpt` models —
each request names a model, loaded once and cached, with path-traversal protection. A Godot 4.7 C# client
(DRY/SOLID components: `ApiClient`, `Sidebar`, `Transcript`, `Composer` with a **model picker**, `Palette`) in
`Client/ai-client/` is the Claude-desktop-style front end.
**`convert` done (Tier 2)** — `Converter` + the `convert` CLI map a HuggingFace Llama `config.json` +
`.safetensors` into a `LlamaModel` (HF tensor-name remap; no RoPE permutation since we already use HF's
rotate-half convention; fails loudly on untied embeddings / QK-norm / non-SiLU / attention-bias / **rope_scaling**).
It also loads the model's `tokenizer.json` via **`HfTokenizer`** (faithful byte-level BPE: GPT-2 byte↔unicode map,
rank-ordered merges, EOS/BOS by id) so generated text is meaningful. Checkpoints are now tokenizer-agnostic
(`bpe`/`hf` kinds) and the whole inference path uses `ITokenizer`. Synthetic tests cover the weight round-trip
(bit-identical), the arch guards, the HF-BPE encode/decode (incl. splitting added/special tokens out before BPE), and the `hf`-checkpoint
round-trip. `dotnet test` is green (233 passing, 0 skipped). *(Caveat: CPU/F32 → only small models are practical; byte-exact HF tokenizer
parity and the real end-to-end run happen on the user's machine — no model download in this sandbox.)*
Next in Stage 1: **S1-5** (GGUF); then **Stage 2** GPU backends to make real models fast.
(Deferred: `NamedParameters` ordering at S1-4; **RoPE scaling** + Llama-3 pre-tokenizer regex in convert; paged
KV cache (S3-4) — see `docs/BUILD_PLAN.md`.)
