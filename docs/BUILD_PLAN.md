# ProjectAI — Build & Construction Plan

> Local, self-hosted, general-purpose AI runtime written by hand in C# / .NET 10.
> Toolchain target: **.NET 10 (LTS)**. Plan authored 2026-06-20.

This document turns the researched architecture into an ordered, ticket-ready backlog.
Stages 0 and 1 are specified in depth (they are the immediate work); Stages 2–4 are
summarized and will be expanded as Stage 1 lands. Every ticket has an ID, the files it
touches, explicit **acceptance criteria**, and **dependencies**. Ticket IDs are referenced
directly from `NotImplementedException` messages in the scaffold so the code and the plan
stay in lock-step.

---

## 1. Guiding constraints (non-negotiable)

- **Hand-written ML logic.** Tensors, autograd, optimizers, attention, training — all written by us.
  Standard math / SIMD / IO / binding libraries are allowed (`System.Numerics.Tensors`, `Span<T>`,
  hardware intrinsics). A third-party tensor library (TorchSharp) is permitted **only** as an
  interchangeable compute backend behind `IComputeBackend`, never as the source of model or
  autograd logic.
- **Cross-platform GPU compute.** Primary: Windows + RTX 4090 (CUDA). Secondary: Apple Silicon
  M4 Pro/Max (Metal). Any approach without a credible Metal path is disqualified — this is why
  ILGPU (no Metal) and ComputeSharp (DX12/Windows-only) were ruled out.
- **Clean architecture + DOD.** SOLID/DRY at the seams; data-oriented inner loops on .NET
  high-performance primitives (`Span`, SIMD, `TensorPrimitives`).
- **Two jobs, one core.** Must both *train* small custom models and *run* existing open weights.
- **Modular & swappable.** Three reusable modules over one numeric core: LLM (first priority),
  image generation, 3D mesh generation.

**Scale reality check.** Pretraining from scratch beyond ~1B parameters is infeasible on a single
4090. The architecture is therefore designed so that the same core serves fine-tuning and LoRA of
open weights at larger scales; from-scratch training targets small (≤100M) models for correctness
and research.

---

## 2. Architecture at a glance

Everything is built above a single seam, `IComputeBackend`. Swapping the implementation below the
seam (managed CPU, TorchSharp, or hand-written Vulkan) must require **zero** changes above it.

```
            ┌──────────────────────────────────────────────────────────┐
            │  ProjectAI (CLI host / composition root)                  │
            │  generate · train · convert                               │
            └───────────────┬──────────────────────────────────────────┘
                            │
   ┌────────────┬───────────┼─────────────┬───────────────┬─────────────┐
   │ Tokenizers │  Models   │  Training   │   Formats     │  (Image/Mesh│
   │  (BPE)     │ Llama-like│ loop+AdamW  │ safetensors,  │   Stage 4)  │
   │            │ RoPE/GQA/ │ datasets,   │ GGUF, StateDict           │ │
   │            │ RMSNorm/  │ checkpoints │               │             │
   │            │ SwiGLU    │             │               │             │
   └────────────┴─────┬─────┴──────┬──────┴───────┬───────┴─────────────┘
                      │            │              │
                ┌─────▼────────────▼──────────────▼─────┐
                │            ProjectAI.Core             │
                │  Tensor · Shape · DType · Device      │
                │  Module · GradNode (autograd)         │
                │  IOptimizer / AdamW                   │
                │  IComputeBackend  ◄── THE SEAM        │
                └───────────────────┬───────────────────┘
                                    │ implemented by
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  Backends.Cpu               Backends.Torch              Backends.Vulkan
  TensorPrimitives/SIMD      TorchSharp → libtorch       Silk.NET compute
  (reference oracle)         CUDA (Win) / MPS (mac)      SPIR-V; MoltenVK→Metal
```

**Backend strategy (two real GPU paths + one oracle):**

- **CPU reference** (`Backends.Cpu`) — fully managed, `TensorPrimitives`-based, *not* fast. Its job
  is to be the **correctness oracle**: every accelerated kernel is validated against it.
- **TorchSharp** (`Backends.Torch`) — batteries-included baseline. CUDA on the 4090, MPS on Apple
  Silicon. Gets us to "training works on GPU" fastest.
- **Vulkan** (`Backends.Vulkan`) — hand-written compute shaders via Silk.NET; this is where the
  shader expertise pays off. On macOS the same Vulkan calls reach Metal through MoltenVK.

---

## 3. Solution map

Flat layout, one project per concern, all siblings at the repo root. `Directory.Build.props`
centralizes `net10.0`, nullable, implicit usings, and doc-gen (DRY).

| Project | Responsibility | Depends on | Stage |
|---|---|---|---|
| **ProjectAI.Core** | `Tensor`, `Shape`, `DType`, `Device`; `Module`; `GradNode` autograd; `IOptimizer`/`AdamW`; **`IComputeBackend`** (the seam) | — | 0 |
| **ProjectAI.Backends.Cpu** | Managed reference backend (`TensorPrimitives`/SIMD); correctness oracle | Core | 0 |
| **ProjectAI.Backends.Torch** | TorchSharp/libtorch backend (CUDA/MPS) | Core | 2 |
| **ProjectAI.Backends.Vulkan** | Hand-written Vulkan compute (Silk.NET); MoltenVK on macOS | Core | 2 |
| **ProjectAI.Tokenizers** | Byte-level BPE: train, encode, decode, load/save | — | 1 |
| **ProjectAI.Formats** | `safetensors` + `GGUF` loaders; `StateDict` | Core | 1 |
| **ProjectAI.Models** | Llama-style transformer (RoPE, GQA, RMSNorm, SwiGLU), KV cache, samplers, `ModelConfig` | Core | 1 |
| **ProjectAI.Training** | Training loop, datasets, AdamW driver, checkpointing | Core, Models, Formats, Tokenizers | 1 |
| **ProjectAI** (CLI) | Composition root / host: `generate`, `train`, `convert` | all of the above (GPU backends wired in Stage 2) | 1 |
| **ProjectAI.Tests** | xUnit v3: contract tests, gradient checks, backend conformance | Core, Backends.Cpu, Models | all |

---

## 4. Dependencies & package policy

The CPU/test packages below are confirmed by a successful `dotnet restore`/`build` on the .NET 10 SDK
(10.0.301). The Stage-2 GPU packages (TorchSharp, Silk.NET) are **not** yet referenced by any `.csproj`;
their versions are aspirational and must be re-confirmed against NuGet before Stage 2.

| Package | Version | Used by | Purpose |
|---|---|---|---|
| `System.Numerics.Tensors` | **10.0.9** | Backends.Cpu | `TensorPrimitives`, SIMD span ops |
| `TorchSharp` | **0.107.0** | Backends.Torch (Stage 2) | libtorch bindings |
| `TorchSharp-cuda-windows` | **0.107.0** | Backends.Torch (Win) | bundles libtorch 2.10.0 / CUDA 12.8; **must match `TorchSharp`** (released in lockstep) |
| `libtorch-cpu` | **2.10.0.x** | Backends.Torch (mac/CPU) | CPU/MPS libtorch |
| `Silk.NET.Vulkan` | **2.23.0** | Backends.Vulkan (Stage 2) | Vulkan 1.4.336 bindings; bundles MoltenVK 1.4.1 |
| `Silk.NET.Vulkan.Extensions.KHR` | **2.23.0** | Backends.Vulkan (Stage 2) | KHR swapchain / extension entry points |
| `xunit.v3` | **3.2.2** | Tests | test framework |
| `Microsoft.NET.Test.Sdk` | **17.12.0** | Tests | test host / runner integration |
| `xunit.runner.visualstudio` | **3.1.0** | Tests | VS / `dotnet test` test adapter |

**Policy:** a dependency is allowed if it provides math primitives, SIMD, IO, or device bindings.
A dependency is **disallowed** if it would implement model definitions, autograd, optimization, or
training logic on our behalf. TorchSharp sits behind `IComputeBackend` as a tensor-op provider; we
never call its `torch.nn` model layer.

---

## 5. Staged roadmap

| Stage | Theme | Goal | Exit criteria (Definition of Done) |
|---|---|---|---|
| **0** | Numeric core & autograd | A correct, CPU-only tensor + autograd + optimizer foundation | Gradient-check harness passes on all core ops; AdamW drives a toy regression to convergence; 100% on CPU oracle unit tests |
| **1** | First working LLM | Train a tiny GPT/Llama from scratch on CPU, and load + run an open-weight model | `projectai train` learns a small model whose loss drops on real text; `projectai generate` produces coherent samples; `projectai convert` loads a safetensors checkpoint and matches reference logits |
| **2** | GPU backends | Same models, now on CUDA and Metal, validated against the CPU oracle | Torch + Vulkan backends pass the conformance suite within tolerance; ≥10× training throughput vs CPU on the 4090 |
| **3** | Scale & efficiency | BF16, gradient checkpointing, quantization, LoRA | Fine-tune/LoRA a multi-B open model on the 4090; run a 7B-class model quantized for inference |
| **4** | Image & mesh modules | DiT image gen + MeshGPT-style mesh gen reusing the core | Both modules produce outputs end-to-end on the shared backend |

---
## 6. Stage 0 — Numeric core & autograd  *(deep)*

**Outcome:** a hand-written, CPU-only tensor engine with reverse-mode autodiff and AdamW, proven
correct by finite-difference gradient checks. No GPU, no model yet — just a foundation you trust.

Scaffold status: **Stage 0 is complete (S0-1 … S0-6).** The CPU oracle implements elementwise ops
(broadcasting + non-contiguous), batched `MatMul`, and axis reductions; the `Autograd` facade +
`Tensor.Backward` provide reverse-mode autodiff (topo-order accumulation, broadcast-aware grad
reduction, differentiable view ops, `no_grad`); `AdamW` does bias-corrected, decoupled-weight-decay
updates with a per-parameter timestep; and finite-difference gradient checks cover every differentiable
op. `dotnet run --project ProjectAI -- demo` trains `y = Wx + b` to ~0 loss (the Stage 0 milestone).

### S0-1 — Finalize core value types
- **Files:** `Core/Tensor.cs`, `Core/Shape.cs`, `Core/DType.cs`, `Core/Device.cs`
- **Do:** add strides + row-major offset math to `Shape`; broadcasting rules; contiguous/view
  distinction on `Tensor`; reshape/transpose/slice that produce views where possible; bounds-checked
  indexing in debug.
- **Acceptance:** broadcasting two shapes follows NumPy rules (unit tests for compatible/incompatible
  pairs); a transpose followed by `ToHost` round-trips correct data; `Shape.ElementCount`, `Rank`,
  equality, and hashing covered by tests.
- **Depends on:** —

### S0-2 — Backend seam + CPU skeleton
- **Files:** `Core/IComputeBackend.cs`, `Backends.Cpu/CpuComputeBackend.cs`
- **Do:** lock the op set on the interface (allocation, host transfer, elementwise, matmul,
  reductions, the transformer primitives). Implement allocation/transfer/copy on CPU (done) and
  decide the storage contract (`float[]` handle for now; revisit for dtype generality in S3).
- **Acceptance:** `FromHost`/`ToHost`/`Copy` round-trip; allocating and disposing 10k tensors leaks
  nothing; interface compiles against all three backend stubs. (The full cross-backend conformance
  suite that asserts numerical equality is built in Stage 2, ticket **S2-1**; until then "enforcement"
  is only that every backend implements the interface so the solution compiles.)
- **Known gap:** the seam does **not** yet declare the reduction ops (`Sum`/`Mean`/`Max` along an axis)
  that **S0-3** implements and **S0-4** needs for broadcast/fan-out backward — add them to
  `IComputeBackend` as part of this ticket.
- **Depends on:** S0-1

### S0-3 — CPU reference kernels
- **Files:** `Backends.Cpu/CpuComputeBackend.cs` (+ `Backends.Cpu/Kernels/*.cs`)
- **Do:** implement the math the rest of the system needs from the oracle: tiled GEMM
  (`MatMul`, incl. `transposeB`), reductions (sum/mean/max along axis), broadcasting elementwise.
  Correctness first, then a light SIMD/`TensorPrimitives` pass. (Elementwise add/mul/scalar already done.)
- **Acceptance:** `MatMul` matches a naïve triple-loop reference to 1e-5 on random 1–256 dim cases;
  reductions match NumPy-style references; all ops handle non-contiguous inputs.
- **Depends on:** S0-1, S0-2

### S0-4 — Reverse-mode autograd engine
- **Files:** `Core/GradNode.cs`, `Core/Tensor.cs` (`Backward`), `Core/Autograd.cs` (new: tape + ops)
- **Do:** wrap forward ops so each records a `GradNode` (op name, inputs, local backward closure);
  implement `Tensor.Backward()` with topological ordering and gradient accumulation for fan-out;
  support `requires_grad` propagation and `no_grad` scopes. **`IComputeBackend` ops stay
  non-differentiable**: the differentiable layer lives one level above, in `Core/Autograd.cs`, which
  calls a backend op and then attaches a `GradNode`. `Module.Forward` builds the graph through this
  Core autograd facade — never by calling the backend directly.
- **Acceptance:** **finite-difference gradient check** (S0-6) passes for add, mul, matmul, sum, and a
  small composite expression; double-use of a tensor accumulates (does not overwrite) gradients;
  backward over a 3-op chain matches hand-derived gradients.
- **Depends on:** S0-3

### S0-5 — AdamW optimizer
- **Files:** `Core/Optimizers.cs`
- **Do:** implement `Step()` (bias-corrected first/second moments, decoupled weight decay) and lazy
  per-parameter state allocation; keep `ZeroGrad()` (done); add an optional LR schedule hook.
- **Acceptance:** AdamW minimizes a known convex quadratic to < 1e-4 in < 200 steps; weight decay is
  decoupled (verified against a closed-form step); state survives a `ZeroGrad`/`Step` cycle.
- **Depends on:** S0-4

### S0-6 — Gradient-check + tensor test harness  *(verification)*
- **Files:** `Tests/GradientCheckTests.cs`, `Tests/CoreContractsTests.cs` (exists)
- **Do:** central finite-difference utility (perturb ±ε, compare numeric vs analytic grad within
  tolerance); property tests for shapes/broadcasting; enable the currently-skipped
  `Autograd_NumericGradientCheck` test; backfill the S0-1 `Shape` equality/`GetHashCode` tests
  (currently only exercised incidentally) and a multi-dim incompatible-broadcast case.
- **Acceptance:** every differentiable core op has a passing gradient-check test; CI runs `dotnet test`
  green; coverage of Core ≥ 80%.
- **Depends on:** S0-3, S0-4, S0-5 (grad-checks matmul/reductions from S0-3 and the AdamW convex check from S0-5)

### S0-7 — Determinism, RNG & tolerance infrastructure  *(new)*
- **Files:** `Core/Random.cs` (new: seedable PRNG), `Core/Tolerances.cs` (new)
- **Do:** a centralized, host-side, seedable PRNG used by every stochastic path (parameter init,
  sampling, and dropout later); a single global seed policy; and a `Tolerances` home for the numeric
  comparison constants the tests share, replacing scattered `1e-5`/`1e-4` literals. Satisfies the
  cross-cutting "fixed seeds / centralized tolerances" discipline (§11) that currently has no ticket.
- **Acceptance:** the same seed reproduces an identical random stream across runs and processes;
  `Tolerances` is the only place comparison epsilons are defined; a fixed-seed op test is bit-reproducible.
- **Depends on:** S0-2

### S0-8 — Parameter initialization  *(new)*
- **Files:** `Core/Init.cs` (new), `Core/Module.cs` (init hooks)
- **Do:** initialization strategies (zeros, normal, Xavier/Glorot, Kaiming) that fill registered
  parameters through the active backend, driven by the S0-7 RNG so init is reproducible; model
  construction selects an init per parameter. Without this, S1-10's from-scratch training milestone
  cannot start.
- **Acceptance:** each strategy reproduces the expected mean/variance to tolerance on large samples;
  a fixed seed reproduces identical initial weights; a freshly-initialized `LlamaModel`'s parameters
  are finite and non-degenerate.
- **Depends on:** S0-7

### S0-9 — Module forward contract  *(new)*
- **Files:** `Core/Module.cs`, `Core/ForwardContext.cs` (new), `Models/KvCache.cs`
- **Do:** decide and lock how non-trivial forward inputs flow through `Module` before any model layer
  is written. Introduce a `ForwardContext` (causal mask, position ids, optional `KvCache`, training
  flag) and/or richer per-module signatures so `Attention`/`TransformerBlock`/`LlamaModel` can thread
  mask/positions/cache without redesigning `Core` mid-Stage-1. Define `KvCache`'s read/write API here
  (it is currently a property bag with no methods).
- **Acceptance:** `Module.Forward` (or its agreed replacement) threads a mask, positions, and a
  `KvCache` end-to-end on a 2-layer toy; the single-tensor convenience path still works for simple
  modules; no module needs to reach around the contract.
- **Depends on:** S0-1

**Stage 0 milestone:** `dotnet test` green; a 20-line script trains `y = Wx + b` on synthetic data to
near-zero loss using only our `Tensor` + autograd + AdamW on the CPU backend.

---

## 7. Stage 1 — First working LLM  *(deep)*

**Outcome:** a Llama-style decoder you can (a) train from scratch on CPU on a small corpus and
(b) load open weights into and sample from. Still CPU-only — speed comes in Stage 2.

Scaffold status: `ModelConfig`, `RmsNorm`, `RotaryEmbedding`, `Attention`, `SwiGluFeedForward`,
`TransformerBlock`, `LlamaModel`, `KvCache`, samplers, BPE, safetensors/GGUF loaders, and `Trainer`
all exist as compiling stubs with matching ticket IDs.

### S1-1 — BPE tokenizer
- **Files:** `Tokenizers/Tokenizer.cs`, `Tokenizers/BpeTrainer.cs` (new)
- **Do:** byte-level BPE (GPT-2/Llama style): train merges from a corpus, encode/decode, special
  tokens (BOS/EOS/PAD), save/load vocab+merges JSON.
- **Acceptance:** `Decode(Encode(s)) == s` for a UTF-8 fuzz corpus incl. emoji/CJK; trained vocab
  reproduces known merges on a fixed seed corpus; loads a published Llama tokenizer and matches its
  token ids on a reference string.
- **Depends on:** — (parallelizable with all of Stage 0)

### S1-2 — Transformer numeric ops (CPU)
- **Files:** `Backends.Cpu/CpuComputeBackend.cs`
- **Do:** implement `Softmax` (numerically stable), `RmsNorm`, `Silu`, and `RotaryEmbedding` on the
  CPU backend (currently `NotImplementedException` → "ticket S1-2").
- **Acceptance:** each matches a NumPy/PyTorch reference to 1e-5; softmax is stable for large logits;
  RoPE rotation is exact for known angle tables; all four pass gradient checks.
- **Depends on:** S0-3, S0-4

### S1-3 — Embedding, LM head, cross-entropy
- **Files:** `Models/Embedding.cs` (new), `Models/Loss.cs` (new)
- **Do:** token embedding lookup (+ optional weight tying with the LM head), final projection to
  logits, stable cross-entropy with ignore-index for padding.
- **Acceptance:** embedding gradient flows only to used rows; tied-weight gradient sums both paths;
  cross-entropy matches reference incl. label smoothing off/on; gradient-checked.
- **Depends on:** S0-4, S1-2

### S1-4 — safetensors loader
- **Files:** `Formats/Loaders.cs` (`SafetensorsLoader`)
- **Do:** parse the JSON header, map dtypes, materialize tensors into backend storage (zero-copy
  where the dtype matches), build a `StateDict`.
- **Acceptance:** loads a real `.safetensors` file; tensor shapes/dtypes/values match a Python
  `safetensors` reference within exact bit equality for F32 and correct cast for BF16/F16.
- **Depends on:** S0-2

### S1-5 — GGUF loader
- **Files:** `Formats/Loaders.cs` (`GgufLoader`)
- **Do:** parse GGUF metadata + tensor table; support F32/F16 first, then common quant blocks
  (Q8_0, Q4_K) with dequant to F32 on load.
- **Acceptance:** reads a llama.cpp GGUF; metadata (arch, hyperparams) parsed into a `ModelConfig`;
  dequantized Q4_K tensor matches llama.cpp dequant within tolerance.
- **Depends on:** S0-2

### S1-6 — Norm / RoPE / SwiGLU modules
- **Files:** `Models/Modules.cs`
- **Do:** wire `RmsNorm`, `RotaryEmbedding`, `SwiGluFeedForward` `Forward` (and backward via autograd)
  onto the S1-2 ops and registered parameters.
- **Acceptance:** each module's output matches a reference layer; parameters appear in
  `Module.Parameters()`; gradient-checked end-to-end.
- **Depends on:** S1-2, S0-4, S0-9 (forward contract)

### S1-7 — Grouped-query attention + KV cache
- **Files:** `Models/Modules.cs` (`Attention`), `Models/KvCache.cs` (split out)
- **Do:** Q/K/V projections, RoPE on Q/K, GQA head grouping (`KvHeadCount < HeadCount`), causal
  mask, scaled-dot-product attention via the backend, output projection; incremental decode path
  reading/writing `KvCache`.
- **Acceptance:** full-sequence forward matches a reference attention to 1e-4; incremental decode
  (token-by-token with cache) equals the full-sequence result; GQA with `KvHeadCount=1` and
  `=HeadCount` both correct; gradient-checked on a 2-layer toy.
- **Depends on:** S1-6

### S1-8 — Block + model assembly
- **Files:** `Models/Modules.cs` (`TransformerBlock`, `LlamaModel`)
- **Do:** pre-norm residual block (`x + attn(norm(x))`, `x + ffn(norm(x))`), stack N blocks, final
  norm + LM head; expose forward returning logits.
- **Acceptance:** a configured `LlamaModel` forward produces `[batch, seq, vocab]` logits; parameter
  count matches the closed-form formula for the config; loading S1-4 weights reproduces reference
  logits for a known open model within tolerance.
- **Depends on:** S1-7, S1-3

### S1-9 — Samplers
- **Files:** `Models/Sampling.cs`
- **Do:** greedy/argmax; temperature; top-k; top-p (nucleus); seedable RNG for reproducibility.
- **Acceptance:** greedy is deterministic; temperature→0 approaches greedy; top-k/top-p restrict the
  support set correctly (unit tests on crafted logits); fixed seed reproduces a sequence.
- **Depends on:** S1-8, S0-7 (seedable RNG)

### S1-10 — Training loop
- **Files:** `Training/Training.cs` (`Trainer`), `Training/Datasets.cs` (new)
- **Do:** batched data pipeline (tokenize → pack → shift labels), forward → cross-entropy → backward
  → AdamW step, gradient accumulation, warmup+cosine LR, periodic checkpoint to `StateDict`,
  resumable.
- **Acceptance:** training loss on a small real corpus (e.g. TinyStories subset) decreases
  monotonically over an epoch; checkpoint reload reproduces identical logits; grad-accum of N steps
  equals one N×-batch step within tolerance.
- **Depends on:** S0-5, S1-8, S0-8 (parameter init)

### S1-11 — CLI end-to-end  *(milestone / verification)*
- **Files:** `ProjectAI/Program.cs`
- **Do:** wire `train` (S1-10), `generate` (S1-8 + S1-9), `convert` (S1-4/S1-5 → internal checkpoint)
  to real implementations behind the existing command switch.
- **Acceptance:** `projectai train --config tiny.json` trains and checkpoints; `projectai generate
  --prompt "..."` emits coherent continuation from that checkpoint; `projectai convert model.safetensors`
  loads an open model and `generate` produces sensible text.
- **Depends on:** S1-10, S1-9, S1-4

**Stage 1 milestone:** from a cold checkout, `projectai train` learns a small model whose samples are
coherent, and `projectai convert` + `generate` runs a real open-weight model — all on the CPU backend.

---
## 8. Stage 2 — GPU backends  *(summary)*

Goal: run the *exact same* models on CUDA (4090) and Metal (M4), validated against the CPU oracle.

- **S2-1 — Backend conformance suite.** Parameterized tests that run every op on each backend and
  assert equality with `Backends.Cpu` within tolerance. *This is the safety net for everything below.*
- **S2-2 — TorchSharp backend.** Implement `IComputeBackend` over libtorch tensors; map our `DType`/
  `Device`; CUDA on Windows, MPS on macOS. Fastest route to GPU training.
- **S2-3 — Device memory & lifetime.** Pooled allocation, explicit transfer, dispose discipline,
  stream/queue handling.
- **S2-4 — SPIR-V build pipeline.** GLSL/HLSL compute → SPIR-V at build time; shader hot-reload for
  dev; Silk.NET device/queue/descriptor plumbing.
- **S2-5 — Vulkan compute kernels.** Hand-written GEMM, fused attention, norms, elementwise, RoPE;
  MoltenVK validation on macOS. *Where the shader expertise applies directly.*
- **S2-6 — Backend selection.** Config/CLI flag (`--backend cpu|torch|vulkan`, `--device`), runtime
  capability probe, sensible default per platform.

Exit: conformance suite green on all backends; ≥10× training throughput vs CPU on the 4090.

## 9. Stage 3 — Scale & efficiency  *(summary)*

Goal: make larger open models trainable/fine-tunable on one 4090 and runnable quantized.

- **S3-1 — BF16 / mixed precision** (master weights F32, compute BF16) + loss scaling.
- **S3-2 — Gradient checkpointing** (recompute activations to trade compute for memory).
- **S3-3 — Quantization** for inference: Q8/Q4 (GGUF-compatible) with dequant-on-use kernels.
- **S3-4 — PagedAttention KV cache** (block table) for long-context, multi-request decode.
- **S3-5 — LoRA / fine-tuning** adapters — the realistic path to multi-B models on a 4090.
- **S3-6 — Throughput**: multi-batch, fused kernels, profiling-driven optimization.

Exit: LoRA-fine-tune a multi-B open model on the 4090; run a 7B-class model quantized for inference.

## 10. Stage 4 — Image & mesh modules  *(summary)*

Goal: prove the core is genuinely reusable by adding two more generative modalities.

- **S4-1 — DiT image module** (`ProjectAI.Image`): patch embed, timestep conditioning, diffusion
  transformer blocks (reusing attention/norm/FFN), a noise scheduler, and sampling — all over the
  same `IComputeBackend`.
- **S4-2 — MeshGPT-style mesh module** (`ProjectAI.Mesh`): mesh tokenization (face/vertex sequences),
  autoregressive decode reusing the transformer stack, detokenize to a mesh.
- **S4-3 — Shared generation services**: schedulers, samplers, and checkpoint IO promoted to shared
  code so all three modalities reuse them.

Exit: both modules produce end-to-end outputs on the shared backend.

---

## 11. Cross-cutting concerns

- **Testing strategy.** (1) CPU oracle unit tests; (2) finite-difference gradient checks on every
  differentiable op; (3) cross-backend conformance (Stage 2+); (4) golden-file tests comparing our
  logits to a PyTorch reference for a fixed open model. `dotnet test` gates every change.
- **Benchmarking.** BenchmarkDotNet harness per kernel (GEMM, attention) and per train-step; track
  tokens/sec and peak memory per backend. Optimize against profiles, not guesses.
- **Coding standards.** SOLID at the seams, DOD in inner loops; `Span<T>`/SIMD/`TensorPrimitives`;
  nullable enabled; analyzers on. Consider turning on `TreatWarningsAsErrors` once stubs are filled.
- **Numerical discipline.** Fixed seeds; deterministic CPU path; tolerance constants centralized;
  every new op ships with a reference comparison.
- **CI.** `dotnet build` + `dotnet test` + `dotnet format --verify-no-changes` on push. Native
  GPU/libtorch jobs are opt-in (self-hosted runners with the 4090 / Apple Silicon).
- **Repo hygiene.** Model weights, checkpoints, and the local SDK are git-ignored; large artifacts
  live outside the repo.

---

## 12. Immediate next actions

1. **Build & test the scaffold locally** (verified building clean on the .NET 10 SDK — see note below):
   ```
   dotnet restore
   dotnet build         # 0 warnings, 0 errors across all 10 projects
   dotnet test          # 18 passing, 1 skipped (S0-4 placeholder)
   dotnet run --project ProjectAI -- help
   ```
2. **Continue Stage 0 in order** — S0-1 is done; next is S0-2 → S0-6 plus the new foundation tickets
   **S0-7/S0-8/S0-9**. Stage 0 is the critical path; nothing model-related is trustworthy until
   gradient checks pass.
3. **Parallelize S1-1 (BPE).** It has no dependency on Stage 0 and can be built alongside it.
4. **Stand up CI early** (`dotnet build` + `dotnet test`) so the gradient-check suite guards every commit.

> **Verification note.** The scaffold has been compiled and tested on the .NET 10 SDK (10.0.301):
> `dotnet build` succeeds with **0 warnings / 0 errors** across all 10 projects, and `dotnet test`
> reports **18 passed, 1 skipped** (the S0-4 autograd placeholder), **0 failed**. Every `ProjectReference`
> resolves, the solution config is consistent, and all three `IComputeBackend` implementations cover the
> full interface. (TorchSharp is still commented out in `Backends.Torch`, so no libtorch is downloaded yet.)
