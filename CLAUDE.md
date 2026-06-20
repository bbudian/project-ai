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
| `ProjectAI.Tests` | xUnit v3: contracts, gradient checks, backend conformance | Core, Backends.Cpu, Models |

## SOLID, concretely
- **S — Single responsibility:** one concern per project and per class. A file that mixes
  "what to compute" with "how to compute it on hardware" is a smell — push the *how* below the seam.
- **O / D — Open-closed & dependency inversion:** code against `IComputeBackend`, `ITokenizer`,
  `IWeightLoader`, `IOptimizer`, `ISampler` — never a concrete type. Add a backend or a loader
  without editing a single call site.
- **L — Liskov:** every `IComputeBackend` must be substitutable, enforced by the **conformance suite**
  (each backend must match `Backends.Cpu` within tolerance). A backend that can't pass isn't done.
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
dotnet test                         # contract + gradient-check + conformance tests
dotnet run --project ProjectAI -- help
```

## Status
Stage 0 in progress. Done: the scaffold + **S0-1** — the `Tensor`/`Shape` value types now carry
row-major strides, NumPy-style broadcasting, and zero-copy views (`Reshape`/`Transpose`/`Permute`/
`Slice`) over shared storage, with a stride-aware `ToHost` that materializes any view. CPU elementwise
ops are real; remaining ops are stubs whose `NotImplementedException` messages cite their ticket IDs.
Next: **S0-3** (CPU GEMM + reductions) and **S0-4** (reverse-mode autograd), gated by the **S0-6**
gradient-check harness. See `docs/BUILD_PLAN.md`.
