# ProjectAI — Lineage & Consolidation Layer (Design Spec)

> **Status:** design spec, produced 2026-07-01 by a multi-agent design pass (4 approaches -> adversarial critique -> synthesis), grounded in the actual repo (it read Module.cs/Linear.cs and discovered the LM head is weight-tied). **Depends on M0** (docs/MEMORY_DESIGN.md) and its reserved hooks. This is the second layer: model lineage + the two-way files<->weights consolidation loop. NOT YET IMPLEMENTED.

---
# ProjectAI — Lineage & Consolidation Layer (Unified Design Spec)

> **Status:** proposed, implementable. **Depends on:** M0 (file memory + reserved hooks) landed; S1-* (model/trainer/checkpoint), S2-3 (per-step scope), S3-2 (grad checkpointing) done; S3-1 (BF16) started.
> **Verified against actual code:** `ProjectAI.Core/Module.cs`, `Optimizers.cs`; `ProjectAI.Models/Linear.cs`, `Modules.cs` (**LM head is weight-tied to the embedding** — line 286); `ProjectAI.Training/Checkpointing.cs`; `ProjectAI/ModelRegistry.cs`; `docs/MEMORY_DESIGN.md` §2.4/§3.4/§4/§11.
> **Proposed tickets:** **L1 … L9** (mapped in §10).

This spec synthesizes four design passes and their adversarial critiques into one coherent path. The four passes agreed on the *skeleton* (LoRA over a frozen base, freeze-by-omission in AdamW, lineage in `Meta`, per-user adapters) and their critiques converged on the same four *fatal flaws*. **This unified design keeps the skeleton and structurally resolves every convergent flaw** rather than restating the optimistic version. Where a design lost, §0.1 says why.

---

## 0. Summary, the three-tier picture, and the two-way loop

ProjectAI can do something no API-backed runtime can: **a running model can rewrite its own weights from its own file memory, locally and auditably.** That is the entire reason this layer exists. But the four critiques established, with code evidence, that the *naive* version of "the model learns overnight" is dangerous at this model scale. So the unified design commits to one inverted principle that dissolves most of the flaws:

> **Files are the source of truth. Weights are a *cache* of well-proven, stable facts. Consolidation is a cache-fill; it never makes a fact's only home the weights.**

### The three tiers

| Tier | Physical location | Mutability | Read path at inference | Owner |
|---|---|---|---|---|
| **INHERITED** | Frozen base `.ckpt` weights (+ frozen per-user adapter delta) | Immutable for the life of the checkpoint | *Implicit* — it's the forward pass | this layer |
| **LONG-TERM** | M0 memory cards (`.md` + frontmatter), per-user store | Editable text | RECALL injects selected cards into context | M0, retagged by this layer |
| **SHORT-TERM** | Live `ForwardContext` / prompt window / KV cache | Ephemeral | The prompt itself | inference path |

### The two-way loop

- **UP (consolidation / "learning"):** eligible LONG-TERM cards → training pairs → a small trainable **adapter delta** over the frozen base → verified against an **external, held-out** gate → on pass, cards are marked `tier: inherited` **but stay RECALL-eligible at a reduced weight** (the critical departure — see §6). The weights now *accelerate* recall of the fact; the file remains the correctable ground truth.
- **DOWN (correction):** a fact baked into frozen weights that is *wrong* is corrected by writing an authoritative override card that RECALL surfaces and ranks above the stale belief. Because consolidated cards were never fully removed from RECALL (§6), the DOWN-flow never has to fight an always-on silent weight alone.

```
        UP: files "learn" into a weight-cache (never the sole home)
  LONG-TERM ── ExportForTraining ─► synth+mask ─► ADAPTER delta ─► EXTERNAL eval gate ─► retag inherited
   cards        (M0 hook)          (PromptComp)    (LoRA, frozen base)   (held-out)      (still RECALL-able, §6)
     ▲                                                                                          │
     │  DOWN: an override card outranks the stale baked belief (RECALL, not a weight edit)      │
     └──────────────────────────────  correction card (trust:curated) ◄─────────────────────────┘

        Serve time:  shared frozen base  ⊕  this user's adapter delta   (selected by userId, §7)
```

### 0.1 Key decisions, and why the alternatives lost

| Decision | Chosen | Rejected alternative | Why it lost |
|---|---|---|---|
| **UP write mechanism** | **LoRA adapter** default; full-FT is an explicit "re-baseline the base" op | Full-FT as default (Design 2) | Per-user multi-client can't afford a full model/user (~3.3 GB for `large`); adapters give rollback-by-file-delete and structural forgetting resistance. Design 2's own cost table proves the point. |
| **Do consolidated cards leave RECALL?** | **No — retag `inherited`, keep RECALL-eligible at reduced weight** | Retag `inherited` + halt RECALL (Designs 1, 4) | Every critique's #1 fatal flaw. Halting RECALL makes consolidation a one-way door whose only exit (DOWN-flow) needs strong in-context override that 0.1–1.5B models lack. Keeping the card cheap-but-present makes UP an *accelerant*, not a *replacement*. |
| **Eval gate** | **External, human/independently-authored held-out probe set, versioned with the base; capability probe is task-accuracy, not mean-perplexity** | Gate on paraphrases of the trained cards + fixed canary perplexity (all four designs) | All four critiques: self-referential (trained-on-test) reward side + mean-perplexity blind to *localized* forgetting = a gate that grades its own homework. |
| **DOWN "negative consolidation"** | **Dropped from v1** (override card only; permanent fix waits for the *positive* re-bake of the correction) | Bake a corrective delta to un-learn (Designs 1, 2) | Low-rank subtraction over a tied head is unreliable; stacking corrective deltas compounds drift. Critiques 1 & 3. |
| **Freeze mechanism** | Frozen tensors go in a **`Buffers()` set, excluded from `Parameters()`** | `RegisterParameter` + rely on every caller filtering `TrainableParameters()` (Designs 1, 3, 4) | Critique 4's footgun: `Parameters()` and the trainable set diverge, so the existing `new AdamW(model.Parameters())` call site silently trains the frozen base. Fix at the source. |
| **Adapter attach points** | **Q, V, and the FFN (`gate`/`up`/`down`)** by default; **never** the tied embedding/LM head or norms | Q,V only (Designs 1, 4) | The reversal/fact-recall critiques: Q,V-only can't move output logits enough to install a fact; FFN is where factual associations live. Embedding stays frozen because it's the tied output head (§3.4). |
| **Cross-subsystem commit** | **Weights-first, retag-last, with a journal** (§5.4) | Retag + save with no transactionality (Design 1) | Critique 1: a crash between steps loses a fact from *both* tiers. A journal + "retag is idempotent and last" closes it. |
| **Serve cache** | **Bounded LRU** over composed `(base, user)` models, base pinned | "load-once, cache-forever" (Design 4) | Critique 5: per-user cache with no eviction is a resident-memory leak the project already fought twice (S2-3, S3-2). |

---

## 1. What "inherited" means, concretely

An INHERITED fact lives in frozen weights: either the shared base `.ckpt`, or a frozen per-user adapter delta over it. It is never surfaced *primarily* by RECALL (the model computes it in the forward pass), but — unlike the rejected designs — the consolidated card is **retained and remains RECALL-eligible at a reduced tier weight** so the fact is correctable and re-verifiable. This is the hinge decision of the whole spec: it makes the weight tier a *cache*, not a *tomb*.

Physically: after inheritance the child's `.ckpt` **is** the parent's weights (bit-identical at the same dtype) plus a `MemoryLineage` in `Meta` naming the parent. There is no separate "inherited weights file"; the base tensors *are* the inherited tier.

---

## 2. Model inheritance — starting a child from a base

Inheritance is warm-start, not conversion: **build a `LlamaModel` with the parent's `ModelConfig`, then copy the parent's weights in instead of random init.** Every piece exists (`Checkpointing.LoadModel`, `ApplyWeights`).

```csharp
// ProjectAI.Training/Lineage.cs
public static class Lineage
{
    /// Build a child that INHERITS a base checkpoint's weights (copied in, frozen) and carries a
    /// MemoryLineage back to the parent. Attaches fresh zero-init adapters per `spec` (LoRA mode)
    /// so the child is bit-identical to the base until trained (adapter B = 0 ⇒ ΔW = 0).
    public static ChildModel BeginChild(
        string baseCheckpointPath, IComputeBackend backend, AdapterSpec spec, DType? computeDType = null);
}

public sealed record ChildModel(
    LlamaModel Model,                 // base (frozen) + adapters (trainable), one Module tree
    ModelConfig Config,
    ITokenizer Tokenizer,
    string BaseCheckpointId,          // content hash of the base .ckpt (see §5.2)
    MemoryLineage Lineage,
    IReadOnlyList<LoraLinear> Adapters);
```

`BeginChild` calls `Checkpointing.LoadModel` (builds the base at its stored `ComputeDType`), then wraps targeted `Linear`s in `LoraLinear` (§3). **Bit-identity caveat resolved:** if `computeDType` differs from the base's, the copy is a downcast and identity is *approximate*, not exact — `BeginChild` records the actual dtype in the lineage and the gate (§5) compares against the *freshly-forked child*, not an assumed-identical base, so the property the gate relies on is "child-before-training == child-after-forking," which always holds.

Full-FT ("re-baseline") reuses the same entry with `spec.Mode = Full`: no adapters, all base tensors trainable, produces a *new base* `.ckpt`. Reserved for the operator-driven promotion path (§7.3), not per-user consolidation.

---

## 3. The adapter primitive vs full-FT — decision and C# design

### 3.1 Decision: **LoRA adapter is the default; full-FT is an explicit re-baseline op.**

Rationale specific to ProjectAI (all four passes agreed; the critiques only sharpened *where* it applies):

1. **Multi-client requires it.** N users × a few-MB delta over one shared base is the only thing that fits (Design 2's cost table: a `large` full model is ~3.3 GB resident; a rank-16 delta is single-digit MB).
2. **Rollback is a file delete** — additive deltas are separable; full-FT rollback needs the whole prior checkpoint retained.
3. **Structural forgetting resistance** — the base literally cannot move (§8).
4. **We own the stack** — LoRA is `MatMul+MatMul+Add` over existing autograd-facade ops, so it needs **no new backend op** and is auto-covered by the S2-1 conformance suite and S0-6 gradient checks. This is the single most defensible engineering call across all four designs.

Full-FT stays as the opt-in path to mint a *new frozen base* others inherit (§7.3).

### 3.2 The primitive: `LoraLinear`

```csharp
// ProjectAI.Models/LoraLinear.cs
/// y = x·Wᵀ  +  (α/r)·(x·Aᵀ)·Bᵀ.  W is the inherited base weight (FROZEN, a buffer, not a parameter).
/// A:[r,in] trainable (Kaiming), B:[out,r] trainable (ZERO-init ⇒ ΔW starts exactly 0).
/// Reuses Linear's [out,in] transposed-matmul convention, so it drops in anywhere a Linear is used.
public sealed class LoraLinear : Module
{
    private readonly Tensor _weight;      // inherited base, FROZEN — registered as a BUFFER (§3.4)
    private readonly Tensor _a, _b;       // adapter, trainable
    private readonly float  _scale;       // α / r
    public int Rank { get; }
    public bool Enabled { get; set; } = true;   // serve-time toggle; disabled ⇒ pure base

    public LoraLinear(ParameterContext ctx, Tensor frozenBase, int inDim, int outDim, int rank, float alpha) : base(ctx)
    {
        _weight = frozenBase;
        RegisterBuffer("weight", frozenBase);                     // NOT in Parameters() ⇒ optimizer never sees it (§3.4)
        _a = Param("lora_a", new Shape(rank, inDim),  Init.Normal(0f, 0.02f));
        _b = Param("lora_b", new Shape(outDim, rank), Init.Zeros); // ΔW == 0 at construction
        _scale = alpha / rank; Rank = rank;
    }

    public override Tensor Forward(Tensor input, ForwardContext ctx)
    {
        var baseOut = /* Linear-style flatten */ Ag.MatMul(Flatten(input), _weight, transposeB: true);
        if (!Enabled) return Restore(baseOut, input);
        var low = Ag.MatMul(Ag.MatMul(Flatten(input), _a, transposeB: true), _b, transposeB: true);
        return Restore(Ag.Add(baseOut, Ag.MulScalar(low, _scale)), input);
    }
}
```

**Zero-init `B`** ⇒ a freshly-forked child reproduces the base exactly (at the same dtype), which is what makes verify-before-bake meaningful.

### 3.3 Freezing the base — resolved at the source (`Buffers`, not caller discipline)

`AdamW` iterates the `IReadOnlyList<Tensor>` it was constructed with and skips `p.Grad is null` (verified: `Optimizers.cs:90,92`). Freezing is therefore "don't hand it the tensor." Designs 1/3/4 did this via `RegisterParameter` + a `TrainableParameters()` filter — but Critique 4 correctly flagged the footgun: the existing `Trainer` does `new AdamW(model.Parameters()...)`, so a frozen base registered as a *parameter* would be silently trained.

**Unified fix:** frozen base weights are registered as **buffers**, a set `Parameters()` does *not* enumerate. Small, SOLID additions to `Module`:

```csharp
// ProjectAI.Core/Module.cs — additive
private readonly Dictionary<string, Tensor> _buffers = new();
private readonly List<string> _bufferOrder = new();

/// Registers a non-trainable tensor (frozen weight, constant). Excluded from Parameters()/NamedParameters()
/// so no optimizer can ever update it; INCLUDED in a separate Buffers()/NamedBuffers() for checkpoint IO.
protected void RegisterBuffer(string name, Tensor buffer) { /* mirrors RegisterParameter into _buffers */ }

public IEnumerable<(string Name, Tensor Buffer)> NamedBuffers();   // recursive, stable order (like NamedParameters)
```

Consequences, all favorable:
- The existing `new AdamW(model.Parameters()...)` in `Trainer` **automatically** trains only adapter A/B — no call-site audit, no divergence footgun.
- `p.Grad` is never populated for a buffer (no grad node), a second independent guard.
- Checkpoint IO enumerates `NamedParameters() ∪ NamedBuffers()`; the *adapter* save format writes only `NamedParameters()` (the A/B tensors), so the multi-GB base is **never** duplicated per user (closes Critique 5's `RegisterParameter("weight")`-in-full-checkpoint landmine).

### 3.4 Attach points

Selected by glob over the stable dotted names from `NamedParameters()` (e.g. `block.3.attn.wq.weight`). Attention/FFN register their `Linear`s under fixed names (verified: `wq/wk/wv/wo`, `gate/up/down`), so globs are deterministic.

```csharp
public sealed record AdapterSpec(
    string[] TargetGlobs,          // default below
    int Rank = 16, float Alpha = 32f,
    AdapterMode Mode = AdapterMode.LoRA);   // LoRA | Full

// DEFAULT targets: attention Q/V + the whole FFN. NOT Q,V-only (Designs 1/4).
static readonly string[] DefaultTargets = {
    "block.*.attn.wq", "block.*.attn.wv",
    "block.*.ffn.gate", "block.*.ffn.up", "block.*.ffn.down",
};
```

**Why FFN is in the default set and the embedding is never:** the critiques were unanimous that Q,V-only cannot move output logits enough to install a fact, and that the MLP is where factual associations concentrate. But the **LM head is weight-tied to the embedding** (verified: `Modules.cs:286`), so adapting the embedding for output *simultaneously corrupts input token representations* — a documented trap. We therefore leave embedding/norms/LM-head frozen and rely on **FFN + Q/V** deltas to shift the residual stream feeding the frozen tied head. This is an honest capacity bound (§11 open risk R1), not a claim that any fact is installable.

Attach is a post-construction visitor. It needs one `Module` capability to swap a registered child while preserving order:

```csharp
// ProjectAI.Core/Module.cs — additive
protected internal bool TryReplaceChild(string name, Module replacement); // same registration slot
public void VisitLinears(Action<string /*dotted path*/, Module /*parent*/, string /*localName*/, Linear> visit);
```

`AdapterAttacher.Attach(model, spec, ctx)` walks the tree; for each `Linear` whose path matches a glob it builds a `LoraLinear` over that `Linear`'s existing `_weight` (reference-preserved — the frozen base stays the same object the checkpoint loaded) and calls `parent.TryReplaceChild(localName, adapter)`. Detach restores the plain `Linear`.

### 3.5 Save/load format and relation to `Checkpointing.Meta`

An adapter is its **own** small `.adapter.ckpt`, reusing the hardened `Checkpoint` binary format (`PAICKPT2`, same defensive bounds-checking). Payload = only the `lora_a`/`lora_b` tensors (+ their AdamW moments, if resumable). The base is **not** duplicated.

`Meta` gains the reserved nullable trailing field (back-compat exactly like `ComputeDType`, verified `Checkpointing.cs:19`):

```csharp
private sealed record Meta(
    ModelConfig Config, string TokenizerKind, string Tokenizer,
    DType ComputeDType = DType.F32,
    MemoryLineage? Lineage = null);              // ← reserved by M0 §11

public sealed record MemoryLineage(
    string?  ParentCheckpointId,                 // content hash of the base this rides on (null = genesis)
    int      ConsolidationStep,                  // monotonic: base=0, each bake +1
    string[] ConsolidatedMemoryIds,              // M0 card ids folded into THIS delta's generation
    string   Kind,                               // "root" | "adapter" | "full-ft"
    AdapterManifest? Adapter = null,             // non-null iff Kind=="adapter"
    string?  EvalGateReport = null);             // the external-gate metrics that let this bake through

public sealed record AdapterManifest(string[] TargetGlobs, int Rank, float Alpha, string BaseCheckpointId);
```

`Checkpointing.SaveAdapter` / `LoadChild(baseId, adapterPath, backend)`: load base frozen, **verify `manifest.BaseCheckpointId == baseId` (hard error on mismatch)**, re-attach per manifest, copy A/B in. Same "clear error, not silently-wrong" ethos as the existing arch-mismatch guard.

---

## 4. The consolidation UP-loop, end to end

Runs as a job on the **same `Trainer` / `POST /train` path** as Stage-2 special-token training (the reserved shared hook), so it inherits grad-accum, warmup+cosine LR, S2-3 scoping, S3-2 checkpointing, and the one-job-at-a-time gate for free.

```
① SELECT      IMemoryStore.ExportForTraining(Eligible)  →  IEnumerable<(Prompt, Completion)>   [M0 hook]
② SYNTHESIZE  PromptCompletionDataset(pairs, tokenizer)  + REPLAY mix  →  IDataset
③ TRAIN       BeginChild(base, spec:LoRA) → Trainer.Train(child, dataset, consolidationCfg)   [adapters only]
④ VERIFY      EXTERNAL held-out gate (recall on independent probes + capability task-accuracy)  [§5]
⑤ COMMIT      (pass) journal → SaveAdapter → retag cards inherited (still RECALL-able) → bump step  [§5.4]
              (fail) discard delta, nothing retagged, report failing gate
```

### 4.1 Eligibility predicate (§1 select)

Every field is live M0 frontmatter (verified §2.2 table). The predicate is deliberately strict — we bake only *stable, trusted, recurring, non-conflicting* facts:

```csharp
bool Eligible(MemoryCard c) =>
    c.Tier is "long" or "core"          // not already inherited, not raw session
 && c.Trust == "curated"               // never bake chat/untrusted (the injection trust boundary, §2.4)
 && c.Status == "active"               // not superseded/retracted/unverified
 && c.Confidence >= 0.7f
 && c.Uses >= MinUses                  // it actually recurred in real use
 && StableFor(c.AsOf, MinDwell)        // survived a cool-down without being retracted (stability, not just trust)
 && !ConflictsWithAny(c, corpus);      // contradictory pairs are held out for human resolution, never baked
```

`ConflictsWithAny` is best-effort (subject/attribute overlap on `keys`); on any ambiguity it **holds out** rather than bakes (Critique 3/§11 R3).

### 4.2 Data synthesis (`PromptCompletionDataset`)

`ExportForTraining` yields `(Prompt, Completion)`. A new `IDataset` (the confirmed clean seam; `TextDataset` drops trailing remainders and trains loss on every token):

```csharp
// ProjectAI.Training/Datasets.cs
public sealed class PromptCompletionDataset : IDataset
{
    // Each pair → [BOS] prompt [SEP] completion [EOS], packed to (seqLen+1) blocks (multiple short pairs/block).
    // Emits a parallel LOSS MASK: prompt positions → ignore-index (S1-3 CrossEntropy already honors it),
    // so we train "given the question, produce the fact," not "reproduce the question."
    public ReadOnlyMemory<int> GetSequence(int i);
    public ReadOnlyMemory<int> GetLossMask(int i);   // NEW seam; TextDataset returns all-ones (back-compat)
}
```

**Where does the prompt come from?** (Critique 5's self-amplification concern.) M0 bodies are free-form Markdown, not QA pairs. The prompt is **template-derived from the card's `title`/`keys`, not model-invented** (e.g. title → a canonical question form). Paraphrase augmentation uses fixed templates, not the model generating its own training questions. This removes the "model invents its own training data and grades itself" loop.

`IDataset.GetLossMask` is additive: `TextDataset` returns all-ones, so existing training is byte-identical (the one honest, well-scoped `Trainer.NextBatch` change — write ignore-index where the mask is zero). Because masking makes tokens-per-micro-batch non-uniform, consolidation defaults to **grad-accum = 1** until the trainer's documented "weight by valid-token count" follow-up lands; sufficient for the small curated corpora we bake (§11 R5).

### 4.3 Orchestrator + server wiring

```csharp
// ProjectAI.Training/Consolidator.cs
public static ConsolidationResult Consolidate(
    IComputeBackend backend, string baseCkptPath, string adapterOutPath,
    IMemoryStore store, Func<MemoryCard,bool> select,
    AdapterSpec adapter, ConsolidationConfig cfg, EvalGate gate, Action<int,float>? onStep = null);

public sealed record ConsolidationResult(
    bool Baked, string? AdapterPath, string[] ConsolidatedIds,
    float RecallAfter, float CapabilityDelta, string? RejectReason);
```

`POST /train` gains `mode: "scratch" | "consolidate"` + `parentCheckpointId`; `TrainingService.Run` branches to `Consolidator.Consolidate` using the requesting user's store. The existing `_job`/`_gate` one-at-a-time lock and the `/generate`-gated-during-training rule apply unchanged. A `GET /consolidate/preview` dry-run (`Baked=false`, no retag) powers the "N memories ready to bake" UI.

`ConsolidationConfig` carries consolidation-specific defaults: low LR (`1e-4`), small `MaxSteps`, `ReplayFraction=0.2` (§8), `GradientCheckpointing` auto for `large`.

---

## 5. Verify-before-bake, the eval gate, and rollback

**No consolidation commits until it passes an external gate.** This is the safety core, and it is where all four original designs scored 3/10 — because their gates graded themselves. The unified gate is redesigned.

### 5.1 The eval gate — external and held-out (the fix)

```csharp
public sealed record EvalGate(
    string ProbeSetPath,           // EXTERNAL: shipped WITH the base, independently authored, NOT derived from the baked cards
    float MinRecall = 0.7f,        // consolidated facts must be answered on INDEPENDENT probes
    float MaxCapabilityDrop = 0.03f, // held-out capability battery: TASK-ACCURACY drop, not mean perplexity
    int   ProbeSamples = 256);
```

Two gates in series, both fixing a specific critique:

1. **Recall gate (did it learn — for real?).** For each consolidated fact, score the candidate (base+new-adapter) on **independently-authored held-out probes** distributed with the base, *not* paraphrases of the trained card. RECALL is suppressed during the probe (we test the *weights*). This measures knowledge, not memorization of surface form — resolving the "trained-on-test" flaw all four critiques raised. Practically: the base ships a probe set; consolidation may only bake facts that have a matching held-out probe, forcing knowledge that generalizes past the trained phrasing.
2. **Capability gate (did it break?).** A fixed held-out battery scored by **task accuracy on discrete Q&A / classification items**, not mean perplexity. Localized forgetting shows up as specific items flipping wrong — which mean-perplexity averages away (Critique 4). Reject if accuracy drops > `MaxCapabilityDrop` vs. the freshly-forked child.

The report (metrics + which probes) is stored in `Meta.Lineage.EvalGateReport` for audit. **Fail → discard the delta, retag nothing.**

### 5.2 Checkpoint identity (content hash) — resolved

`BaseCheckpointId` is SHA-256 over the checkpoint's **payload *and* the identity-bearing metadata (config, tokenizer, dtype), but excluding the `Lineage` field itself** (which would be a self-reference cycle — Critique 4/5). This closes both the collision gap (two models with identical weights but different tokenizers no longer alias) and the cycle (re-tagging/adding lineage doesn't change the child's own id).

### 5.3 Rollback

Trivial by construction (additive, versioned, separable):

- **Adapter rollback:** point the registry at `step{n-1}` (or bare base = `step0`); no retraining. The bad delta is retained for forensics.
- **Un-retag:** each baked card recorded `bakedInto = deltaId`; on rollback, flip exactly those ids from `inherited` back to `long`, restoring full RECALL weight.
- **Bisect a bad fact:** `Meta.Lineage.ConsolidatedMemoryIds` lists exactly which cards each step baked, so we can re-bake *minus* one suspect card without redoing history (requires pinned seed/LR/data — the project's determinism culture supports this; the cost is retained per-generation export data, budgeted in §11 R6).

### 5.4 Cross-subsystem transactionality (the fix for Critique 1)

Promote touches **two** subsystems: the weights (adapter file) and the M0 store (retag). A crash between them could leave a fact in *neither* tier (card tagged `inherited` → down-weighted in RECALL, but adapter not persisted). Because we **keep consolidated cards RECALL-eligible** (§6), this failure is already far less severe. We further guard it with a journal and ordering:

1. Write adapter to `adapterOutPath.tmp`, fsync, atomic-rename (durable artifact first).
2. Append a **commit journal** entry `{deltaId, consolidatedIds, step}`.
3. Retag cards `inherited` (idempotent; each records `bakedInto=deltaId`).
4. Mark journal entry done.

On startup, an unfinished journal entry means "adapter durable, retag maybe incomplete" → re-run step 3 (idempotent). The invariant: **a durable, verified adapter exists before any card is retagged, and retag is last and replayable.** Worst case is a valid adapter with a card still `long` — harmless (RECALL keeps injecting a fact the weights also know; next pass dedups).

---

## 6. The DOWN-flow and read-side inherited-tier behavior

The interesting, hard direction — and where the unified design most departs from the rejected passes.

### 6.1 Read-side behavior of `inherited` (the departure)

The rejected designs made RECALL **skip** `inherited` entirely, so a baked fact's only correction path was in-context override. Every critique showed this makes consolidation a one-way door that fails exactly when you need it (weak in-context obedience at 0.1–1.5B; retrieval must already know the topic is contested).

**Unified rule:** RECALL keeps `inherited` cards eligible but at a **reduced tier weight** (extend the existing `tierWeight(mem.tier)` in the ranking, verified `MEMORY_DESIGN.md:202`, with `core > long > inherited`). Consequences:
- The context window stays lean (inherited facts rank low, rarely injected when the model already answers well).
- But the card is **still there** — so a correction can outrank it, and a re-verification can re-export it. The weight tier is a *cache*, not a *tomb*.

### 6.2 Correcting a wrong inherited fact — externalize, don't excise

Weights are frozen; you cannot edit a fact out. So a correction is a high-priority LONG-TERM card that wins at serve time:

1. **Trigger:** user flags a wrong answer ("correct this") or an eval probe fails.
2. **Write an override card:** `tier: long`, `status: override` (a new status that *supersedes a weight*), `trust: curated`, `confidence: 1.0`, `provenance.source: correction`, `supersedes: <baked id if known>`, fresh `asof`.
3. **Serve-time precedence:** override cards get top RECALL priority + a system-prompt directive ("a retrieved corrected fact outranks your prior belief"). Because the baked card was **never removed** from RECALL (§6.1), the override competes against a *present, down-weighted* card rather than an *invisible always-on weight* — the DOWN-flow's coverage is now bounded by ordinary RECALL, not by the model's in-context-override strength alone.
4. **Retract the baked belief:** set the original baked card `status: retracted` with `supersededBy` → the override. It's now excluded from the next consolidation selector (`status=="active"` gate), so the error never re-bakes.
5. **Eventual permanent fix:** the override card is `curated`/high-use, so a *future positive consolidation* bakes the *correction* (the loop closes by re-baking the right fact — never by "negative consolidation," which §0.1 rejected as unreliable).

### 6.3 Honest bound (Critique 2/3/4)

The DOWN-flow's coverage equals RECALL's coverage: a correction only fires when retrieval surfaces the override for the query. This is a real limit (§11 R2), materially softened by §6.1 (the baked fact is down-weighted, not silently dominant) and by keeping volatile/correctable facts out of the eligibility predicate entirely (§4.1 `StableFor`).

---

## 7. Multi-client — per-user deltas over a shared frozen base

**Decision: one shared frozen base + per-user LoRA deltas, selected by `userId` at serve time, bounded-LRU-cached. Consolidation is per-user; the base is global; promotion to a new base is an operator-gated offline op.**

This is the only design that scales to per-user partitioned memory stores. Global-single-delta (leaks users' knowledge into one model) and per-user-full-model (~3.3 GB/user for `large`) are both rejected (§0.1 / Design 1 §7.1).

### 7.1 Storage layout

```
models/
  base/  smollm2-360m.ckpt              # shared frozen base (one resident copy) — INHERITED tier
  users/
    alice/  {baseId}.step1.adapter.ckpt  {baseId}.step2.adapter.ckpt  current -> step2
    bob/    {baseId}.step1.adapter.ckpt  current -> step1
```

### 7.2 Serve-time selection and caching — mirror `ModelRegistry`, with eviction

An `AdapterRegistry` analogous to the verified `ModelRegistry` (same lazy-load, path-traversal-safe pattern — `ResolvePath`'s invalid-char + inside-dir check, extended one level to `users/<uid>/`), **plus a bounded LRU** (the fix for Critique 5's unbounded per-user cache):

```csharp
internal sealed record ServeKey(string BaseName, string? UserId);

internal sealed class AdapterRegistry
{
    private readonly ModelRegistry _bases;                          // shared frozen bases, loaded once
    private readonly LruCache<ServeKey, LoraLinear[]> _adapters;    // BOUNDED (config: MaxCachedAdapters); base pinned

    public ServedModel? Resolve(string baseName, string? userId);   // base (cached) + user's adapters (LRU) or base-only
}
```

Request flow (`/generate` carries `model` + authenticated `userId`):
1. `_bases.Get(baseName)` → shared frozen base (one cache entry, one set of weights on the GPU).
2. Resolve `users/<uid>/current`; if absent → **serve the bare base** (valid fallback for new users).
3. Load the small A/B tensors (verify `BaseCheckpointId`), cache in the bounded LRU.
4. **Bind:** set the active adapters. Because the server is already **one-inference-at-a-time** (`InferenceLock`, and `/generate` gated during `/train`), binding is serialized with inference — no data race on the shared base graph.

**Concurrency & KV-cache honesty (Critique 5):** the existing `KvCache` documents a **single-stream / uniform-batch** assumption (verified `Modules.cs:298-300`). So true *concurrent* multi-user decode is **not** claimed for v1 — v1 serves users serially behind `InferenceLock`, swapping adapters between requests. The `LoraLinear.Enabled` toggle + reference-swapped A/B make the swap O(#adapter-tensors). A future concurrent server would thread the active adapter set through `ForwardContext` (stateless `LoraLinear` reading A/B from context) so the base graph is never mutated — but that also needs per-stream KV, which is a separate ticket (S3-4 paged cache). We flag this rather than hand-wave it.

### 7.3 Per-user isolation and global promotion

- Consolidation reads **only** the requesting user's store, trains **only** their adapter, retags **only** their cards. The base opens read-only. Path-traversal protection extends to `userId`.
- **Global promotion** (a fact true for everyone) is an operator-gated offline op: run **full-FT** consolidation over a curated *global/shared-scope* store → a new frozen base all users then fork. Private per-user cards **never** bake into the shared base (hard rule, gated by `trust: curated` + an explicit shared-scope check). Base bumps invalidate per-user adapters by `BaseCheckpointId`; **re-basing rolls back the affected `inherited` retags first** (so the cards are RECALL-eligible and re-exportable) — closing Design 1's "global base update strands per-user knowledge" cascade.

---

## 8. Catastrophic-forgetting mitigations

Layered, cheapest first; the first three are defaults, the gate is the backstop:

1. **Frozen base + low-rank delta (structural).** The base distribution cannot be overwritten; the delta is rank-16 scaled by `α/r`. Biggest mitigation, on by default.
2. **Zero-init `B` + low LR + few steps.** Training starts from exact base behavior and moves only as much as the data demands.
3. **Replay/rehearsal mixing (`ReplayFraction≈0.2`).** Mix base-distribution data with the memory pairs so the delta learns the new fact *without* being told to forget everything else. **Source of replay:** a small retained slice of the base's own training/canary corpus **preferred**; self-generated samples from the frozen base only as a fallback — with the explicit caveat (Critique 2) that self-generation under-samples tail knowledge and can launder hallucinations, so it is *not* the headline defense and never the sole one.
4. **Loss-masked completion training** — gradient only on answer tokens (§4.2).
5. **Template paraphrase augmentation** — bake the proposition across fixed phrasings (bounded; more paraphrases ≠ more interference because LR/steps are capped).
6. **Capability gate (§5.1) is the hard backstop** — task-accuracy regression on a held-out battery *rejects* a forgetting delta. This is the measurement the rejected designs got wrong (mean perplexity); fixed here.
7. **Rank/target budget + periodic re-baseline** — small `r`, a bounded target set; when a user's adapter chain grows, re-baseline (merge + re-verify) rather than growing rank unboundedly. **Chain-drift is tracked** across the lineage (a long-horizon capability trend, not just per-bake) — Critique 3's silent-rot risk (§11 R4).

Honest framing (Critique 2): "the base can't move" prevents gross forgetting but not **interference** (a QV/FFN delta shifting behavior on unrelated prompts sharing patterns). The capability gate is what catches interference; it is the load-bearing defense, which is why §5.1 makes it external and item-level rather than aggregate.

---

## 9. Open decisions for the user

1. **Replay corpus:** do we retain a slice of each base's training/canary corpus for rehearsal (better forgetting protection, some storage) or rely on self-generation (free, but weaker and hallucination-prone)? Default proposed: **retain a small canary slice with the base.**
2. **Who authors the external probe set** (§5.1)? It gates every bake. Proposed: shipped **with each base**, independently authored, versioned; a fact is only bake-eligible if it has a matching held-out probe. This is real curation work — accept it or the gate is theater.
3. **Auto vs. human-in-the-loop consolidation:** the critiques warn a one-click "bake N memories" chip turns the failure modes into the default operating regime. Proposed default: **consolidation is explicit and gated, not auto-nudged**; the "ready to bake" preview is informational, the bake is a deliberate action.
4. **Default rank/targets** (`r=16`, Q/V+FFN) vs. a narrower/wider preset — capacity vs. forgetting/interference tradeoff.
5. **Global-promotion governance** (§7.3): who is the operator, what shared-scope gate, how often base bumps happen (each bump forces per-user re-basing).
6. **Chain-length cap** before a forced re-baseline (drift budget, §8.7).

---

## 10. Build order — relative to M0–M4 and the Stage-2 special-token work

M0 (file memory + reserved hooks: `tier: inherited`, `ExportForTraining`, `Meta.MemoryLineage?`) is the prerequisite (verified reserved, `MEMORY_DESIGN.md:9,138,249,414`). This layer is **additive** and rides behind M0; it consumes `ExportForTraining` and writes the `tier`/`status` fields M0 already reserved, so it is largely orthogonal to M1–M4 (RECALL/curate UX) except the DOWN-flow (which is a RECALL ranking rule, lands with M1/M2). Consolidation reuses the **same** `Trainer`/`POST /train` path as the Stage-2 special-token training — the two share the fine-tune infrastructure and the one-job gate; consolidation is "special-token training whose corpus comes from `ExportForTraining`."

| Ticket | Deliverable | Depends on | Effort |
|---|---|---|---|
| **L1** | `LoraLinear` + `Module.RegisterBuffer`/`NamedBuffers` (freeze at source, §3.3); gradient-check; "forked child == base to 0 ULP at same dtype" conformance | S0-6, S2-1 | **S** |
| **L2** | `Module.TryReplaceChild`/`VisitLinears` + `AdapterAttacher`/`AdapterSpec` glob targeting | L1 | **S** |
| **L3** | `Meta.MemoryLineage`/`AdapterManifest`; content-hash `BaseCheckpointId` (excludes Lineage, §5.2); `Lineage.BeginChild`; `SaveAdapter`/`LoadChild` + base-id-mismatch guard | L2, Checkpointing | **M** |
| **L4** | `PromptCompletionDataset` + `IDataset.GetLossMask` + `NextBatch` mask wiring + template/replay | S1-3, M0 export | **M** |
| **L5** | External `EvalGate` (independent held-out probes; task-accuracy capability gate, §5.1); `Consolidator` orchestration; commit journal + retag-last (§5.4); rollback + un-retag + bisect | L3, L4, `Inference` | **M–L** |
| **L6** | Loop wiring: `POST /train mode=consolidate`, `GET /consolidate/preview`, `--rollback`; idle/pressure/cadence triggers on the one-job gate | L5, TrainingService | **M** |
| **L7** | Read-side `inherited` ranking (down-weight, keep eligible, §6.1); DOWN-flow: `status: override` card + serve-time precedence + retract-and-exclude | L6, M1/M2 RECALL | **S–M** |
| **L8** | Multi-client serve: `AdapterRegistry` (shared base + per-user bounded-LRU, §7.2), `ServeKey`, fallback-to-base, path-safety; per-user consolidation routing | L3, ModelRegistry pattern | **M–L** |
| **L9** | Full-FT re-baseline (`Kind=full-ft`) + `Lineage.Merge` (`W += (α/r)BA`) for global promotion + per-user re-basing on base bump (§7.3) | L5, L8 | **M** |

**Ordering rationale:** L1–L3 (primitive + freeze + inheritance + format) are the load-bearing, seam-touching core — small/medium, independently testable against the CPU oracle; **ship first**. L4–L5 deliver the single-node consolidation loop *with the external gate and journaled rollback* — prove "the model learns a fact, verified, reversibly" on one node before any multi-tenant complexity. L5 (verify + rollback) **must land before L6 turns the loop on** so no unverified bake reaches weights. L7 (read-side + DOWN-flow) rides M1/M2. L8/L9 (multi-client serve + global promotion) are the productionization tail. The whole layer is additive: no existing checkpoint, training run, or serve path breaks (freeze via buffers, adapter reuses existing ops + `Checkpoint` format, lineage reuses the defaulted-trailing-param back-compat proven by `ComputeDType`).

---

## 11. Risks (honest, with the mitigation each critique demanded)

| # | Risk (critique convergence) | Mitigation in this design |
|---|---|---|
| **R1** | **Fact installation is hard at small scale / tied head.** Q,V-only can't move logits; embedding is the tied output head and can't be adapted safely. | Default targets **FFN + Q/V** (§3.4); the **external recall gate** (§5.1) refuses to bake facts that don't generalize to independent probes, so a failed install *no-ops* (nothing retagged) rather than shipping a brittle memorization. Honest capacity bound, not a guarantee. |
| **R2** | **DOWN-flow depends on retrieval firing / in-context obedience weak models lack.** | Consolidated cards stay **RECALL-eligible, down-weighted** (§6.1), so a correction competes against a *present* card, not an invisible weight. Volatile/correctable facts are excluded from eligibility (`StableFor`, §4.1). Bound stated, not hidden. |
| **R3** | **Baking a wrong memory is near-permanent; eligibility is the only firewall and it's unenforceable.** | Weights are a **cache, never the sole home** (§0); the retracted-and-excluded flow (§6.2) plus conflict-hold-out (§4.1) plus the external gate (§5.1) mean a wrong fact stays correctable via files and never re-bakes. |
| **R4** | **Gate gameable / self-referential / perplexity-blind to localized forgetting.** | Gate is **external, held-out, independently authored**; capability measured by **item-level task accuracy**, not mean perplexity (§5.1). Chain-drift tracked long-horizon (§8.7). |
| **R5** | **Masked grad-accum not supported by today's trainer.** | Consolidation defaults to **grad-accum=1** (exact for small corpora) until the "weight by valid-token count" trainer follow-up lands (§4.2). Reuse claim scoped honestly. |
| **R6** | **Cross-subsystem rollback race loses a fact from both tiers.** | **Weights-first, retag-last, journaled, idempotent** (§5.4); and §6.1 keeps the card present so the worst case is a redundant RECALL, not a lost fact. |
| **R7** | **Per-user serve cache unbounded → resident leak; concurrent multi-user fights single-stream KV.** | **Bounded LRU**, base pinned (§7.2); v1 serves **serially** behind `InferenceLock` (adapter swap between requests), concurrency + per-stream KV explicitly deferred to a later ticket — not claimed. |
| **R8** | **Consolidation VRAM competes with the resident serving base on the 8 GB dev GPU.** | Consolidation is a job behind the **one-at-a-time gate** with `/generate` gated; adapter AdamW state is MB, but the forward/backward runs the full frozen base — so `large` consolidation is a 4090 job (matches the existing CLAUDE.md caveat); tiny/small/medium consolidate on 8 GB. Stated, not glossed. |
| **R9** | **BF16 base + F32 adapter mixed-precision `Add` is under-specified.** | Adapters default **F32** (they're tiny, so the cost is negligible and training is more stable) over a BF16 frozen base; a mixed-input conformance case is required in L1 before serving relies on it. |
| **R10** | **Chain drift / self-generated-replay hallucination ratchet across generations.** | Prefer a retained canary slice for replay (§8.3, open decision §9.1); cap chain length before forced re-baseline (§8.7); the external capability gate re-runs each generation against a *fixed* battery, catching accumulated rot. |

---

### Files this layer touches (all absolute)

- **New:** `C:\Users\bbudi\Development\Apps\project-ai\ProjectAI.Models\LoraLinear.cs`; `...\ProjectAI.Training\Lineage.cs`, `...\ProjectAI.Training\Consolidator.cs`, `...\ProjectAI.Training\EvalGate.cs`; `...\ProjectAI\AdapterRegistry.cs`.
- **Edited (small, back-compat):** `...\ProjectAI.Core\Module.cs` (`RegisterBuffer`/`NamedBuffers`, `TryReplaceChild`/`VisitLinears`); `...\ProjectAI.Training\Checkpointing.cs` (`Meta.MemoryLineage`, `SaveAdapter`/`LoadChild`); `...\ProjectAI.Formats\Checkpoint.cs` (content-hash id); `...\ProjectAI.Training\Datasets.cs` (`PromptCompletionDataset`, `GetLossMask`); `...\ProjectAI.Training\Training.cs` (`NextBatch` mask wiring); `...\ProjectAI\Server.cs` + `...\ProjectAI\TrainingService.cs` (`mode=consolidate`, `/consolidate/preview`, `--rollback`); `...\ProjectAI.Models\Modules.cs` (attach seam on Attention/FFN `Linear`s).
- **Reused unchanged (proves the fit):** `AdamW` (freezing = buffers omitted from `Parameters()`, no optimizer edit — verified `Optimizers.cs:90,92`); `Autograd` facade (adapter = existing `MatMul`/`Add`/`MulScalar`); `Trainer` (grad-accum/LR/scope); `Checkpoint` binary format (adapter reuses it); `ModelRegistry`/`ComputeRegistry` caching pattern; `IMemoryStore.ExportForTraining` (M0 §11 hook).

The through-line: this is a **per-user LoRA + provenance + verified cache-bake** system, honest about being one. It keeps the genuinely novel, own-the-stack move — files consolidating into an auditable weight cache with a cryptographic lineage chain — while structurally removing the four convergent fatal flaws (one-way `inherited` retag, self-grading gate, leaky DOWN-flow, and the freeze/cache/transactionality footguns) by inverting one principle: **weights cache files; files are truth.**