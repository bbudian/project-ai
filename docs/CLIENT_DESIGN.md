# ProjectAI — One-Stop-Shop Client (Design Spec)

> **Status:** design spec, produced 2026-07-01 by a multi-agent design pass (5 perspectives -> adversarial critique -> synthesis), grounded in the actual Godot client + server code. Companion to the prototype harness in prototype/ (the harness prototypes ALL screens mock-backed; this spec scopes what the SHIPPING Godot client launches with). Visual mockups were shown in-session.
>
> **Notable code-grounded decisions:** ship 4 live destinations (Chat, Models, Benchmark) + Settings as a modal Window; Memory + Upgrade reserved until their server backends exist; Train folds into Models. Benchmark copies the /train background-job gating pattern (NOT holding InferenceLock for a whole run); TTFT needs an explicit server-side stopwatch split (WS token frames carry no timestamp) or is omitted; report bits-per-byte, not cross-tokenizer perplexity; NO LLM-judge in v1; secrets in config/secrets.json OUTSIDE the models dir with a real ACL; single-local-user scoping (MemoryStoreRegistry trusts an unauthenticated string); the net8 client re-declares its own DTOs (can't share types with the net10 server); AppState uses plain C# events, not Godot signals.

---
I have the key facts confirmed. The critiques converge on a clear signal: the shell/router + AppState + settings persistence + a descoped, rigor-first benchmark core are the coherent path; the over-scoped bits (six live tabs, LLM-judge, shared cross-framework records, plaintext secrets in the models dir, fabricated TTFT, cross-tokenizer PPL) are what lose. Now I'll write the unified spec.

Key resolved facts driving decisions:
- `/generate` and `/chat` take `InferenceLock`; `/train` does **not** — it runs on a background `Task` and is gated by a separate `training.IsTraining` 409 check. So benchmark must copy the **train** gating pattern (background thread + `IsBenchmarking` 409 gate + per-case lock), not "hold InferenceLock for the whole run."
- The WS `token` frame carries **no timestamp**; `done` reports total `seconds` only. So TTFT is not client-derivable — it needs an explicit server-side stopwatch split or it must be cut.
- `MemoryPolicy.BridgeCards` etc. are `const int` → the facade-over-settings move is real and cheap.
- `MemoryStoreRegistry.Resolve(user, store)` trusts an unauthenticated string → the user-picker/multi-user framing is unsafe and must be scoped to single-local-user.

---

# ProjectAI One-Stop-Shop Client — Unified Design Spec

Status: proposal, ready to build. Scope: the Godot client shell + two new subsystems (Benchmarking, Settings), with reserved-but-not-built hooks for Memory-curation and Upgrade/lineage. This spec resolves the five reviewed designs into one coherent path and states, at each fork, which option lost and why.

**The through-line (survives all five critiques):** *destinations are a list, not a boolean; cross-tab state is one store with one fan-out; transport is one typed seam per endpoint; background jobs copy the `TrainingService` template (background thread + `IsX` 409 gate + poll + `/health` fan-out); styling is `Palette`-only.* Everything grows by adding one view + one service without reshaping the composition root.

**What lost, up front (the decisions that define the spec):**
- **Six live rail tabs → rejected.** Ship four working destinations (Chat, Models, Benchmark, Settings). Memory and Upgrade are *not* live rail icons; they are reserved (Upgrade behind lineage which is unbuilt; Memory behind read-only render endpoints). Dead "coming soon" icons train users to ignore the rail. All five docs over-populated; the IA critique is decisive.
- **LLM-as-judge in v1 → rejected.** At SmolLM2/CPU scale a local judge is self-preference-biased theater. The metric schema leaves a nullable slot; no judge ships until an external, stronger model is wired.
- **Client reuses server record types → rejected.** net8 client / net10 server cannot share a compiled contract type. The client re-declares its own DTOs; the wire is JSON only. This is stated as a rule, not left implicit.
- **Client-side / WS-derived TTFT → rejected as a metric.** The server emits no per-token timestamp. Either add one explicit server-side first-token stopwatch split (Phase 2b) or omit TTFT. We add the split, once, in the shared inference path — but it is *not* "reuse existing output."
- **Cross-tokenizer perplexity ranking → rejected.** Report **bits-per-byte (bpb)**, never raw per-token PPL, so a byte-BPE model and an HF-tokenizer SmolLM2 are comparable. Same-tokenizer PPL is shown only within a tokenizer family.
- **Secrets in `config.json` under the models dir → rejected.** Secrets live in a separate `config/secrets.json` outside the models tree, git-ignored, locked down with a real Windows ACL (not a `chmod` comment). Env var still wins.
- **Benchmark holding `InferenceLock` for a whole run → rejected (it's also factually wrong about train).** Copy the train pattern: background thread, `IsBenchmarking` 409 gate on `/generate` and `/chat`, take `InferenceLock` **per case**, with a cancel endpoint.
- **Godot `[Signal]` on AppState mixed with C# events → rejected.** One eventing paradigm. AppState uses plain C# `event Action<T>` (matching `IApiClient`), not Godot signals — avoids the two-eventing-systems cognitive tax the critique flagged.
- **Generic sortable `DataTable` as a Phase-0 prerequisite → deferred.** Build cheap extractions first (`Card`, `Field`, `ProgressRow`, `Badge`, `EmptyState`); build the table only when the Benchmark compare view needs it.

---

## 1. App IA / shell

### 1.1 Nav model

Replace `SwitchMode(bool train)` with a **NavRail + view router keyed by a string id**. The rail is a fixed left column (~56–72px of glyph buttons, or a labeled 236px rail — labeled ships first, icon-only is a later polish). The old `Sidebar` splits into a shared **ConnectionPanel** (app-global, pinned bottom) and per-view **context panels** (Recents belong to Chat only). Content hosts exactly one active view.

```
┌──────────┬──────────────────────────────────────────────────────┐
│ NavRail  │ Header:  <view title>              <view header actions>│
│          ├──────────────────────────────────────────────────────┤
│ 💬 Chat  │                                                        │
│ 🧩 Models│              Active view (router slot)                 │
│ 📊 Bench │                                                        │
│ ───────  │                                                        │
│ ⚙ Settings (opens modal Window, not a routed view)               │
│ ───────  │                                                        │
│ ● Connected  http://localhost:8080   [check]                     │
└──────────┴──────────────────────────────────────────────────────┘
```

Settings is a **modal `Window` (`PopupCentered`)**, reachable from a rail gear — not a routed destination. Rationale: it overlays any view, has no "context panel," and its tabbed form is a poor fit for the router's single-content slot. (Two docs made it a tab, two made it a window; the window wins because it must be reachable *from within* Chat mid-conversation without losing chat state.)

### 1.2 Destinations (the shipped IA)

| id | Rail | Purpose | Context panel | Status |
|----|------|---------|---------------|--------|
| `chat` | 💬 Chat | Converse (existing Transcript+Composer, internals unchanged) | Recents + New chat | Ship (port) |
| `models` | 🧩 Models | List/inspect/launch models; "Train new"; entry to Benchmark | Filter/search | New, quick |
| `bench` | 📊 Benchmark | Define runs, compare outputs, generate docs | Past runs | New (flagship) |
| — | ⚙ Settings | App/model/memory/benchmark config + secrets (modal) | — | New, quick |

**Train folds into Models** as a "＋ Train new model" action opening the existing `TrainPanel` inline/modal. Training is a way to *make* a model, so it belongs with models; `TrainPanel` is re-hosted unchanged.

**Reserved, not built:** `memory` and `upgrade` destinations are defined in the registry as constants but not registered as rail entries until their server backends exist (Memory render endpoints; lineage runtime). Adding them later = one `Register(...)` line.

### 1.3 Cross-tab state — one store, one fan-out

The single source of cross-cutting truth is `AppState`, mutated only through named command methods, each raising one C# event. When `/health` refreshes after any job finishes, `ApplyHealth` fans out to every view at once — a newly trained/benchmarked model appears in the Chat picker, the Models grid, and the Benchmark selector simultaneously. This is the existing `OnHealth → _composer.SetModels(...)` pattern, generalized to multicast. Any subsystem that mutates the model set ends its status handler with `_api.CheckHealth()` — exactly what `OnTrainStatus` already does.

---

## 2. Screen-by-screen inventory

### 2.1 Chat (`ChatView`) — port, don't rewrite
- **Purpose:** converse; the only streaming surface.
- **Components:** `Transcript`, `TurnCard`, `Composer` (model/backend/sampling/research/font), `ChatSocket` (WS). All unchanged internals.
- **State:** owns `_chatBusy`, `_resetSession`, `_sessionModel`, `_sessionBackend` (moved out of `Main` into the view). Reads selected model/backend from `AppState`; adds `memory/user/store` request fields from `AppState.Settings` defaults (single local user — no user picker).
- **Empty state:** placeholder transcript ("Start a conversation"); Composer disabled with reason when `AppState` reports a running train/bench job (reuse the training-gates-generate pattern).

### 2.2 Models (`ModelsView`) — new, quick
- **Purpose:** the launch/inspect hub.
- **Components:** `Card` grid, one per model from enriched `/health`; each shows name, size class, params, ctx, tokenizer kind, step, load state; actions **Chat with** (routes to Chat, preselects), **Benchmark** (routes to Bench, preselects), **Train new** (opens `TrainPanel`), **Tokenize probe**. Context panel: filter/search.
- **Primary states:** on-disk vs loaded (badge); loading spinner during a `/models/load`.
- **Empty state:** `EmptyState("No models", "Train one or point --models at a checkpoint dir", cta="Train new")`.
- **VRAM note:** show per-model *estimated* footprint from params×dtype-bytes. Do **not** claim per-model VRAM — `torch.cuda.memory_allocated` reports the whole process arena, not per-model. A real device-memory line is a later, clearly-labeled "process VRAM" stat.

### 2.3 Benchmark (`BenchmarkView`) — flagship (see §3)
- **Purpose:** define runs, compare, generate docs.
- **Sub-tabs (`TabStrip`):** Define · Compare · Reports.
- **States:** idle → running (ProgressRow, Done/Total, cancel) → done (Compare populated) → error.
- **Empty state:** Define tab with the built-in `baseline` suite preselected; "No runs yet — run the baseline."

### 2.4 Settings (`SettingsWindow`) — modal (see §4)
- Tabs: App · Models · Memory · Benchmark · Secrets. App + Secrets ship first; others fill in.

---

## 3. The Benchmarking suite

The centerpiece. Structural copy of the **training** subsystem (the correct template, precisely because train is *not* under `InferenceLock` for its whole duration). Rigor-first: it ships only metrics that are honest at SmolLM2/CPU scale, and refuses to enshrine noise.

### 3.1 Data model (server, `ProjectAI.Bench`)

Files-as-truth under `<modelsDir>/benchmarks/` (matching `checkpoints/`, `memory/`). Suites and runs are inspectable JSON.

```csharp
public sealed record BenchSuite(string Id, string Label, string? EvalCorpus, BenchCase[] Cases);

public sealed record BenchCase(
    string Id, string Prompt,
    string? Reference = null,        // gold answer (deterministic checks)
    string[]? MustInclude = null,    // substring/keyword checks
    int MaxTokens = 128);            // seed is fixed run-wide (see run config)

public sealed record BenchRunConfig(
    string Suite, string[] Models, string Backend,
    long Seed = 12345,               // FIXED, non-zero. Greedy by default (Sample=false).
    bool Sample = false,             // decoding is fixed across all models in a run
    int Repeats = 3);                // repeat each generation; report MEDIAN

public sealed record CellResult(
    string Model, string CaseId, string Output,
    int PromptTokens, int GeneratedTokens, string Stop,
    double MedianSeconds, double MedianTokPerSec,   // median over Repeats, first run discarded (warmup)
    double? Bpb,                                     // bits-per-byte, tokenizer-invariant quality signal
    CheckResult[] Checks, double CheckPassRate,
    double? JudgeScore,                             // reserved; null until external judge exists
    string? Error);

public sealed record CheckResult(string Kind, string Value, bool Passed);

public sealed record ArmAggregate(
    string Model, int Cases, int N,                 // N = sample count (shown on every rate — see risks)
    double? MeanBpb, double MedianTokPerSec, double CheckPassRate,
    Dictionary<string,double> PassRateByTag);

public sealed record BenchRun(
    string Id, string SuiteId, string Backend, string StartedUtc, string? FinishedUtc,
    BenchRunConfig Config, RunMeta Meta,
    CellResult[] Cells, ArmAggregate[] Aggregates, string State);

public sealed record RunMeta(
    string ProjectAiVersion, string Host, string Os, string BackendUsed,
    ModelStamp[] Models, string[] BackendsAvailable);
public sealed record ModelStamp(string Name, int Step, long ParamCount, string ConfigSummary, string CheckpointSha256);
```

**Metrics captured, and why each is honest:**
1. **bits-per-byte (bpb)** on a held-out corpus — the *primary* quality signal, and the only cross-model-comparable one (tokenizer-invariant). Computed via the shipped `CrossEntropy` forward (no backward), normalized by UTF-8 byte count, not token count. This is the SmolLM2 baseline number.
2. **Throughput (median tok/s)** — from server-side `seconds`, with **explicit warmup (discard first run) + median over `Repeats`**. Not a cold-vs-warm artifact.
3. **Deterministic checks** (contains / exact / regex / jsonValid / maxTokens) on `MustInclude`/`Reference` — objective, run under **greedy** (so checks are reproducible). Substring matches are reported with the caveat that they are floor signals, not quality.
4. **Stop-reason distribution** — surfaces the generation-length asymmetry (instruct stops at EOS, base rambles to cap) so throughput is read in context, not as a naive race.
5. **(reserved) TTFT** — only if the Phase-2b server first-token split lands; otherwise omitted. Never client-derived.
6. **(reserved) Judge score** — nullable, off until an external judge exists.

### 3.2 Harness architecture — `BenchmarkService` (copies TRAIN gating, not InferenceLock-for-run)

```csharp
internal sealed class BenchmarkService   // structural twin of TrainingService
{
    private readonly object _gate = new();
    private BenchJob? _job;
    public BenchJob? Current { get { lock (_gate) return _job; } }
    public bool IsBenchmarking { get { lock (_gate) return _job is { Done: false }; } }

    public (bool ok, string message) Start(BenchmarkService.Deps deps, BenchRunConfig cfg)
    {
        lock (_gate) { if (_job is { Done: false }) return (false, "a benchmark is already running"); }
        // validate suite/models/backend (throws -> 400/409 exactly like Start-training)
        var job = new BenchJob(runId, cfg, total: cfg.Models.Length * caseCount);
        lock (_gate) _job = job;
        Task.Run(() => Run(deps, cfg, job));       // background thread — returns immediately
        return (true, "started");
    }

    private void Run(Deps deps, BenchRunConfig cfg, BenchJob job)
    {
        try {
            foreach (var model in cfg.Models)
              foreach (var c in suite.Cases) {
                if (job.Cancelled) { job.Status = "canceled"; return; }
                // PER-CASE lock (NOT one lock for the whole run) — mirrors /generate's scope.
                var runs = new List<double>();
                for (int r = 0; r <= cfg.Repeats; r++) {          // r==0 is warmup, discarded
                    lock (Server.InferenceLock) {
                        var outp = Inference.RunCase(model, cfg.Backend, c, cfg.Seed, cfg.Sample);
                        if (r > 0) runs.Add(outp.Seconds);
                        if (r == cfg.Repeats) lastOutput = outp;
                    }
                }
                var cell = Score(lastOutput, runs, c);            // median, bpb, checks
                job.Cells.Add(cell); job.Done++;                  // volatile progress
              }
            job.Aggregates = Aggregate(job.Cells);
            WriteRun(deps.BenchDir, job);                          // run.json
            job.Status = "done";
        } catch (Exception e) { job.Status = "error"; job.Error = e.Message; }
    }
}
```

Gating: `/generate` and `/chat` gain an `if (bench.IsBenchmarking) 409` check next to the existing `if (training.IsTraining) 409` at Server.cs:143. A benchmark holds `InferenceLock` **per case**, so `/health` and `/bench/status` stay responsive between cases, but two full generations never run at once. A **cancel** endpoint sets `job.Cancelled`.

### 3.3 Endpoints (mirror `/train`)

```
POST /benchmark               {suite, models[], backend, seed?, sample?, repeats?} -> 202 {runId, total}   (409 if running)
GET  /benchmark/status        -> { bench: { state, runId, done, total, currentModel, error } }
POST /benchmark/cancel        -> { ok }
GET  /benchmark/suites        -> { suites:[{id,label,caseCount}] }
GET  /benchmark/runs          -> { runs:[{id,suiteId,models,backend,startedUtc,done,total,state}] }
GET  /benchmark/run/{id}      -> full BenchRun (cells+aggregates)
POST /benchmark/run/{id}/report {format:"md"|"html"} -> { path, text }
POST /score                   {model, text} -> { tokens, bytes, meanNll, bpb }   (Phase 2b; powers bpb)
```
`GET /health` gains `"bench": { state, done, total, runId }` alongside `training`.

### 3.4 `projectai bench` CLI (second front-end, like train)

```
projectai bench --suite benchmarks/suites/baseline.json \
  --models smollm2-360m-base,smollm2-360m-instruct,smollm2-1.7b-base,smollm2-1.7b-instruct \
  --backend cpu --seed 12345 --repeats 3 --out benchmarks/runs
```
Builds a `BenchRunConfig` and calls the **same `Run` code path** the server uses. Reproducible, CI-scriptable, produces `run.json` + `report.md`. **This is the day-one deliverable and touches zero client code.**

### 3.5 Compare UX

```
Benchmark                                        [ New run ]
[ Define ] [ Compare ] [ Reports ]

Define:  Suite [baseline ▾]  Models ☑360m-base ☑360m-instr ☑1.7b-base ☑1.7b-instr
         Backend [cpu ▾]  Decoding: (•) greedy  Seed[12345]  Repeats[3]   [ Run ▶ ]
         [ProgressRow: 48/96 · 1.7b-base · case 12/24]     [Cancel]

Compare: rows = cases, columns = models (DataTable). Best bpb / pass per row highlighted.
         ┌ Case "Explain gravity" ──────────────────────────────┐
         │ 360m-base │ 360m-instr │ 1.7b-base │ 1.7b-instr        │
         │ <output>  │ <output>   │ <output>  │ <output>          │
         │ bpb 1.42  │ 1.31       │ 1.08 ★    │ 1.02 ★  (n=3)     │
         │ 38 t/s    │ 38 t/s     │ 11 t/s    │ 11 t/s            │
         └───────────────────────────────────────────────────────┘
         Aggregates table: model | bpb↓ | pass% (n) | tok/s | stop-mix
```

Compare is a `DataTable` (rows=cases, cols=models); best-per-row via `Palette.Delta`. Row-activate opens a side-by-side `TurnCard`-style output diff. **Every rate shows `n`**; small-sample tag rates carry a caveat, not a leaderboard gloss.

### 3.6 Generated documentation

`BenchReporter.Render(run, format)` — pure string building (no deps, matching house rules). **Markdown first** (diffable, opens anywhere); HTML with inline-SVG bars later. Charts are emitted **once, in the report** (server), and the client Compare view reuses the same SVG builder — one renderer, not two.

Concrete `report.md` skeleton:

```markdown
# Benchmark: baseline
_Run 2026-07-01T14:22Z · ProjectAI v0.9 · host DESKTOP-… · Windows 11 · backend cpu · greedy · seed 12345 · repeats 3 (median, 1 warmup discarded)_

## Configuration
| Model | sha256 | Params | Ctx | Tokenizer |
|---|---|---|---|---|
| smollm2-1.7b-instruct | a1b2… | 1.7B | 8192 | hf |

## Summary (per model)   — bpb is bits-per-byte (tokenizer-invariant); lower is better
| Model | bpb ↓ | pass% (n) | median tok/s | stop mix (eos/len) |
|---|---|---|---|---|
| smollm2-360m-base     | 1.42 | 33% (n=24) | 38.1 | 4/20 |
| smollm2-1.7b-instruct | **1.02** | 79% (n=24) | 11.4 | 21/3 |

## Pass rate by tag  (n per cell shown; small n = wide CI, treat as directional)
| Tag | 360m-base | 1.7b-instruct |
|---|---|---|
| arithmetic (n=4) | 25% | 75% |

## Per-case detail
### arith-01 — "What is 17 + 26?"  (ref: 43)
| Model | Output | Checks | median tok/s |
|---|---|---|---|
| 360m-instr | "The answer is 42." | contains 43 ✗ | 38.0 |
| 1.7b-instr | "43" | contains 43 ✓ | 11.4 |

_Caveats: throughput is CPU wall-clock, median of 3 (first discarded); bpb is the only cross-model quality signal. No LLM judge was used._
```

### 3.7 Baselining the existing SmolLM2 models

Ship `benchmarks/suites/baseline.json`: a fixed held-out prose+code+dialogue corpus (bpb), ~20–30 instruction/arithmetic/format cases with `MustInclude`/`Reference` checks (greedy), and fixed-`MaxTokens` throughput prompts. Flow: `projectai bench --suite baseline --models <all four> --backend cpu` → `benchmarks/reports/baseline-<date>.md`. That checked-in doc is the reference every future model diffs against. **The CLI produces it before any client UI exists** — the fastest honest win, and it validates the whole pipeline. Note in the doc: base-vs-instruct bpb is a loaded comparison (different training exposure), so read within-family.

---

## 4. Settings architecture

### 4.1 Three tiers, three owners (the spine)

| Tier | Location | Owner | Edited via | Secrets? |
|---|---|---|---|---|
| **Client prefs** | Godot `user://settings.json` | Client | local only | No |
| **App/server config** | `config/settings.json` (outside models dir) | Server | `GET/PUT /config` | No |
| **Secrets** | `config/secrets.json` (git-ignored, ACL-locked) | Server | `PUT/DELETE /config/secrets/{key}` | Yes |
| **Model settings** | sidecar `<name>.settings.json` next to `.ckpt` | Server | `GET/PUT /models/{name}/settings` | No |

Rule: *identical-across-clients, survives-reinstall, or secret → server; per-device convenience → client.*

**Cut from all the settings docs (over-scope for a solo local app):** telemetry settings (nothing consumes them), editable-but-inert `ServerSettings.Port/Host` (they're CLI-only and win over the file — display read-only in About instead), and a three-provider `KnownKeys` list (ship `["tavily"]`; grow when a consumer exists). A **single scrolling settings pane grouped by section** is preferred over a six-tab container where three tabs are near-empty — tabs are added when a domain earns one.

### 4.2 Client prefs (fixes the "lost on restart" papercut — the quick win)

```csharp
public sealed record ClientPrefs(
    string ServerUrl = "http://localhost:8080",
    int FontSize = 14, string DefaultModel = "", string DefaultBackend = "",
    bool Sample = false, float Temperature = 0.8f, int TopK = 40, float TopP = 0.9f,
    int MaxTokens = 0, bool Research = false, string LastView = "chat");
```
Loaded in `Main._Ready` before `CheckHealth`; saved debounced. `ServerUrl` is client-local (it's *how you reach* the server). Composer/Sidebar stop being ephemeral — the view seeds controls from prefs and persists on change (wiring lives in the view, not the component). **~half a day, high value, no server change.**

### 4.3 Server config + the `MemoryPolicy` facade (the cheap memory-settings win)

`MemoryPolicy` (currently `const int`) becomes a thin facade over live settings — every existing call site keeps compiling, values become tunable:

```csharp
internal static class MemoryPolicy {
    public static int BridgeCards  => SettingsStore.Current.Memory.BridgeCards;
    public static int BridgeBudget => SettingsStore.Current.Memory.BridgeBudget;
    public static int RecallHits   => SettingsStore.Current.Memory.RecallHits;
    public static int RecallBudget => SettingsStore.Current.Memory.RecallBudget;
}
```
`GET /config` returns app settings + a **presence-only** secrets block; `PUT /config` validates (budgets ≥ 0, backend id in the `Backends.cs` catalog) and atomic-writes (temp + fsync + rename, like the checkpoint writer).

### 4.4 Secrets handling (the security-critical part)

1. **Never leave the server.** `GET /config` returns `{key, set:true, hint:"…a1b2", source:"env"|"config"}` — never the value.
2. **Set-only from the client.** `PUT /config/secrets/tavily {value}` stores; `DELETE` clears. The UI field is write-only; on save it clears immediately and re-fetches masked status.
3. **Separate, locked, git-ignored file** at `config/secrets.json` — **outside the models tree** (so it's never one `git add` from a public repo). Lockdown is a **real Windows ACL** (`FileSecurity`/`FileSystemAclExtensions`: strip inheritance, grant current SID only) — not a POSIX `chmod` comment. Add the `.gitignore` line.
4. **Env var wins.** `SecretStore.Resolve("tavily")` checks config then falls back to `TAVILY_API_KEY`; `source` reports which. `TavilySearchProvider` changes one line to call `Resolve`. The provider's singleton lifetime is reworked to read from `SecretStore` (this is a real lifetime change, **not** "one file behind the interface" — budget it honestly).
5. **No secret in any log line**; error messages truncate to the `"…"+last4` hint.

**Single-user only:** there is no auth on the local `HttpListener`. The client ships **no user picker**; `user/store` default to `"default"`/model-name. Multi-user memory partitioning is not exposed until an auth principal exists — surfacing a user switcher over an unauthenticated `Resolve(user)` would be a security regression.

### 4.5 Model settings (sidecar, second wave)

Sidecar `<name>.settings.json` holds editable per-model defaults (display name, system prompt, decoding defaults, bound store); architecture stays immutable in checkpoint `Meta`. Split by mutability — a settings tweak must never rewrite weights. Absent sidecar → fall back to app defaults. `/generate` resolves these as request defaults (request field wins → sidecar → app default).

---

## 5. Client architecture to scale

### 5.1 Layers

```
Main (composition root — wiring only, ~40 lines)
 ├ AppState  (shared store; C# events; command methods)
 ├ ApiClient : IApiClient   (REST control plane; HttpRequest pool)
 ├ ChatSocket : IChatTransport (WS streaming; unchanged)
 └ ViewRouter + IView  (register N destinations; show one)
        └ ChatView · ModelsView · BenchmarkView   (+ SettingsWindow modal)
             └ Palette + Ui/Design component library
```
**One rule (client mirror of the runtime's inward-dependency rule):** views depend on `AppState` + the seams + the component library. Views never reference each other; cross-view effects flow only through `AppState`.

### 5.2 AppState (C# events, not Godot signals)

```csharp
public sealed class AppState
{
    public HealthSnapshot Health { get; private set; } = HealthSnapshot.Empty;
    public string SelectedModel { get; private set; } = "";
    public string SelectedBackend { get; private set; } = "";
    public ClientPrefs Prefs { get; private set; } = new();
    public JobStatus Train { get; private set; } = JobStatus.Idle;
    public JobStatus Bench { get; private set; } = JobStatus.Idle;
    public ConnState Connection { get; private set; } = ConnState.Unknown;

    public event Action? HealthChanged;      // fan-out to ALL views
    public event Action? SelectionChanged;
    public event Action? PrefsChanged;
    public event Action? JobsChanged;
    public event Action? ConnectionChanged;

    public void ApplyHealth(HealthResult h) { /* set catalog+defaults; HealthChanged?.Invoke() */ }
    public void SelectModel(string m) { SelectedModel = m; SelectionChanged?.Invoke(); }
    public void ApplyPrefs(ClientPrefs p) { Prefs = p; PrefsChanged?.Invoke(); }
    public void SetBench(JobStatus s) { Bench = s; JobsChanged?.Invoke(); }
    // ... one mutator per field, each raises exactly one event
}
```
Plain C# events (not `[Signal]`) so the whole client uses one eventing model. `AppState` is a plain object passed by constructor to each view, not a Godot autoload global — keeps the anti-god-object stance and the DI-by-constructor pattern consistent.

### 5.3 Typed ApiClient (one method per endpoint) + the pool fix

`IApiClient` widens to the full control plane (health, tokenize, train, benchmark, config, model-settings). Same "one method per endpoint, `Ok`-flag result, fires on main thread" shape.

**Concurrency fix (the one real transport change):** today `ApiClient` serializes on a single `HttpRequest` (`if (Busy) return`). Multiple pollers + user actions will collide. Give `ApiClient` a small **pool of `HttpRequest` child nodes** with per-node continuation state and out-of-order completion. Policy: **user actions queue; status polls drop-if-busy** (dropping a redundant poll is correct — the existing single-request drop behavior is actually right for polling and must be preserved for it). This is localized behind the interface; no view sees it.

**Records are client-only DTOs** in `Api/*Contracts.cs`, parsed from JSON via the existing `JsonDict` walker — **never** shared with the net10 server types. Nested graphs (`BenchRun.Cells`, aggregates) need hand-written dictionary walkers in the `JsonDict` style; this is real work, budgeted in Phase 2c, not "reads a BenchRun."

### 5.4 Per-tab structure & testability
Each view is a self-contained `Control` implementing `IView` (`Root`, `Title`, `OnShown`, `OnHidden`). `OnShown` refreshes from `AppState` and starts polling; `OnHidden` stops it (reuses today's `_pollTimer` + `!_api.Busy` gate). A `FakeApiClient : IApiClient` raising canned results drives any view headless — the reason the seam exists.

---

## 6. Look & feel / design language

Extend `Palette` (single style source), split into `Ui/Design/` so it doesn't bloat. Aesthetic unchanged: dark, terracotta accent.

**Token scales (add to Palette):**
```csharp
public static class Type   { public const int Caption=12, Label=13, Body=14, H3=16, H2=18, H1=22, Mono=13; }
public static class Space  { public const int Xs=4, Sm=8, Md=12, Lg=16, Xl=24, Xxl=32; }
public static class Radius { public const int Sm=8, Md=12, Lg=14, Pill=999; }
```

**Semantic colors + factories (drawn from existing Good/Bad/Accent/Muted):**
```csharp
public static Button   RailButton(string glyph, bool active);
public static PanelContainer Card(string? title = null, int pad = Space.Lg);
public static Label    Badge(string text, Color tone);        // pill
public static Label    StatusBadge(string state);             // idle/running/done/error/available
public static Label    TierBadge(string size);                // tiny/small/medium/large
public static Color    Delta(double value, bool higherIsBetter);  // Good/Bad/Muted
public static Control  ProgressRow(string label);             // extracted from TrainPanel
public static VBoxContainer EmptyState(string glyph, string title, string hint, Button? cta = null);
public static HBoxContainer Field(string label, Control control, int labelWidth = 120);
```

**Build cheap first:** `Card`, `Badge`, `Field`, `EmptyState`, `ProgressRow`, `SectionHeader`, `RailButton` are extractions/wrappers of existing code (~a day). `DataTable` (sortable, row-select) is the one real build (~1–2 days) — build it when the Benchmark Compare view needs it, not before. Two data-viz series colors reserved for report charts so charts never invent colors.

**Theme:** defer runtime theme-switching. When it lands, do it as a Godot `Theme` resource + `NOTIFICATION_THEME_CHANGED` (idiomatic), **not** a bespoke `Palette.Changed` event bus every component must subscribe to. Dark-only ships fine today.

---

## 7. New server endpoints + records

**Config / secrets** (`ProjectAI/Settings/`):
```
GET    /config                     -> AppSettings + secrets[]{key,set,hint,source}   (no values)
PUT    /config                      <- partial AppSettings                            (200 | 400 {problems[]})
PUT    /config/secrets/{key}        <- {value}    -> masked status
DELETE /config/secrets/{key}        -> masked status
GET    /models/{name}/settings      -> ModelSettings (or synthesized defaults)
PUT    /models/{name}/settings      <- ModelSettings
```
New: `AppSettings`, `MemorySettings`, `ModelSettings`, `DecodingDefaults`, `SettingsStore`, `SecretStore`; edit `MemoryPolicy.cs` (→ facade), `TavilySearchProvider.cs` (→ `SecretStore.Resolve`).

**Benchmark** (`ProjectAI/BenchmarkService.cs`, `ProjectAI.Bench/`):
```
POST /benchmark ; GET /benchmark/status ; POST /benchmark/cancel
GET  /benchmark/suites ; GET /benchmark/runs ; GET /benchmark/run/{id}
POST /benchmark/run/{id}/report ; POST /score
```
`/health` gains a `bench` block; `/generate` + `/chat` gain an `IsBenchmarking` 409 gate.

**Enriched catalog:** `/health.models` grows from `string[]` to `modelInfos[]{name,size,params,layers,ctx,instruct,tokenizerKind,step}` (keep `models: string[]` for back-compat; client prefers the rich array). No lineage/adapter fields until that runtime exists.

**Reused unchanged:** `ProjectAI.Training/Inference` (+ optional TTFT split in 2b), `ModelRegistry`, `ComputeRegistry`, `InferenceLock`, `ModelPresets`, `TrainingService` (the template), `Converter`, `ChatSocket`/`ChatSession`.

---

## 8. Phased roadmap (quick win → growth)

| Phase | Deliverable | New server work | Effort |
|---|---|---|---|
| **0 — Shell refactor** | Extract `AppState`, `ViewRouter`/`IView`, `NavRail`, `ConnectionPanel`; wrap Chat as `ChatView`, host `TrainPanel` in `ModelsView`; multicast `HealthChanged`; add Palette token scales + cheap components. **App behaves identically on a scalable skeleton.** Rename command-palette `Palette` collision. *(As shipped: the `ApiClient` HttpRequest pool + per-path poll dedupe was deliberately pulled forward from 2c — the old single-slot drop-while-busy client could wedge TrainPanel: `SetBusy(true)` followed by a silently dropped `StartTraining`. `IApiClient` consequently lost its `Busy` member; a `/health` sequence guard keeps racing checks latest-wins.)* | none | ~1–2 d. Highest leverage; do first, ship, live in it. |
| **1a — Client prefs** | `ClientPrefs` + `user://settings.json`; persist ServerUrl/model/backend/sampling/font. | none | ~½ d |
| **1b — Models view** | Card grid over enriched `/health`; Chat-with / Benchmark / Train-new actions; estimated footprint badges. | `/health` `modelInfos` | ~1 d |
| **1c — Bench CLI + baseline** | `ProjectAI.Bench` records, `CheckEvaluator`, `BenchReporter` (md), `/score` bpb, `projectai bench` CLI, `baseline.json`; **generate the SmolLM2 baseline doc**. No client. | `/score`, CLI | ~2–3 d |
| **2a — Settings App+Secrets** | `SettingsStore` (Defaults+Memory only) + `GET/PUT /config`; `MemoryPolicy` facade; `SecretStore` (tavily, real Windows ACL); Settings window App+Secrets sections. | `/config*` | ~2 d |
| **2b — Bench server + TTFT** | `BenchmarkService` (train-pattern) + endpoints + cancel + `bench` block; optional server first-token split for TTFT. | `/benchmark*` | ~2–3 d |
| **2c — Bench client** | `BenchmarkView` (Define/Compare/Reports), `DataTable`, `IApiClient` methods + hand-written JSON walkers, poll reuse. *(HttpRequest pool landed early, in Phase 0.)* | none | ~2–3 d |
| **3 — Growth** | Per-model sidecar settings + Models settings tab; HTML+SVG reports; enriched Memory read-only preview (needs `RenderBridge`/`RenderRecall` endpoints); then Memory-curation and Upgrade *only when their runtimes exist*. | render endpoints; (later) lineage | as scoped |

**Ship-and-stop line:** Phases 0 + 1a + 1b + 1c give a scalable shell, persisted settings, a Models hub, and an honest baseline report — a genuine one-stop-shop increment. Everything past is additive.

---

## 9. Risks & mitigations

- **Benchmark rigor (highest risk).** Misleading numbers are worse than none. Mitigations, all mandatory in v1: **greedy + fixed non-zero seed**, decoding held constant across models; **warmup discard + median-of-N** throughput; **bpb not raw PPL** (tokenizer-invariant); **`n` printed on every rate** with a small-sample caveat; **stop-reason mix shown** so throughput isn't read as a naive race; **no LLM judge** until an external model exists; **TTFT only if the server first-token split lands**, never client-derived. Report copy explicitly labels CPU wall-clock and the base-vs-instruct caveat.
- **Secrets.** Separate git-ignored `config/secrets.json` outside the models tree; **real Windows ACL** (not `chmod`); set-only/masked/env-fallback; no secret in logs. The local server has no auth → single-user only, no user picker, `--port` bound to localhost. The `TavilySearchProvider` lifetime rework is real work, budgeted.
- **Scope.** Four live destinations, not six; Memory/Upgrade reserved behind their (unbuilt) runtimes; no telemetry, no inert infra knobs, no cross-provider secret keys, no LLM judge. Speculative depth (LoRA-rank pickers, chip slots) is out until the lineage runtime is a real project.
- **net8/net10 split.** Client re-declares DTOs; JSON-only wire; nested bench results need hand-written `JsonDict` walkers (budgeted in 2c). No server-project reference from the client, ever.
- **InferenceLock contention.** Bench (per-case lock) + `IsBenchmarking` 409 gate = one job at a time, chat cleanly blocked/blocking, `/health` responsive between cases. A cancel endpoint prevents a long matrix from wedging you out. A job queue is a later concern.
- **`Main.cs` refactor regression.** Riskiest step because everything routes through it. Keep `ChatSocket` byte-identical; migrate Chat first and verify against today's behavior before adding destinations. *(`IApiClient` is the one deliberate exception: it lost `Busy` when the pool + queue replaced drop-while-busy — pulled forward from 2c because the old client could wedge TrainPanel on a silently dropped Train click.)*

Key files: **client new** — `State/AppState.cs`, `Nav/{ViewRouter,IView}.cs`, `Ui/{NavRail,ConnectionPanel,AppShell,SettingsWindow}.cs`, `Ui/Design/{Components,DataTable}.cs`, `Views/{Chat,Models,Benchmark}View.cs`, `Prefs.cs`, `Api/{BenchContracts,ConfigContracts}.cs`; **client edit** — `Main.cs`, `Ui/Palette.cs`, `Api/ApiClient.cs`+`IApiClient`, `Ui/Composer.cs`/`TrainPanel.cs` (re-host, seed from prefs). **Server new** — `ProjectAI.Bench/{BenchSuite,BenchRunConfig,BenchResult,CheckEvaluator,BenchReporter}.cs`, `ProjectAI/BenchmarkService.cs`, `ProjectAI/Settings/{AppSettings,ModelSettings,SettingsStore,SecretStore}.cs`, `benchmarks/suites/baseline.json`, `config/secrets.json` (git-ignored); **server edit** — `Server.cs` (routes, `bench` block, `IsBenchmarking` gate, enriched `/health`), `MemoryPolicy.cs` (facade), `TavilySearchProvider.cs` (`SecretStore.Resolve`), `ProjectAI.Training/Inference.cs` (optional TTFT split).