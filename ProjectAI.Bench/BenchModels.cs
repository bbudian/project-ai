namespace ProjectAI.Bench;

// The benchmark data model (docs/CLIENT_DESIGN.md §3.2) — files-as-truth: a run serializes to
// <modelsDir>/benchmarks/runs/<id>.json and the report renders from the same records. Metric rules are baked into
// the shapes: bpb is the primary cross-model quality signal (tokenizer-invariant), throughput is a median with a
// warmup discarded, checks run under greedy decoding so they reproduce, and JudgeScore stays a nullable reserved
// slot (no LLM-judge in v1).

/// <summary>A named set of cases plus an optional held-out corpus (for bpb) — loaded from a suite JSON file.</summary>
public sealed record BenchSuite(string Id, string Label, string? EvalCorpus, IReadOnlyList<BenchCase> Cases);

/// <summary>One prompt to run against every model. <see cref="Reference"/> feeds the exact/regex/jsonValid checks;
/// <see cref="MustInclude"/> feeds the contains checks (floor signals, labeled as such in the report).</summary>
public sealed record BenchCase(
    string Id, string Prompt, string? Reference = null, IReadOnlyList<string>? MustInclude = null, int MaxTokens = 128);

/// <summary>The run configuration: decoding is held constant across every model in the run (greedy by default,
/// fixed non-zero seed) so differences are the models', not the dice's.</summary>
public sealed record BenchRunConfig(
    string Suite, IReadOnlyList<string> Models, string Backend,
    ulong Seed = 12345, bool Sample = false, int Repeats = 3);

/// <summary>One deterministic check outcome. Kind is contains/exact/regex/jsonValid/maxTokens.</summary>
public sealed record CheckResult(string Kind, string Value, bool Passed);

/// <summary>One (model, case) cell: the output, its stop reason, timing medians (over Repeats, one extra warmup
/// generation discarded), and the check outcomes. <see cref="Error"/> is set if this cell failed to run.</summary>
public sealed record CellResult(
    string Model, string CaseId, string Output, int PromptTokens, int GeneratedTokens, string Stop,
    double MedianSeconds, double MedianTokPerSec, IReadOnlyList<CheckResult> Checks, double CheckPassRate,
    double? JudgeScore = null, string? Error = null);

/// <summary>Per-model aggregates over the whole suite. <see cref="MeanBpb"/> is null when the suite has no eval
/// corpus. N is printed with every rate (small-sample caveat).</summary>
public sealed record ArmAggregate(
    string Model, int Cases, int N, double? MeanBpb, double MedianTokPerSec, double CheckPassRate,
    IReadOnlyDictionary<string, int> StopReasons);

/// <summary>Identity of a benchmarked checkpoint, pinned so a report can never be mistaken for another build's.</summary>
public sealed record ModelStamp(string Name, int Step, long ParamCount, string ConfigSummary, string CheckpointSha256);

/// <summary>Environment the run happened in — half of what makes a number meaningful.</summary>
public sealed record RunMeta(
    string ProjectAiVersion, string Host, string Os, string BackendUsed, IReadOnlyList<ModelStamp> Models);

/// <summary>A whole benchmark run. State is running | done | canceled | error.</summary>
public sealed record BenchRun(
    string Id, string SuiteId, string Backend, string StartedUtc, string? FinishedUtc,
    BenchRunConfig Config, RunMeta Meta, IReadOnlyList<CellResult> Cells, IReadOnlyList<ArmAggregate> Aggregates,
    string State);
