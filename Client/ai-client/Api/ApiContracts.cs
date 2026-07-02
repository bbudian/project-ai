using System;

// Transport-agnostic contract for the ProjectAI server's HTTP control plane (health + training). The UI depends only
// on this interface (dependency inversion), so the implementation can be swapped — e.g. a fake for tests — without
// touching any view code. Results always come back as Ok/!Ok so the UI has one uniform success path. (Chat itself
// streams over a separate WebSocket transport, ChatSocket — it is not part of this request/response interface.)
public interface IApiClient
{
    string BaseUrl { get; set; }
    void CheckHealth();
    void StartTraining(TrainRequest request);
    void CheckTrainStatus();
    void Tokenize(string model, string text);
    void MemoryList(string store, string query);
    void MemoryRender(string store, string query);
    void MemoryPut(string store, string title, string[] keys, string body, string tier, string trust);
    void StartBenchmark(BenchStartRequest request);
    void CheckBenchStatus();
    void CancelBenchmark();
    void FetchBenchSuites();
    void FetchBenchRuns();
    void FetchBenchRun(string id);
    void FetchConfig();
    void SaveMemoryBudgets(int bridgeCards, int bridgeBudget, int recallHits, int recallBudget);
    void SaveSecret(string key, string value);
    void ClearSecret(string key);
    event Action<HealthResult> HealthReceived;
    event Action<TrainStartResult> TrainStarted;
    event Action<TrainStatus> TrainStatusReceived;
    event Action<TokenizeResult> TokenizeReceived;
    event Action<MemoryListResult> MemoryListReceived;
    event Action<MemoryRenderResult> MemoryRenderReceived;
    event Action<MemorySaveResult> MemorySaved;
    event Action<BenchStartResult> BenchStarted;
    event Action<BenchStatusInfo> BenchStatusReceived;
    event Action<BenchSuiteInfo[]> BenchSuitesReceived;
    event Action<BenchRunSummary[]> BenchRunsReceived;
    event Action<BenchRunDetail> BenchRunReceived;
    event Action<ConfigInfo> ConfigReceived;
    event Action<SecretStatus> SecretUpdated;
}

/// <summary>A secret's presence on the server (never its value): set + a last-4 hint + where it came from ("env" | "config").</summary>
public sealed record SecretStatus(bool Ok, string Key, bool Set, string Hint, string Source, string Error);

/// <summary>GET /config: the server-side memory injection budgets plus secret presence.</summary>
public sealed record ConfigInfo(
    bool Ok, int BridgeCards, int BridgeBudget, int RecallHits, int RecallBudget, SecretStatus[] Secrets, string Error);

/// <summary>One generation request: the prompt, the chosen model and compute backend (empty = server default), decoding settings, whether to ground the answer in a live web search (RAG), and whether to attach the server-side memory store (session-scoped — honored on the chat start frame). Carried over the chat WebSocket (the old REST /generate result type was retired in favor of streaming).</summary>
public sealed record GenerateRequest(string Prompt, string Model, string Backend, bool Sample, float Temperature, int TopK, float TopP, int MaxTokens, bool Research, bool Memory, ulong Seed);

/// <summary>The server's "ready" frame after a chat session starts: what it actually loaded, whether the model is instruct-tuned (ChatML detected), and the context window size.</summary>
public sealed record SessionInfo(string Model, string Backend, bool Instruct, int ContextLimit);

/// <summary>The server's "done" frame at the end of a turn: stop reason plus token/timing/context accounting. The short form (e.g. a research turn canceled mid-search) carries only <see cref="Stop"/> — numeric fields default to 0.</summary>
public sealed record TurnStats(string Stop, int PromptTokens, int GeneratedTokens, float Seconds, int Position, int ContextLimit);

/// <summary>A web source the model was grounded in (web-research mode): a title and its URL, shown as a citation under the reply.</summary>
public sealed record SourceLink(string Title, string Url);

/// <summary>A compute backend the server offers (the CPU/GPU picker). <see cref="Available"/> is false when its runtime/device isn't present, with the reason in <see cref="Reason"/>.</summary>
public sealed record BackendOption(string Id, string Label, bool Available, string Reason);

/// <summary>A training size preset the server offers (the Train-tab size picker): an id sent with the request and a human-readable label.</summary>
public sealed record SizeOption(string Id, string Label);

/// <summary>One entry of the server's enriched model catalog (/health <c>modelInfos</c>). <see cref="Error"/> is set (and the numbers zero) when the server couldn't read that checkpoint's metadata; a name-only fallback is synthesized when talking to an older server without <c>modelInfos</c>.</summary>
public sealed record ModelInfo(string Name, long Params, int Layers, int Ctx, int Vocab, string Tokenizer, string Dtype, int Step, bool Instruct, long FileBytes, string Error);

/// <summary>Result of a /health call: the available models, backends, training-size presets, and the server's defaults. On failure, only <see cref="Error"/> is meaningful.</summary>
public sealed record HealthResult(bool Ok, string[] Models, ModelInfo[] Infos, string Default, BackendOption[] Backends, string DefaultBackend, SizeOption[] Sizes, string Error);

/// <summary>Result of a /tokenize probe: how the named model's tokenizer splits a text.</summary>
public sealed record TokenizeResult(bool Ok, string Model, int Count, string[] Pieces, string Error);

/// <summary>One card of the memory catalog (GET /memory): identity + ranking frontmatter, never the body.</summary>
public sealed record MemoryCardInfo(string Id, string Title, string[] Keys, string Tier, string Trust, string AsOf);

/// <summary>The memory catalog for a store. <see cref="Count"/> is the store's total; <see cref="Memories"/> is the (possibly query-filtered) listing.</summary>
public sealed record MemoryListResult(bool Ok, string Store, int Count, MemoryCardInfo[] Memories, string Error);

/// <summary>What would be injected for a message (GET /memory/render): the pinned bridge and the recall block for the query.</summary>
public sealed record MemoryRenderResult(bool Ok, string Bridge, string Recall, string Error);

/// <summary>Result of a manual memory inject (PUT /memory).</summary>
public sealed record MemorySaveResult(bool Ok, string Id, string Error);

/// <summary>A benchmark run request (POST /benchmark). Decoding is greedy with a fixed seed — the v1 rigor rule.</summary>
public sealed record BenchStartRequest(string Suite, string[] Models, string Backend, int Repeats);

/// <summary>202 from POST /benchmark: the run id to follow, and how many (model, case) cells it will produce.</summary>
public sealed record BenchStartResult(bool Ok, string RunId, int Total, string Error);

/// <summary>Live progress from GET /benchmark/status. State is idle | running | done | canceled | error.</summary>
public sealed record BenchStatusInfo(
    string State, string RunId, string Suite, int Done, int Total, string CurrentModel, string CurrentCase, string Error);

/// <summary>One suite the server offers (GET /benchmark/suites).</summary>
public sealed record BenchSuiteInfo(string Id, string Label, int CaseCount, bool HasCorpus);

/// <summary>One past run in the Reports list (GET /benchmark/runs).</summary>
public sealed record BenchRunSummary(string Id, string SuiteId, string[] Models, string Backend, string StartedUtc, string State, int Cases);

/// <summary>Per-model aggregates of a finished run. Bpb is NaN when the suite had no eval corpus.</summary>
public sealed record BenchAggregateInfo(string Model, int N, double Bpb, double MedianTokPerSec, double CheckPassRate);

/// <summary>One (model, case) cell of a finished run — enough for the Compare grid and the output diff.</summary>
public sealed record BenchCellInfo(
    string Model, string CaseId, string Output, int GeneratedTokens, string Stop, double MedianTokPerSec,
    double CheckPassRate, string Error);

/// <summary>A full run (GET /benchmark/run/{id}), reduced to what the Compare view renders.</summary>
public sealed record BenchRunDetail(
    bool Ok, string Id, string SuiteId, string State, BenchAggregateInfo[] Aggregates, BenchCellInfo[] Cells, string Error);

/// <summary>A request to train a new model on the server: a name, the training text, a size preset, step count, and compute backend.</summary>
public sealed record TrainRequest(string Name, string Text, string Size, int Steps, string Backend);

/// <summary>Result of POST /train (starting a job). On failure, <see cref="Error"/> is set.</summary>
public sealed record TrainStartResult(bool Ok, string Error);

/// <summary>Live training progress from /train/status. <see cref="State"/> is "idle" | "running" | "done" | "error".</summary>
public sealed record TrainStatus(string State, string Name, int Step, int TotalSteps, float Loss, string Error);
