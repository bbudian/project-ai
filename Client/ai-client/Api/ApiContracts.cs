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
    event Action<HealthResult> HealthReceived;
    event Action<TrainStartResult> TrainStarted;
    event Action<TrainStatus> TrainStatusReceived;
}

/// <summary>One generation request: the prompt, the chosen model and compute backend (empty = server default), decoding settings, and whether to ground the answer in a live web search (RAG). Carried over the chat WebSocket (the old REST /generate result type was retired in favor of streaming).</summary>
public sealed record GenerateRequest(string Prompt, string Model, string Backend, bool Sample, float Temperature, int TopK, float TopP, int MaxTokens, bool Research);

/// <summary>A web source the model was grounded in (web-research mode): a title and its URL, shown as a citation under the reply.</summary>
public sealed record SourceLink(string Title, string Url);

/// <summary>A compute backend the server offers (the CPU/GPU picker). <see cref="Available"/> is false when its runtime/device isn't present, with the reason in <see cref="Reason"/>.</summary>
public sealed record BackendOption(string Id, string Label, bool Available, string Reason);

/// <summary>A training size preset the server offers (the Train-tab size picker): an id sent with the request and a human-readable label.</summary>
public sealed record SizeOption(string Id, string Label);

/// <summary>Result of a /health call: the available models, backends, training-size presets, and the server's defaults. On failure, only <see cref="Error"/> is meaningful.</summary>
public sealed record HealthResult(bool Ok, string[] Models, string Default, BackendOption[] Backends, string DefaultBackend, SizeOption[] Sizes, string Error);

/// <summary>A request to train a new model on the server: a name, the training text, a size preset, step count, and compute backend.</summary>
public sealed record TrainRequest(string Name, string Text, string Size, int Steps, string Backend);

/// <summary>Result of POST /train (starting a job). On failure, <see cref="Error"/> is set.</summary>
public sealed record TrainStartResult(bool Ok, string Error);

/// <summary>Live training progress from /train/status. <see cref="State"/> is "idle" | "running" | "done" | "error".</summary>
public sealed record TrainStatus(string State, string Name, int Step, int TotalSteps, float Loss, string Error);
