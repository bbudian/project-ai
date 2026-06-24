using System;

// Transport-agnostic contract for the ProjectAI server. The UI depends only on this interface (dependency
// inversion), so the HTTP implementation can be swapped — e.g. a fake for tests, or a future streaming client —
// without touching any view code. Results always come back as Ok/!Ok so the UI has one uniform success path.
public interface IApiClient
{
    string BaseUrl { get; set; }
    bool Busy { get; }
    void CheckHealth();
    void Generate(GenerateRequest request);
    event Action<HealthResult> HealthReceived;
    event Action<GenerateResult> GenerationReceived;
}

/// <summary>One generation request: the prompt, the chosen model (empty = server default), and decoding settings.</summary>
public sealed record GenerateRequest(string Prompt, string Model, bool Sample, float Temperature, int TopK, float TopP, int MaxTokens);

/// <summary>Result of a /generate call. On failure, <see cref="Error"/> is set and <see cref="Text"/> is null.</summary>
public sealed record GenerateResult(bool Ok, string Text, string Error);

/// <summary>Result of a /health call: the available models and the server's default. On failure, only <see cref="Error"/> is meaningful.</summary>
public sealed record HealthResult(bool Ok, string[] Models, string Default, string Error);
