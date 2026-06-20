using ProjectAI.Core;

namespace ProjectAI.Models;

/// <summary>Selects the next token id from a logits tensor.</summary>
public interface ISampler
{
    int Sample(Tensor logits);
}

/// <summary>Deterministic argmax decoding.</summary>
public sealed class GreedySampler : ISampler
{
    public int Sample(Tensor logits) => throw new NotImplementedException("argmax — ticket S1-9.");
}

/// <summary>Temperature + top-k + top-p (nucleus) sampling.</summary>
public sealed class TopKTopPSampler(float temperature = 1.0f, int topK = 0, float topP = 1.0f) : ISampler
{
    public float Temperature { get; } = temperature;
    public int TopK { get; } = topK;
    public float TopP { get; } = topP;
    public int Sample(Tensor logits) => throw new NotImplementedException("ticket S1-9.");
}
