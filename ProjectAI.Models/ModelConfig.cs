namespace ProjectAI.Models;

/// <summary>
/// Hyperparameters for a Llama-style decoder-only transformer. Encodes the researched architecture:
/// RoPE (<see cref="RopeTheta"/>), grouped-query attention (<see cref="KvHeadCount"/> &lt; <see cref="HeadCount"/>),
/// RMSNorm (<see cref="NormEpsilon"/>), and a SwiGLU FFN (<see cref="FeedForwardHiddenDim"/>).
/// </summary>
public sealed record ModelConfig
{
    public required int VocabSize { get; init; }
    public required int EmbeddingDim { get; init; }
    public required int LayerCount { get; init; }
    public required int HeadCount { get; init; }
    public required int KvHeadCount { get; init; }
    public required int FeedForwardHiddenDim { get; init; }
    public int MaxSequenceLength { get; init; } = 4096;
    public float RopeTheta { get; init; } = 10_000f;
    public float NormEpsilon { get; init; } = 1e-5f;

    public int HeadDim => EmbeddingDim / HeadCount;
}
