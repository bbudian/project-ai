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

    /// <summary>Validates the head/dim relationships GQA attention relies on. Throws on an impossible config.</summary>
    public void Validate()
    {
        if (HeadCount < 1) throw new ArgumentException($"HeadCount must be ≥ 1; got {HeadCount}.");
        if (KvHeadCount < 1 || KvHeadCount > HeadCount)
            throw new ArgumentException($"KvHeadCount ({KvHeadCount}) must be in [1, HeadCount={HeadCount}].");
        if (HeadCount % KvHeadCount != 0)
            throw new ArgumentException($"HeadCount ({HeadCount}) must be divisible by KvHeadCount ({KvHeadCount}).");
        if (EmbeddingDim % HeadCount != 0)
            throw new ArgumentException($"EmbeddingDim ({EmbeddingDim}) must be divisible by HeadCount ({HeadCount}).");
        if (HeadDim % 2 != 0)
            throw new ArgumentException($"HeadDim ({HeadDim}) must be even for RoPE.");
    }
}
