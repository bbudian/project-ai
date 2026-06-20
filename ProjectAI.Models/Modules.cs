using ProjectAI.Core;

namespace ProjectAI.Models;

/// <summary>Root-mean-square layer norm (no mean subtraction, single learned scale).</summary>
public sealed class RmsNorm(int dim, float epsilon) : Module
{
    public int Dim { get; } = dim;
    public float Epsilon { get; } = epsilon;
    public override Tensor Forward(Tensor input) => throw new NotImplementedException("ticket S1-6.");
}

/// <summary>Rotary position embedding applied to query/key projections.</summary>
public sealed class RotaryEmbedding(int headDim, float theta, int maxSequenceLength) : Module
{
    public int HeadDim { get; } = headDim;
    public float Theta { get; } = theta;
    public int MaxSequenceLength { get; } = maxSequenceLength;
    public override Tensor Forward(Tensor input) => throw new NotImplementedException("ticket S1-6.");
}

/// <summary>Grouped-query attention with a KV cache for incremental decoding.</summary>
public sealed class Attention(ModelConfig config) : Module
{
    public ModelConfig Config { get; } = config;
    public override Tensor Forward(Tensor input) => throw new NotImplementedException("ticket S1-7.");
}

/// <summary>SwiGLU feed-forward network: down(silu(gate(x)) * up(x)).</summary>
public sealed class SwiGluFeedForward(int dim, int hiddenDim) : Module
{
    public int Dim { get; } = dim;
    public int HiddenDim { get; } = hiddenDim;
    public override Tensor Forward(Tensor input) => throw new NotImplementedException("ticket S1-6.");
}

/// <summary>One pre-norm transformer block: x + attn(norm(x)), then x + ffn(norm(x)).</summary>
public sealed class TransformerBlock : Module
{
    public TransformerBlock(ModelConfig config)
    {
        Config = config;
        AttentionNorm = new RmsNorm(config.EmbeddingDim, config.NormEpsilon);
        Attention = new Attention(config);
        FeedForwardNorm = new RmsNorm(config.EmbeddingDim, config.NormEpsilon);
        FeedForward = new SwiGluFeedForward(config.EmbeddingDim, config.FeedForwardHiddenDim);

        RegisterModule(nameof(AttentionNorm), AttentionNorm);
        RegisterModule(nameof(Attention), Attention);
        RegisterModule(nameof(FeedForwardNorm), FeedForwardNorm);
        RegisterModule(nameof(FeedForward), FeedForward);
    }

    public ModelConfig Config { get; }
    public RmsNorm AttentionNorm { get; }
    public Attention Attention { get; }
    public RmsNorm FeedForwardNorm { get; }
    public SwiGluFeedForward FeedForward { get; }

    public override Tensor Forward(Tensor input) => throw new NotImplementedException("ticket S1-8.");
}

/// <summary>Full model: token embedding -&gt; N transformer blocks -&gt; final norm -&gt; LM head.</summary>
public sealed class LlamaModel : Module
{
    private readonly List<TransformerBlock> _blocks = [];

    public LlamaModel(ModelConfig config)
    {
        Config = config;
        FinalNorm = new RmsNorm(config.EmbeddingDim, config.NormEpsilon);
        RegisterModule(nameof(FinalNorm), FinalNorm);

        for (var i = 0; i < config.LayerCount; i++)
        {
            var block = new TransformerBlock(config);
            _blocks.Add(block);
            RegisterModule($"block.{i}", block);
        }
    }

    public ModelConfig Config { get; }
    public RmsNorm FinalNorm { get; }
    public IReadOnlyList<TransformerBlock> Blocks => _blocks;

    public override Tensor Forward(Tensor input) => throw new NotImplementedException("ticket S1-8.");
}

/// <summary>
/// Per-request key/value cache for incremental decoding. A PagedAttention-style block table is
/// planned for Stage 3; this is the contract the attention module reads/writes.
/// </summary>
public sealed class KvCache(ModelConfig config, int maxBatch, int maxSequenceLength)
{
    public ModelConfig Config { get; } = config;
    public int MaxBatch { get; } = maxBatch;
    public int MaxSequenceLength { get; } = maxSequenceLength;
}
