using ProjectAI.Core;

namespace ProjectAI.Models;

/// <summary>Root-mean-square layer norm (no mean subtraction) with a single learned per-feature scale.</summary>
public sealed class RmsNorm : Module
{
    private readonly Tensor _weight;
    private readonly float _epsilon;

    public int Dim { get; }

    public RmsNorm(ParameterContext ctx, int dim, float epsilon) : base(ctx)
    {
        Dim = dim;
        _epsilon = epsilon;
        _weight = Param("weight", new Shape(dim), Init.Ones); // RMSNorm scale starts at 1
    }

    public override Tensor Forward(Tensor input, ForwardContext ctx) => Ag.RmsNorm(input, _weight, _epsilon);
}

/// <summary>
/// Rotary position embedding (rotate-half / Llama convention). Holds precomputed, non-trainable cos/sin
/// tables of shape [maxSeq, headDim] in the duplicated layout the backend kernel expects.
/// </summary>
public sealed class RotaryEmbedding : Module
{
    private readonly Tensor _cos;
    private readonly Tensor _sin;

    public int HeadDim { get; }

    public RotaryEmbedding(ParameterContext ctx, int headDim, float theta, int maxSequenceLength) : base(ctx)
    {
        if (headDim % 2 != 0) throw new ArgumentException($"RoPE head dim must be even; got {headDim}.");
        HeadDim = headDim;
        (_cos, _sin) = BuildTables(Backend, headDim, theta, maxSequenceLength);
    }

    private static (Tensor Cos, Tensor Sin) BuildTables(IComputeBackend backend, int headDim, float theta, int maxSeq)
    {
        int half = headDim / 2;
        var cos = new float[(long)maxSeq * headDim];
        var sin = new float[(long)maxSeq * headDim];
        for (int p = 0; p < maxSeq; p++)
            for (int j = 0; j < half; j++)
            {
                float freq = 1f / MathF.Pow(theta, (2f * j) / headDim);
                float angle = p * freq;
                float c = MathF.Cos(angle), s = MathF.Sin(angle);
                // Duplicated halves: feature j and j+half rotate by the same angle.
                cos[p * headDim + j] = c; cos[p * headDim + j + half] = c;
                sin[p * headDim + j] = s; sin[p * headDim + j + half] = s;
            }
        return (backend.FromHost(cos, new Shape(maxSeq, headDim), DType.F32),
                backend.FromHost(sin, new Shape(maxSeq, headDim), DType.F32));
    }

    /// <summary>Applies RoPE to <paramref name="x"/> (shape [.., seq, headDim]) at positions 0..seq-1.</summary>
    public Tensor Apply(Tensor x, ForwardContext ctx)
    {
        // Full-sequence forward (training/prefill) uses positions 0..seq-1 — the case ctx.Positions
        // already represents for prefill, so a non-null Positions is fine here. A position OFFSET only
        // arises with the KV-cache decode path (ctx.Cache), which is wired in later (S1-7b).
        if (ctx.Cache is not null)
            throw new NotImplementedException("RoPE position offset / KV-cache decode — ticket S1-7b.");

        int seq = x.Shape[x.Shape.Rank - 2];
        var cos = _cos.Slice(0, 0, seq); // [seq, headDim] constant view, broadcasts over leading dims
        var sin = _sin.Slice(0, 0, seq);
        return Ag.RotaryEmbedding(x, cos, sin);
    }

    public override Tensor Forward(Tensor input, ForwardContext ctx) => Apply(input, ctx);
}

/// <summary>SwiGLU feed-forward network: down(silu(gate(x)) * up(x)). Three bias-free linears.</summary>
public sealed class SwiGluFeedForward : Module
{
    private readonly Linear _gate;
    private readonly Linear _up;
    private readonly Linear _down;

    public int Dim { get; }
    public int HiddenDim { get; }

    public SwiGluFeedForward(ParameterContext ctx, int dim, int hiddenDim) : base(ctx)
    {
        Dim = dim;
        HiddenDim = hiddenDim;
        _gate = RegisterModule("gate", new Linear(ctx, dim, hiddenDim, Init.Kaiming()));
        _up = RegisterModule("up", new Linear(ctx, dim, hiddenDim, Init.Kaiming()));
        _down = RegisterModule("down", new Linear(ctx, hiddenDim, dim, Init.Kaiming()));
    }

    public override Tensor Forward(Tensor input, ForwardContext ctx)
    {
        var gate = Ag.Silu(_gate.Forward(input, ctx)); // [.., hidden]
        var up = _up.Forward(input, ctx);              // [.., hidden]
        return _down.Forward(Ag.Mul(gate, up), ctx);   // [.., dim]
    }
}

/// <summary>
/// Grouped-query attention (ticket S1-7), full-sequence (training/prefill) path. Q/K/V/O projections (no
/// bias), RoPE on Q/K, GQA head sharing (KvHeadCount ≤ HeadCount), scaled dot-product with a causal mask,
/// softmax, and an output projection. The incremental KV-cache decode path is a follow-up (S1-7b).
/// </summary>
public sealed class Attention : Module
{
    private readonly Linear _wq, _wk, _wv, _wo;
    private readonly RotaryEmbedding _rope;
    private readonly Tensor _invSqrtHeadDim; // [1] constant scale, no gradient

    public ModelConfig Config { get; }

    public Attention(ParameterContext ctx, ModelConfig config) : base(ctx)
    {
        config.Validate();
        Config = config;
        int dModel = config.EmbeddingDim, h = config.HeadCount, kvh = config.KvHeadCount, dh = config.HeadDim;
        _wq = RegisterModule("wq", new Linear(ctx, dModel, h * dh));
        _wk = RegisterModule("wk", new Linear(ctx, dModel, kvh * dh));
        _wv = RegisterModule("wv", new Linear(ctx, dModel, kvh * dh));
        _wo = RegisterModule("wo", new Linear(ctx, h * dh, dModel));
        _rope = RegisterModule("rope", new RotaryEmbedding(ctx, dh, config.RopeTheta, config.MaxSequenceLength));
        _invSqrtHeadDim = Backend.FromHost([1f / MathF.Sqrt(dh)], new Shape(1), DType.F32);
    }

    public override Tensor Forward(Tensor input, ForwardContext ctx)
    {
        if (ctx.Cache is not null) throw new NotImplementedException("KV-cache decode — ticket S1-7b.");

        int b = input.Shape[0], s = input.Shape[1];
        int h = Config.HeadCount, kvh = Config.KvHeadCount, dh = Config.HeadDim, g = h / kvh;

        // 1. Projections → 2. split into heads [b, heads, s, dh].
        var qh = Ag.Transpose(Ag.Reshape(_wq.Forward(input, ctx), b, s, h, dh), 1, 2);   // [b,h,s,dh]
        var kh = Ag.Transpose(Ag.Reshape(_wk.Forward(input, ctx), b, s, kvh, dh), 1, 2); // [b,kvh,s,dh]
        var vh = Ag.Transpose(Ag.Reshape(_wv.Forward(input, ctx), b, s, kvh, dh), 1, 2); // [b,kvh,s,dh]

        // 3. RoPE on q and k (not v).
        qh = _rope.Apply(qh, ctx);
        kh = _rope.Apply(kh, ctx);

        // 4. GQA: regroup heads as (kvh, group) with a size-1 group axis on k/v so it broadcasts.
        var q5 = Ag.Reshape(Ag.Contiguous(qh), b, kvh, g, s, dh); // [b,kvh,g,s,dh]
        var k5 = Ag.Reshape(Ag.Contiguous(kh), b, kvh, 1, s, dh); // [b,kvh,1,s,dh]
        var v5 = Ag.Reshape(Ag.Contiguous(vh), b, kvh, 1, s, dh);

        // 5. Scaled scores = (q/√dh) · kᵀ → [b,kvh,g,s,s].
        var scores = Ag.MatMul(Ag.Mul(q5, _invSqrtHeadDim), k5, transposeB: true);

        // 6. Causal mask + softmax over the key axis. An explicit per-batch mask [b,s,s] must have its
        // batch axis aligned to scores' batch axis (axis 0), not right-broadcast onto the GQA group axis;
        // insert size-1 head/group axes. A rank-2 [s,s] mask broadcasts over all leading axes already.
        var mask = ctx.AttentionMask ?? CausalMask(s);
        if (mask.Shape.Rank == 3) mask = mask.Reshape(b, 1, 1, s, s);
        var probs = Ag.Softmax(Ag.Add(scores, mask), axis: -1); // [b,kvh,g,s,s]

        // 7. Context = probs · v → [b,kvh,g,s,dh].
        var context = Ag.MatMul(probs, v5);

        // 8. Merge heads back to [b, s, h*dh].
        var heads = Ag.Reshape(Ag.Contiguous(context), b, h, s, dh);      // collapse (kvh,g)→h
        var merged = Ag.Reshape(Ag.Contiguous(Ag.Transpose(heads, 1, 2)), b, s, h * dh);

        // 9. Output projection.
        return _wo.Forward(merged, ctx);
    }

    /// <summary>Additive causal mask [s, s]: 0 where key ≤ query, a large negative otherwise (constant).</summary>
    private Tensor CausalMask(int s)
    {
        var m = new float[(long)s * s];
        for (int i = 0; i < s; i++)
            for (int j = 0; j < s; j++)
                m[i * s + j] = j <= i ? 0f : -1e9f;
        return Backend.FromHost(m, new Shape(s, s), DType.F32);
    }
}

/// <summary>One pre-norm transformer block: x + attn(norm(x)), then x + ffn(norm(x)) (ticket S1-8).</summary>
public sealed class TransformerBlock : Module
{
    public TransformerBlock(ParameterContext ctx, ModelConfig config) : base(ctx)
    {
        Config = config;
        AttentionNorm = RegisterModule("attn_norm", new RmsNorm(ctx, config.EmbeddingDim, config.NormEpsilon));
        Attention = RegisterModule("attn", new Attention(ctx, config));
        FeedForwardNorm = RegisterModule("ffn_norm", new RmsNorm(ctx, config.EmbeddingDim, config.NormEpsilon));
        FeedForward = RegisterModule("ffn", new SwiGluFeedForward(ctx, config.EmbeddingDim, config.FeedForwardHiddenDim));
    }

    public ModelConfig Config { get; }
    public RmsNorm AttentionNorm { get; }
    public Attention Attention { get; }
    public RmsNorm FeedForwardNorm { get; }
    public SwiGluFeedForward FeedForward { get; }

    // Pre-norm residual block: x + attn(norm(x)), then x + ffn(norm(x)).
    public override Tensor Forward(Tensor input, ForwardContext ctx)
    {
        var h = Ag.Add(input, Attention.Forward(AttentionNorm.Forward(input, ctx), ctx));
        return Ag.Add(h, FeedForward.Forward(FeedForwardNorm.Forward(h, ctx), ctx));
    }
}

/// <summary>
/// Full Llama-style decoder: token embedding → N pre-norm transformer blocks → final RMSNorm → LM head
/// (weight-tied to the embedding). Forward maps token ids to logits over the vocabulary (ticket S1-8).
/// </summary>
public sealed class LlamaModel : Module
{
    private readonly Embedding _embedding;
    private readonly List<TransformerBlock> _blocks = [];

    public LlamaModel(ParameterContext ctx, ModelConfig config) : base(ctx)
    {
        config.Validate();
        Config = config;
        _embedding = RegisterModule("embedding", new Embedding(ctx, config.VocabSize, config.EmbeddingDim));
        for (var i = 0; i < config.LayerCount; i++)
            _blocks.Add(RegisterModule($"block.{i}", new TransformerBlock(ctx, config)));
        FinalNorm = RegisterModule("final_norm", new RmsNorm(ctx, config.EmbeddingDim, config.NormEpsilon));
    }

    public ModelConfig Config { get; }
    public RmsNorm FinalNorm { get; }
    public IReadOnlyList<TransformerBlock> Blocks => _blocks;

    /// <summary>Maps token ids [batch, seq] (carried as floats) to logits [batch, seq, vocab].</summary>
    public override Tensor Forward(Tensor tokenIds, ForwardContext ctx)
    {
        var h = _embedding.Forward(tokenIds, ctx); // [batch, seq, dModel]
        foreach (var block in _blocks) h = block.Forward(h, ctx);
        h = FinalNorm.Forward(h, ctx);
        return Ag.MatMul(h, _embedding.Weight, transposeB: true); // tied LM head → [batch, seq, vocab]
    }
}

/// <summary>
/// Per-request key/value cache for incremental decoding. A PagedAttention-style block table is planned
/// for Stage 3; the read/write path is implemented with attention (ticket S1-7).
/// </summary>
public sealed class KvCache(ModelConfig config, int maxBatch, int maxSequenceLength) : IKvCache
{
    public ModelConfig Config { get; } = config;
    public int MaxBatch { get; } = maxBatch;
    public int MaxSequenceLength { get; } = maxSequenceLength;

    public int Length(int layer) => throw new NotImplementedException("ticket S1-7.");
    public (Tensor Keys, Tensor Values) Append(int layer, Tensor keys, Tensor values) => throw new NotImplementedException("ticket S1-7.");
    public void Reset() => throw new NotImplementedException("ticket S1-7.");
}
