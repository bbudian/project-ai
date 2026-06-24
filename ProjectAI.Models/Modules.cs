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
    private readonly int _maxSeq;

    public int HeadDim { get; }

    public RotaryEmbedding(ParameterContext ctx, int headDim, float theta, int maxSequenceLength) : base(ctx)
    {
        if (headDim % 2 != 0) throw new ArgumentException($"RoPE head dim must be even; got {headDim}.");
        HeadDim = headDim;
        _maxSeq = maxSequenceLength;
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
    public Tensor Apply(Tensor x, ForwardContext ctx) => ApplyAtOffset(x, 0);

    /// <summary>
    /// Applies RoPE to <paramref name="x"/> (shape [.., seq, headDim]) at absolute positions
    /// <paramref name="positionOffset"/>..<paramref name="positionOffset"/>+seq-1. The offset is non-zero only
    /// on the incremental KV-cache decode path, where the new token sits past the already-cached positions.
    /// </summary>
    public Tensor ApplyAtOffset(Tensor x, int positionOffset)
    {
        int seq = x.Shape[x.Shape.Rank - 2];
        if (positionOffset < 0 || positionOffset + seq > _maxSeq)
            throw new ArgumentOutOfRangeException(nameof(positionOffset),
                $"RoPE positions {positionOffset}..{positionOffset + seq - 1} exceed the table length {_maxSeq}.");
        var cos = _cos.Slice(0, positionOffset, seq); // [seq, headDim] view at positions offset..offset+seq-1
        var sin = _sin.Slice(0, positionOffset, seq);
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
/// Grouped-query attention (ticket S1-7). Q/K/V/O projections (no bias), RoPE on Q/K, GQA head sharing
/// (KvHeadCount ≤ HeadCount), scaled dot-product with a causal mask, softmax, and an output projection. Handles
/// both the full-sequence (training/prefill) path and the incremental KV-cache decode path (ticket S1-7b): when
/// <c>ctx.Cache</c> is set, only the new tokens are projected/RoPE'd (at the cached position offset), their K/V
/// are appended to the cache, and the new queries attend over the full cached K/V. <see cref="_layerIndex"/>
/// addresses this layer's slot in the shared cache.
/// </summary>
public sealed class Attention : Module
{
    private readonly Linear _wq, _wk, _wv, _wo;
    private readonly RotaryEmbedding _rope;
    private readonly Tensor _invSqrtHeadDim; // [1] constant scale, no gradient
    private readonly int _layerIndex;

    public ModelConfig Config { get; }

    public Attention(ParameterContext ctx, ModelConfig config, int layerIndex = 0) : base(ctx)
    {
        config.Validate();
        Config = config;
        _layerIndex = layerIndex;
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
        var cache = ctx.Cache;
        if (cache is not null && GradMode.IsEnabled)
            throw new InvalidOperationException("KV-cache decode is inference-only; wrap it in GradMode.NoGrad().");

        int b = input.Shape[0], s = input.Shape[1];
        int h = Config.HeadCount, kvh = Config.KvHeadCount, dh = Config.HeadDim, g = h / kvh;
        int posOffset = cache?.Length(_layerIndex) ?? 0; // absolute position of the first new token

        // 1. Projections → 2. split into heads [b, heads, s, dh].
        var qh = Ag.Transpose(Ag.Reshape(_wq.Forward(input, ctx), b, s, h, dh), 1, 2);   // [b,h,s,dh]
        var kh = Ag.Transpose(Ag.Reshape(_wk.Forward(input, ctx), b, s, kvh, dh), 1, 2); // [b,kvh,s,dh]
        var vh = Ag.Transpose(Ag.Reshape(_wv.Forward(input, ctx), b, s, kvh, dh), 1, 2); // [b,kvh,s,dh]

        // 3. RoPE on q and k (not v), at absolute positions posOffset..posOffset+s-1.
        qh = _rope.ApplyAtOffset(qh, posOffset);
        kh = _rope.ApplyAtOffset(kh, posOffset);

        // 3b. Decode: append this layer's new K/V to the cache and attend over the full history.
        int keyLen = s;
        if (cache is not null)
        {
            (kh, vh) = cache.Append(_layerIndex, kh, vh); // now [b,kvh,posOffset+s,dh]
            keyLen = posOffset + s;
        }

        // 4. GQA: regroup heads as (kvh, group) with a size-1 group axis on k/v so it broadcasts.
        var q5 = Ag.Reshape(Ag.Contiguous(qh), b, kvh, g, s, dh);      // [b,kvh,g,s,dh]
        var k5 = Ag.Reshape(Ag.Contiguous(kh), b, kvh, 1, keyLen, dh); // [b,kvh,1,keyLen,dh]
        var v5 = Ag.Reshape(Ag.Contiguous(vh), b, kvh, 1, keyLen, dh);

        // 5. Scaled scores = (q/√dh) · kᵀ → [b,kvh,g,s,keyLen].
        var scores = Ag.MatMul(Ag.Mul(q5, _invSqrtHeadDim), k5, transposeB: true);

        // 6. Causal mask + softmax over the key axis. An explicit per-batch mask [b,q,k] must have its batch
        // axis aligned to scores' batch axis (axis 0), not right-broadcast onto the GQA group axis; insert
        // size-1 head/group axes. A rank-2 mask broadcasts over all leading axes already.
        var mask = ctx.AttentionMask ?? CausalMask(s, keyLen, posOffset);
        if (mask.Shape.Rank == 3) mask = mask.Reshape(b, 1, 1, mask.Shape[1], mask.Shape[2]);
        var probs = Ag.Softmax(Ag.Add(scores, mask), axis: -1); // [b,kvh,g,s,keyLen]

        // 7. Context = probs · v → [b,kvh,g,s,dh].
        var context = Ag.MatMul(probs, v5);

        // 8. Merge heads back to [b, s, h*dh].
        var heads = Ag.Reshape(Ag.Contiguous(context), b, h, s, dh);      // collapse (kvh,g)→h
        var merged = Ag.Reshape(Ag.Contiguous(Ag.Transpose(heads, 1, 2)), b, s, h * dh);

        // 9. Output projection.
        return _wo.Forward(merged, ctx);
    }

    /// <summary>
    /// Additive causal mask [queryLen, keyLen]: query i sits at absolute position <c>posOffset + i</c> and may
    /// attend to keys 0..posOffset+i; later keys get a large negative. Covers the full-sequence path
    /// (posOffset 0, square mask) and decode (queryLen 1 over the full history → all-zero, no masking).
    /// </summary>
    private Tensor CausalMask(int queryLen, int keyLen, int posOffset)
    {
        var m = new float[(long)queryLen * keyLen];
        for (int i = 0; i < queryLen; i++)
            for (int j = 0; j < keyLen; j++)
                m[i * keyLen + j] = j <= posOffset + i ? 0f : -1e9f;
        return Backend.FromHost(m, new Shape(queryLen, keyLen), DType.F32);
    }
}

/// <summary>One pre-norm transformer block: x + attn(norm(x)), then x + ffn(norm(x)) (ticket S1-8).</summary>
public sealed class TransformerBlock : Module
{
    public TransformerBlock(ParameterContext ctx, ModelConfig config, int layerIndex = 0) : base(ctx)
    {
        Config = config;
        AttentionNorm = RegisterModule("attn_norm", new RmsNorm(ctx, config.EmbeddingDim, config.NormEpsilon));
        Attention = RegisterModule("attn", new Attention(ctx, config, layerIndex));
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
            _blocks.Add(RegisterModule($"block.{i}", new TransformerBlock(ctx, config, i)));
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
/// Per-request key/value cache for incremental decoding (ticket S1-7b). Holds, per layer, the post-RoPE keys
/// and raw values for every position seen so far as <c>[batch, kvHeads, length, headDim]</c>; <see cref="Append"/>
/// grows them along the sequence axis and returns the full history for attention. Growth is done host-side
/// (gather → concat → upload), which keeps it backend-agnostic; a paged/in-place layout is a later optimization.
/// <para>
/// The cached length is a single per-layer scalar shared by the whole batch, so a cache supports only a single
/// stream or a uniform-length, lockstep batch (every row at the same position). Ragged batched generation
/// (different prompt lengths, or one stream stopping early) would need per-row lengths and is not supported.
/// </para>
/// </summary>
public sealed class KvCache : IKvCache
{
    private readonly IComputeBackend _backend;
    private readonly Tensor?[] _keys;
    private readonly Tensor?[] _values;
    private readonly int[] _lengths;

    public ModelConfig Config { get; }
    public int MaxBatch { get; }
    public int MaxSequenceLength { get; }

    public KvCache(IComputeBackend backend, ModelConfig config, int maxBatch, int maxSequenceLength)
    {
        _backend = backend;
        Config = config;
        MaxBatch = maxBatch;
        MaxSequenceLength = maxSequenceLength;
        _keys = new Tensor?[config.LayerCount];
        _values = new Tensor?[config.LayerCount];
        _lengths = new int[config.LayerCount];
    }

    public int Length(int layer) => (uint)layer < (uint)_lengths.Length
        ? _lengths[layer]
        : throw new ArgumentOutOfRangeException(nameof(layer), $"layer {layer} is out of range for a {_lengths.Length}-layer cache.");

    public (Tensor Keys, Tensor Values) Append(int layer, Tensor keys, Tensor values)
    {
        if ((uint)layer >= (uint)_keys.Length)
            throw new ArgumentOutOfRangeException(nameof(layer), $"layer {layer} is out of range for a {_keys.Length}-layer cache (model/cache config mismatch?).");
        int batch = keys.Shape[0], newLen = keys.Shape[2];
        if (batch > MaxBatch)
            throw new InvalidOperationException($"batch {batch} exceeds the cache's MaxBatch {MaxBatch}.");
        if (_lengths[layer] + newLen > MaxSequenceLength) // also covers the first append (length 0)
            throw new InvalidOperationException($"KV cache overflow: {_lengths[layer] + newLen} positions exceeds the maximum {MaxSequenceLength}.");

        _keys[layer] = ConcatSequence(_keys[layer], keys);
        _values[layer] = ConcatSequence(_values[layer], values);
        _lengths[layer] = _keys[layer]!.Shape[2];
        return (_keys[layer]!, _values[layer]!);
    }

    public void Reset()
    {
        Array.Clear(_keys);
        Array.Clear(_values);
        Array.Clear(_lengths);
    }

    // Concatenates two [batch, kvHeads, *, headDim] tensors along the sequence axis (2). ToHost handles the
    // (possibly strided) incoming view; the result is a fresh contiguous tensor.
    private Tensor ConcatSequence(Tensor? existing, Tensor incoming)
    {
        if (existing is null)
        {
            var copy = new float[incoming.ElementCount];
            _backend.ToHost(incoming, copy);
            return _backend.FromHost(copy, incoming.Shape, DType.F32);
        }

        int batch = existing.Shape[0], kvHeads = existing.Shape[1], headDim = existing.Shape[3];
        int oldLen = existing.Shape[2], newLen = incoming.Shape[2];
        int total = oldLen + newLen; // bounds already checked in Append

        var oldHost = new float[existing.ElementCount];
        var newHost = new float[incoming.ElementCount];
        _backend.ToHost(existing, oldHost);
        _backend.ToHost(incoming, newHost);

        var combined = new float[(long)batch * kvHeads * total * headDim];
        for (int bi = 0; bi < batch; bi++)
            for (int hi = 0; hi < kvHeads; hi++)
            {
                int dst = ((bi * kvHeads + hi) * total) * headDim;
                Array.Copy(oldHost, ((bi * kvHeads + hi) * oldLen) * headDim, combined, dst, oldLen * headDim);
                Array.Copy(newHost, ((bi * kvHeads + hi) * newLen) * headDim, combined, dst + oldLen * headDim, newLen * headDim);
            }
        return _backend.FromHost(combined, new Shape(batch, kvHeads, total, headDim), DType.F32);
    }
}
