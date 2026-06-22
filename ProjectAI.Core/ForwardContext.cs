namespace ProjectAI.Core;

/// <summary>
/// Per-request key/value cache contract (ticket S0-9). The marker lives in Core so <see cref="ForwardContext"/>
/// can carry it without Core depending on Models; the concrete cache (block layout, paged storage) lives in
/// Models and is fleshed out with attention (ticket S1-7).
/// </summary>
public interface IKvCache
{
    /// <summary>Number of tokens cached so far for the given layer.</summary>
    int Length(int layer);
    /// <summary>Appends post-RoPE keys/values for a layer and returns the full cached span.</summary>
    (Tensor Keys, Tensor Values) Append(int layer, Tensor keys, Tensor values);
    void Reset();
}

/// <summary>
/// Carries the cross-cutting inputs a forward pass may need beyond the activation tensor: the additive
/// attention mask, position ids, an optional KV cache (incremental decode), and a training flag. Leaf
/// modules ignore the fields they don't use; attention reads them. Threading one context object down the
/// stack is what lets S1-7 attention and S1-8 assembly work without changing the module contract again.
/// </summary>
public sealed record ForwardContext
{
    /// <summary>
    /// Additive attention mask (masked positions ≈ large negative); null = default causal. Shape is either
    /// [q, k] (broadcast over batch and heads) or per-batch [batch, q, k]; the attention layer aligns the
    /// latter to the batch axis. Per-head masks are not yet supported.
    /// </summary>
    public Tensor? AttentionMask { get; init; }
    /// <summary>Position ids (I32 [seq] or [batch, seq]); null = contiguous 0..seq-1.</summary>
    public Tensor? Positions { get; init; }
    /// <summary>Non-null only on the incremental-decode path.</summary>
    public IKvCache? Cache { get; init; }
    public bool IsTraining { get; init; }

    public static ForwardContext Inference() => new();
    public static ForwardContext Prefill(Tensor mask, Tensor positions) => new() { AttentionMask = mask, Positions = positions };
    public static ForwardContext Train(Tensor mask, Tensor positions) => new() { AttentionMask = mask, Positions = positions, IsTraining = true };
    public static ForwardContext Decode(IKvCache cache, Tensor positions) => new() { Cache = cache, Positions = positions };
}
