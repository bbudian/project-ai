namespace ProjectAI.Models;

/// <summary>
/// Named model-size presets so a trainer or CLI can pick capacity by name (tiny → large) instead of juggling raw
/// dims. Each returns a <see cref="ModelConfig"/> that passes <see cref="ModelConfig.Validate"/> for the byte-level
/// vocabulary (259 tokens); callers override <see cref="ModelConfig.VocabSize"/> for a different tokenizer and
/// <see cref="ModelConfig.MaxSequenceLength"/> for a longer training context.
/// </summary>
public static class ModelPresets
{
    /// <summary>Vocabulary size of the byte-level tokenizer (256 bytes + PAD/BOS/EOS).</summary>
    public const int ByteLevelVocab = 259;

    /// <summary>Preset names, smallest to largest.</summary>
    public static IReadOnlyList<string> Names { get; } = ["tiny", "small", "medium", "large"];

    /// <summary>Short human description of each preset (layers/dim and the rough parameter count at byte-level vocab).</summary>
    public static string Describe(string name)
    {
        var c = Get(name);
        return $"{c.LayerCount} layers, dim {c.EmbeddingDim}, {c.HeadCount} heads (KV {c.KvHeadCount}), ctx {c.MaxSequenceLength}";
    }

    /// <summary>
    /// Memory-aware default (batch, sequenceLength) for training a given size — bigger models use a smaller batch
    /// so a single step fits typical VRAM (deterministic per-step memory is ticket S2-3). Callers may override.
    /// </summary>
    public static (int Batch, int SequenceLength) DefaultTraining(string name) => (name ?? "").Trim().ToLowerInvariant() switch
    {
        "tiny" => (32, 128),
        "small" => (16, 128),
        "medium" => (16, 128), // fits 8GB at batch 16 thanks to the S2-3 per-step memory scoping
        "large" => (1, 128),   // ~208M params: needs a big-VRAM GPU; a single pass won't fit an 8GB card (→ S3-2)
        _ => (16, 128),
    };

    /// <summary>The config for a named size. Throws on an unknown name.</summary>
    public static ModelConfig Get(string name) => (name ?? "").Trim().ToLowerInvariant() switch
    {
        "tiny" => Make(layers: 2, dim: 64, heads: 4, kvHeads: 2, context: 256),     // ~0.1M params — instant, memorizes
        "small" => Make(layers: 6, dim: 256, heads: 8, kvHeads: 4, context: 512),    // ~5M params — minutes on GPU, learns style
        "medium" => Make(layers: 12, dim: 512, heads: 8, kvHeads: 4, context: 1024),  // ~40M params — more capable, wants more data
        "large" => Make(layers: 24, dim: 768, heads: 12, kvHeads: 4, context: 1024),  // ~150M params — uses real VRAM, needs a big corpus
        _ => throw new ArgumentException($"unknown size preset '{name}' (known: {string.Join(", ", Names)})"),
    };

    // SwiGLU FFN sized at 4x the model dim — the conventional ratio; all dims chosen so HeadDim is even (RoPE) and
    // EmbeddingDim divides by HeadCount and HeadCount by KvHeadCount (GQA), i.e. every preset passes Validate().
    private static ModelConfig Make(int layers, int dim, int heads, int kvHeads, int context)
    {
        var config = new ModelConfig
        {
            VocabSize = ByteLevelVocab,
            EmbeddingDim = dim,
            LayerCount = layers,
            HeadCount = heads,
            KvHeadCount = kvHeads,
            FeedForwardHiddenDim = 4 * dim,
            MaxSequenceLength = context,
        };
        config.Validate(); // fail loudly if a preset is ever edited into an invalid shape
        return config;
    }
}
