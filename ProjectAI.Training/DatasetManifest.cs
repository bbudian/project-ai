using System.Text.Json;

namespace ProjectAI.Training;

/// <summary>
/// Self-describing sidecar for a packed token-id dataset (the first stage of the HuggingFace data path). It records
/// everything needed to (a) read the <c>.bin</c> correctly — element dtype, sequence length, shard list — and
/// (b) reproduce/validate the token stream — the tokenizer that produced it (kind + state + hash), vocab size, and
/// source provenance. Crucially, <c>train --data</c> reads <see cref="SequenceLength"/>, <see cref="VocabSize"/>,
/// and the tokenizer FROM here, never from CLI defaults, so a dataset packed at one sequence length can't be
/// silently trained at another (the default <c>--seqlen</c> would otherwise truncate every block).
/// </summary>
public sealed record DatasetManifest
{
    /// <summary>Bumped on a breaking change to the on-disk layout or this schema.</summary>
    public int FormatVersion { get; init; } = 1;

    /// <summary>Tokenizer persistence tag, e.g. <c>"bpe"</c> or <c>"hf"</c> (see <c>TokenizerCodec</c>).</summary>
    public required string TokenizerKind { get; init; }

    /// <summary>Serialized tokenizer state, sufficient to reconstruct the exact tokenizer that produced the ids.</summary>
    public required string TokenizerState { get; init; }

    /// <summary>Hash of <see cref="TokenizerKind"/>+<see cref="TokenizerState"/>, to detect a tokenizer mismatch.</summary>
    public required string TokenizerHash { get; init; }

    /// <summary>Element width of the token ids in the <c>.bin</c>: <c>"u16"</c> (vocab ≤ 65536) or <c>"u32"</c>.</summary>
    public required string Dtype { get; init; }

    public required int VocabSize { get; init; }

    /// <summary>Model context length used when packing; each block holds <c>SequenceLength + 1</c> ids.</summary>
    public required int SequenceLength { get; init; }

    /// <summary>Total token ids written to the shard(s), including the unused trailing remainder.</summary>
    public required long TotalTokens { get; init; }

    public required long BlockCount { get; init; }

    /// <summary>Trailing tokens that don't fill a final block and are never read.</summary>
    public required long DroppedTokens { get; init; }

    /// <summary>Shard file names relative to the manifest (one for now; sharding is a later phase).</summary>
    public required string[] Shards { get; init; }

    /// <summary>Where the corpus came from — a local path or a HuggingFace dataset id.</summary>
    public string? SourceId { get; init; }

    public string? SourceRevision { get; init; }

    public string? CreatedUtc { get; init; }

    /// <summary>Canonical manifest file name inside a dataset directory.</summary>
    public const string FileName = "manifest.json";

    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    public string ToJson() => JsonSerializer.Serialize(this, JsonOptions);

    public static DatasetManifest FromJson(string json) =>
        JsonSerializer.Deserialize<DatasetManifest>(json, JsonOptions)
        ?? throw new InvalidDataException("dataset manifest JSON is empty or invalid.");

    /// <summary>
    /// Resolves a <c>--data</c> argument to a manifest path: accepts the manifest file itself, or a directory that
    /// contains one. Returns null when the path is neither (so callers can fall back to treating it as raw text).
    /// </summary>
    public static string? TryResolvePath(string dataPath)
    {
        if (Directory.Exists(dataPath))
        {
            string candidate = Path.Combine(dataPath, FileName);
            return File.Exists(candidate) ? candidate : null;
        }
        if (File.Exists(dataPath) &&
            string.Equals(Path.GetFileName(dataPath), FileName, StringComparison.OrdinalIgnoreCase))
            return dataPath;
        return null;
    }
}
