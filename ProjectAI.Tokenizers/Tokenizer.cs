using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace ProjectAI.Tokenizers;

/// <summary>Converts between text and integer token ids.</summary>
public interface ITokenizer
{
    int VocabSize { get; }
    int BosId { get; }
    int EosId { get; }
    IReadOnlyList<int> Encode(string text, bool addBos = false, bool addEos = false);
    string Decode(IReadOnlyList<int> ids);
}

/// <summary>Byte-Pair-Encoding tokenizer. Trained from a corpus or loaded from a vocab/merges file.</summary>
public interface IBpeTokenizer : ITokenizer;

/// <summary>
/// Byte-level BPE over UTF-8 (GPT-2 style), ticket S1-1. The base vocabulary is the 256 byte values,
/// then three special tokens (PAD/BOS/EOS), then the learned merges. Working at the byte level makes
/// <see cref="Decode"/>(<see cref="Encode"/>(s)) == s for every well-formed string (incl. emoji/CJK) by
/// construction. <see cref="Encode"/> rejects ill-formed UTF-16 (unpaired surrogates) with an exception
/// rather than silently substituting U+FFFD; <see cref="Decode"/> is lenient so a truncated/partial token
/// stream (e.g. mid-generation) degrades to U+FFFD instead of throwing.
/// Train with <see cref="BpeTrainer"/>; persist with <see cref="ToJson"/>/<see cref="FromJson"/>.
/// </summary>
public sealed class BpeTokenizer : IBpeTokenizer
{
    internal const int ByteCount = 256;
    internal const int SpecialCount = 3;                 // PAD, BOS, EOS
    internal const int FirstMergeId = ByteCount + SpecialCount; // 259

    // GPT-2-style pre-tokenization: contractions, then runs of letters / numbers / other / whitespace,
    // each optionally led by a single space. Unicode-aware and exhaustive (every char matches exactly one
    // alternative), so the matched chunks partition the text with no dropped characters.
    private static readonly Regex PretokenRegex = new(
        @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    // Strict UTF-8 for encoding: an unpaired surrogate raises EncoderFallbackException instead of being
    // silently replaced by U+FFFD, which would break the lossless-roundtrip guarantee. Decoding stays
    // lenient (Encoding.UTF8) so a partial/truncated token stream degrades to U+FFFD rather than throwing.
    private static readonly UTF8Encoding Utf8Strict = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    /// <summary>UTF-8 encodes text, throwing on ill-formed UTF-16 (shared by encode and training).</summary>
    internal static byte[] EncodeUtf8(string text) => Utf8Strict.GetBytes(text);

    private readonly IReadOnlyList<(int A, int B)> _merges;
    private readonly Dictionary<(int, int), int> _rank = new();      // pair -> merge order
    private readonly Dictionary<(int, int), int> _mergedId = new();  // pair -> resulting token id
    private readonly byte[][] _idToBytes;                            // token id -> raw bytes (specials = empty)

    public int PadId => ByteCount;       // 256
    public int BosId => ByteCount + 1;   // 257
    public int EosId => ByteCount + 2;   // 258
    public int VocabSize { get; }

    /// <summary>Builds a tokenizer from an ordered merge list (each entry merges two existing token ids).</summary>
    public BpeTokenizer(IReadOnlyList<(int A, int B)> merges)
    {
        _merges = merges;
        VocabSize = FirstMergeId + merges.Count;
        _idToBytes = new byte[VocabSize][];
        for (int i = 0; i < ByteCount; i++) _idToBytes[i] = [(byte)i];
        for (int i = ByteCount; i < FirstMergeId; i++) _idToBytes[i] = [];
        for (int i = 0; i < merges.Count; i++)
        {
            var (a, b) = merges[i];
            int id = FirstMergeId + i;
            // Children must be ids defined before this merge (no forward/out-of-range references).
            if ((uint)a >= (uint)id || (uint)b >= (uint)id)
                throw new ArgumentException($"Merge {i} ({a},{b}) references a token id not defined before {id}.", nameof(merges));
            _rank[(a, b)] = i;
            _mergedId[(a, b)] = id;
            _idToBytes[id] = [.. _idToBytes[a], .. _idToBytes[b]];
        }
    }

    /// <summary>Splits text into pre-token chunks; merges never cross chunk boundaries.</summary>
    internal static IEnumerable<string> Pretokenize(string text)
    {
        foreach (Match m in PretokenRegex.Matches(text)) yield return m.Value;
    }

    public IReadOnlyList<int> Encode(string text, bool addBos = false, bool addEos = false)
    {
        var ids = new List<int>();
        if (addBos) ids.Add(BosId);
        foreach (var chunk in Pretokenize(text))
        {
            var bytes = EncodeUtf8(chunk);
            if (bytes.Length == 0) continue;
            var tokens = new List<int>(bytes.Length);
            foreach (var b in bytes) tokens.Add(b);
            MergeTokens(tokens);
            ids.AddRange(tokens);
        }
        if (addEos) ids.Add(EosId);
        return ids;
    }

    /// <summary>Greedily applies the lowest-rank (earliest-learned) adjacent merge until none remain.</summary>
    private void MergeTokens(List<int> tokens)
    {
        while (tokens.Count >= 2)
        {
            int bestRank = int.MaxValue, bestPos = -1;
            for (int i = 0; i < tokens.Count - 1; i++)
                if (_rank.TryGetValue((tokens[i], tokens[i + 1]), out int r) && r < bestRank)
                {
                    bestRank = r;
                    bestPos = i;
                }
            if (bestPos < 0) break;
            tokens[bestPos] = _mergedId[(tokens[bestPos], tokens[bestPos + 1])];
            tokens.RemoveAt(bestPos + 1);
        }
    }

    public string Decode(IReadOnlyList<int> ids)
    {
        var bytes = new List<byte>();
        foreach (int id in ids)
        {
            if (id == PadId || id == BosId || id == EosId) continue; // special tokens carry no bytes
            if ((uint)id >= (uint)VocabSize)
                throw new ArgumentOutOfRangeException(nameof(ids), id, $"Token id out of range [0,{VocabSize}).");
            bytes.AddRange(_idToBytes[id]);
        }
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    // --- Persistence: vocab is implicit (bytes + specials); only the ordered merges need storing. ---
    public string ToJson() => JsonSerializer.Serialize(
        new TokenizerData { Merges = [.. _merges.Select(m => new[] { m.A, m.B })] }, TokenizerJsonContext.Options);

    public static BpeTokenizer FromJson(string json)
    {
        var data = JsonSerializer.Deserialize<TokenizerData>(json, TokenizerJsonContext.Options)
            ?? throw new FormatException("Invalid tokenizer JSON.");
        var merges = data.Merges.Select(m =>
            m is { Length: 2 } ? (m[0], m[1]) : throw new FormatException("Each merge must be a [a, b] pair.")).ToList();
        return new BpeTokenizer(merges); // constructor validates id references
    }

    public void Save(string path) => File.WriteAllText(path, ToJson());
    public static BpeTokenizer Load(string path) => FromJson(File.ReadAllText(path));
}

/// <summary>Serializable form of a trained tokenizer (merge order is the vocabulary).</summary>
internal sealed class TokenizerData
{
    public int Version { get; set; } = 1;
    public List<int[]> Merges { get; set; } = [];
}

internal static class TokenizerJsonContext
{
    public static readonly JsonSerializerOptions Options = new() { WriteIndented = false };
}
