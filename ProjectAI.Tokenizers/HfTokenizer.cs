using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace ProjectAI.Tokenizers;

/// <summary>
/// A HuggingFace byte-level BPE tokenizer loaded from a <c>tokenizer.json</c> (GPT-2 / SmolLM / Llama-3 family).
/// It reproduces the model's <em>exact</em> token ids — which matters because the converted model's embedding is
/// indexed by them. Pipeline (GPT-2 style): pre-tokenize by regex → UTF-8 bytes → map each byte through the
/// GPT-2 byte↔unicode table → apply BPE merges by rank → look the merged symbols up in the vocab.
/// <para>
/// Supports the common case: a BPE model with vocab + merges and a byte-level pre-tokenizer. The pre-tokenization
/// regex defaults to the GPT-2 pattern (overridable). Models with a different split pattern (e.g. Llama-3's
/// cl100k pattern) need that pattern supplied for byte-exact parity.
/// </para>
/// </summary>
public sealed class HfTokenizer : ITokenizer
{
    // GPT-2 pre-tokenization pattern (the default for byte-level BPE models).
    public const string Gpt2Pattern = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

    private static readonly UTF8Encoding Utf8Strict = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    private readonly Dictionary<string, int> _vocab;       // token (byte-char space) → id
    private readonly string[] _idToToken;                  // id → token (null for ids with no string, e.g. specials)
    private readonly Dictionary<(string, string), int> _rank; // merge pair → rank
    private readonly Dictionary<int, string> _byteToChar;  // byte 0..255 → mapped unicode char
    private readonly Dictionary<char, int> _charToByte;    // reverse
    private readonly HashSet<int> _specialIds;             // added/special token ids (skipped on decode)
    private readonly Dictionary<string, int> _addedTokens; // added-token content → id (matched before BPE)
    private readonly Regex? _addedTokens_Regex;            // alternation of added-token contents (longest-first)
    private readonly Regex _pretokenizer;
    private readonly string _pattern;

    public int VocabSize { get; }
    public int BosId { get; }
    public int EosId { get; }

    private HfTokenizer(
        Dictionary<string, int> vocab, List<(string, string)> merges, HashSet<int> specialIds,
        Dictionary<string, int> addedTokens, int bosId, int eosId, string pretokenPattern)
    {
        _vocab = vocab;
        _specialIds = specialIds;
        _addedTokens = addedTokens;
        BosId = bosId;
        EosId = eosId;

        // Added tokens are matched as whole units before BPE; alternation tries the longest content first so
        // e.g. "<|im_start|>" wins over any shorter prefix that's also an added token.
        _addedTokens_Regex = addedTokens.Count == 0 ? null
            : new Regex(string.Join('|', addedTokens.Keys.OrderByDescending(k => k.Length).Select(Regex.Escape)),
                RegexOptions.Compiled | RegexOptions.CultureInvariant);

        // The vocabulary spans the BPE tokens and the added/special tokens (which sit at higher ids); size to the
        // largest id so VocabSize matches the model's embedding rows.
        int maxId = vocab.Count == 0 ? -1 : vocab.Values.Max();
        foreach (int id in specialIds) maxId = Math.Max(maxId, id);
        foreach (int id in addedTokens.Values) maxId = Math.Max(maxId, id);
        VocabSize = maxId + 1;

        _idToToken = new string[VocabSize];
        foreach (var (token, id) in vocab)
            if ((uint)id < (uint)VocabSize) _idToToken[id] = token;

        _rank = new Dictionary<(string, string), int>(merges.Count);
        for (int i = 0; i < merges.Count; i++) _rank.TryAdd(merges[i], i);

        (_byteToChar, _charToByte) = BuildByteCharMaps();
        _pattern = pretokenPattern;
        _pretokenizer = new Regex(pretokenPattern, RegexOptions.Compiled | RegexOptions.CultureInvariant);
    }

    /// <summary>
    /// Loads a tokenizer from <c>tokenizer.json</c> text. BOS/EOS can be given by content string or, more
    /// reliably, by id (e.g. <c>config.json</c>'s <c>eos_token_id</c>); an id overrides the content lookup.
    /// </summary>
    public static HfTokenizer FromTokenizerJson(
        string tokenizerJson, string? bosToken = null, string? eosToken = null,
        int bosTokenId = -1, int eosTokenId = -1, string? pretokenPattern = null)
    {
        using var doc = JsonDocument.Parse(tokenizerJson);
        var root = doc.RootElement;
        if (!root.TryGetProperty("model", out var model) || model.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("tokenizer.json has no 'model' object.");

        var vocab = ParseVocab(model);
        var merges = ParseMerges(model);
        var (specialIds, addedTokens, bosId, eosId) = ParseAddedTokens(root, vocab, bosToken, eosToken);
        if (bosTokenId >= 0) bosId = bosTokenId;
        if (eosTokenId >= 0) eosId = eosTokenId;
        return new HfTokenizer(vocab, merges, specialIds, addedTokens, bosId, eosId, pretokenPattern ?? Gpt2Pattern);
    }

    /// <summary>Serializes the tokenizer's full state (for embedding in a checkpoint); restore with <see cref="FromState"/>.</summary>
    public string ToStateJson()
    {
        var state = new State
        {
            Vocab = _vocab,
            Merges = _rank.OrderBy(kv => kv.Value).Select(kv => new[] { kv.Key.Item1, kv.Key.Item2 }).ToList(),
            SpecialIds = [.. _specialIds],
            AddedTokens = _addedTokens,
            BosId = BosId,
            EosId = EosId,
            Pattern = _pattern,
        };
        return JsonSerializer.Serialize(state);
    }

    /// <summary>Reconstructs a tokenizer saved with <see cref="ToStateJson"/>.</summary>
    public static HfTokenizer FromState(string stateJson)
    {
        var state = JsonSerializer.Deserialize<State>(stateJson) ?? throw new FormatException("invalid HfTokenizer state JSON.");
        var merges = state.Merges.Select(m => (m[0], m[1])).ToList();
        return new HfTokenizer(state.Vocab, merges, [.. state.SpecialIds], state.AddedTokens, state.BosId, state.EosId, state.Pattern);
    }

    private sealed class State
    {
        public Dictionary<string, int> Vocab { get; set; } = new();
        public List<string[]> Merges { get; set; } = [];
        public int[] SpecialIds { get; set; } = [];
        public Dictionary<string, int> AddedTokens { get; set; } = new();
        public int BosId { get; set; } = -1;
        public int EosId { get; set; } = -1;
        public string Pattern { get; set; } = Gpt2Pattern;
    }

    public IReadOnlyList<int> Encode(string text, bool addBos = false, bool addEos = false)
    {
        var ids = new List<int>();
        if (addBos && BosId >= 0) ids.Add(BosId);

        // Split out added/special tokens (e.g. "<|begin_of_text|>") first, emitting each as its single id; only
        // the spans between them go through byte-level BPE. This matches HF's AddedVocabulary pass.
        int pos = 0;
        if (_addedTokens_Regex is not null)
            foreach (Match m in _addedTokens_Regex.Matches(text))
            {
                if (m.Index > pos) EncodeSpan(text[pos..m.Index], ids);
                ids.Add(_addedTokens[m.Value]);
                pos = m.Index + m.Length;
            }
        if (pos < text.Length) EncodeSpan(text[pos..], ids);

        if (addEos && EosId >= 0) ids.Add(EosId);
        return ids;
    }

    // Byte-level BPE of a span containing no added tokens: pre-tokenize → bytes → mapped chars → merge → ids.
    private void EncodeSpan(string text, List<int> ids)
    {
        foreach (Match match in _pretokenizer.Matches(text))
        {
            var symbols = new List<string>();
            foreach (byte b in Utf8Strict.GetBytes(match.Value)) symbols.Add(_byteToChar[b].ToString());
            Merge(symbols);
            foreach (var symbol in symbols)
            {
                if (!_vocab.TryGetValue(symbol, out int id))
                    throw new InvalidDataException($"token '{symbol}' is not in the vocabulary (incompatible merges/vocab).");
                ids.Add(id);
            }
        }
    }

    public string Decode(IReadOnlyList<int> ids)
    {
        var bytes = new List<byte>();
        foreach (int id in ids)
        {
            if (_specialIds.Contains(id)) continue;
            if ((uint)id >= (uint)VocabSize || _idToToken[id] is not { } token) continue;
            foreach (char c in token)
                if (_charToByte.TryGetValue(c, out int b)) bytes.Add((byte)b);
        }
        return Encoding.UTF8.GetString(bytes.ToArray()); // lenient: partial streams degrade to U+FFFD
    }

    // Greedily applies the lowest-rank adjacent merge until none remain (HF BPE order).
    private void Merge(List<string> symbols)
    {
        while (symbols.Count >= 2)
        {
            int bestRank = int.MaxValue, bestPos = -1;
            for (int i = 0; i < symbols.Count - 1; i++)
                if (_rank.TryGetValue((symbols[i], symbols[i + 1]), out int r) && r < bestRank)
                {
                    bestRank = r;
                    bestPos = i;
                }
            if (bestPos < 0) break;
            symbols[bestPos] += symbols[bestPos + 1];
            symbols.RemoveAt(bestPos + 1);
        }
    }

    // The GPT-2 bytes_to_unicode table: printable byte ranges map to themselves, the rest to 256+n, giving a
    // reversible byte↔char mapping where every byte is a visible character.
    private static (Dictionary<int, string>, Dictionary<char, int>) BuildByteCharMaps()
    {
        var bytes = new List<int>();
        for (int i = '!'; i <= '~'; i++) bytes.Add(i);
        for (int i = '¡'; i <= '¬'; i++) bytes.Add(i);
        for (int i = '®'; i <= 'ÿ'; i++) bytes.Add(i);

        var chars = new List<int>(bytes);
        int n = 0;
        for (int b = 0; b < 256; b++)
            if (!bytes.Contains(b)) { bytes.Add(b); chars.Add(256 + n); n++; }

        var byteToChar = new Dictionary<int, string>(256);
        var charToByte = new Dictionary<char, int>(256);
        for (int i = 0; i < bytes.Count; i++)
        {
            byteToChar[bytes[i]] = ((char)chars[i]).ToString();
            charToByte[(char)chars[i]] = bytes[i];
        }
        return (byteToChar.ToDictionary(kv => kv.Key, kv => kv.Value), charToByte);
    }

    private static Dictionary<string, int> ParseVocab(JsonElement model)
    {
        if (!model.TryGetProperty("vocab", out var vocabEl) || vocabEl.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("tokenizer.json model has no 'vocab' object.");
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var entry in vocabEl.EnumerateObject())
            vocab[entry.Name] = entry.Value.GetInt32();
        return vocab;
    }

    private static List<(string, string)> ParseMerges(JsonElement model)
    {
        var merges = new List<(string, string)>();
        if (!model.TryGetProperty("merges", out var mergesEl) || mergesEl.ValueKind != JsonValueKind.Array)
            return merges;
        foreach (var merge in mergesEl.EnumerateArray())
        {
            // Two formats: "a b" (string) or ["a","b"] (array).
            if (merge.ValueKind == JsonValueKind.String)
            {
                string s = merge.GetString()!;
                int space = s.IndexOf(' ');
                if (space > 0) merges.Add((s[..space], s[(space + 1)..]));
            }
            else if (merge.ValueKind == JsonValueKind.Array && merge.GetArrayLength() == 2)
            {
                merges.Add((merge[0].GetString()!, merge[1].GetString()!));
            }
        }
        return merges;
    }

    private static (HashSet<int> SpecialIds, Dictionary<string, int> AddedTokens, int BosId, int EosId) ParseAddedTokens(
        JsonElement root, Dictionary<string, int> vocab, string? bosToken, string? eosToken)
    {
        var specialIds = new HashSet<int>();
        var addedTokens = new Dictionary<string, int>(StringComparer.Ordinal);
        int bosId = -1, eosId = -1;
        if (root.TryGetProperty("added_tokens", out var added) && added.ValueKind == JsonValueKind.Array)
            foreach (var token in added.EnumerateArray())
            {
                if (!token.TryGetProperty("id", out var idEl) || !idEl.TryGetInt32(out int id)) continue;
                bool special = token.TryGetProperty("special", out var sp) && sp.ValueKind == JsonValueKind.True;
                string? content = token.TryGetProperty("content", out var c) ? c.GetString() : null;
                if (special) specialIds.Add(id);
                if (content is not null) addedTokens[content] = id; // matched as a whole unit before BPE
                if (content == bosToken) bosId = id;
                if (content == eosToken) eosId = id;
            }
        if (bosToken is not null && bosId < 0 && vocab.TryGetValue(bosToken, out int b)) bosId = b;
        if (eosToken is not null && eosId < 0 && vocab.TryGetValue(eosToken, out int e)) eosId = e;
        return (specialIds, addedTokens, bosId, eosId);
    }
}
