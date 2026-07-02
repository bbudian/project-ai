using ProjectAI.Core;
using ProjectAI.Models;
using ProjectAI.Tokenizers;
using ProjectAI.Training;

// Discovers and lazily loads checkpoint models from a directory — the "built and trained models" the UI picks
// from. A model's name is its .ckpt filename without the extension. Loaded models are cached so switching back
// is instant. A requested name is resolved ONLY if it maps to a file directly inside the directory, which blocks
// path-traversal (e.g. "../secret") coming from an untrusted /generate request.
internal sealed class ModelRegistry
{
    private readonly IComputeBackend _backend;
    private readonly string _directory;
    // Keyed by the canonical resolved path (case-insensitive) so case-aliased names ("model"/"MODEL") don't
    // load the same checkpoint twice.
    private readonly Dictionary<string, LoadedModel> _cache = new(StringComparer.OrdinalIgnoreCase);
    // Tokenizer-only cache (no weights), guarded because /tokenize runs on request threads without the InferenceLock.
    private readonly Dictionary<string, ITokenizer> _tokenizers = new(StringComparer.OrdinalIgnoreCase);
    private readonly object _tokLock = new();

    public ModelRegistry(IComputeBackend backend, string directory)
    {
        _backend = backend;
        _directory = directory;
    }

    public string DirectoryPath => _directory;

    /// <summary>The available model names (sorted), re-scanned each call so newly trained models appear.</summary>
    public IReadOnlyList<string> List()
    {
        if (!Directory.Exists(_directory)) return [];
        return Directory.GetFiles(_directory, "*.ckpt")
            .Select(p => Path.GetFileNameWithoutExtension(p))
            .Where(n => !string.IsNullOrEmpty(n))
            .Select(n => n!)
            .OrderBy(n => n, StringComparer.Ordinal)
            .ToList();
    }

    // Descriptive info cache keyed by path; invalidated when the file changes (retrain writes a new mtime/length).
    private readonly Dictionary<string, ((DateTime Mtime, long Length) Fingerprint, ModelInfo Info)> _infos =
        new(StringComparer.OrdinalIgnoreCase);
    private readonly object _infoLock = new();

    /// <summary>
    /// The model catalog with metadata read from each checkpoint's header (no weights loaded): parameter count,
    /// layers, context, tokenizer kind, precision, step, and whether it's a chat-templated (instruct) model.
    /// Per-file failures degrade to a name-plus-error entry — /health must never fail because one file is bad.
    /// </summary>
    public IReadOnlyList<ModelInfo> ListInfos()
    {
        if (!Directory.Exists(_directory)) return [];
        var infos = new List<ModelInfo>();
        foreach (string path in Directory.GetFiles(_directory, "*.ckpt").OrderBy(p => p, StringComparer.Ordinal))
        {
            string name = Path.GetFileNameWithoutExtension(path);
            if (string.IsNullOrEmpty(name)) continue;
            var file = new FileInfo(path);
            var fingerprint = (file.LastWriteTimeUtc, file.Length);
            lock (_infoLock)
                if (_infos.TryGetValue(path, out var hit) && hit.Fingerprint == fingerprint)
                {
                    infos.Add(hit.Info);
                    continue;
                }
            var info = BuildInfo(name, path, file.Length);
            lock (_infoLock) _infos[path] = (fingerprint, info);
            infos.Add(info);
        }
        return infos;
    }

    private ModelInfo BuildInfo(string name, string path, long fileBytes)
    {
        try
        {
            var peek = Checkpointing.PeekInfo(path);
            var c = peek.Config;
            // Same probe ChatSession uses: an instruct model's tokenizer has the ChatML markers as single tokens.
            bool instruct = false;
            try { instruct = GetTokenizer(name) is { } tok && tok.Encode("<|im_start|>").Count == 1; }
            catch (Exception) { /* metadata-less tokenizer → not instruct */ }
            return new ModelInfo(name, CountParams(c), c.LayerCount, c.MaxSequenceLength, c.VocabSize,
                peek.TokenizerKind, peek.ComputeDType.ToString(), peek.Step, instruct, fileBytes, null);
        }
        catch (Exception e) // metadata-less/corrupt checkpoint: still listed so the user can see + delete it
        {
            return new ModelInfo(name, 0, 0, 0, 0, "", "", 0, false, fileBytes, e.Message);
        }
    }

    // Weight count from the architecture (tied LM head — the embedding is shared, so no separate head term).
    private static long CountParams(ProjectAI.Models.ModelConfig c)
    {
        long d = c.EmbeddingDim, ffn = c.FeedForwardHiddenDim, kvDim = (long)c.KvHeadCount * c.HeadDim;
        long perLayer = 2L * d * d      // Q and O projections
                      + 2L * d * kvDim  // K and V projections (GQA-sized)
                      + 3L * d * ffn    // SwiGLU gate/up/down
                      + 2L * d;         // the two RMSNorm weights
        return (long)c.VocabSize * d + c.LayerCount * perLayer + d; // embedding + layers + final norm
    }

    /// <summary>Resolves and loads a model by name; null if the name doesn't map to a checkpoint in the directory.</summary>
    public LoadedModel? Get(string name)
    {
        string? path = ResolvePath(name);
        if (path is null) return null; // unknown/invalid names never create a cache entry
        if (_cache.TryGetValue(path, out var cached)) return cached;

        var (model, config, tokenizer, step) = Checkpointing.LoadModel(path, _backend);
        var loaded = new LoadedModel(name, model, config, tokenizer, step);
        _cache[path] = loaded;
        return loaded;
    }

    /// <summary>
    /// Loads just the tokenizer for a model (no weights) — cheap even for multi-GB checkpoints. Null if the name
    /// doesn't map to a checkpoint in the directory. Thread-safe (called off the InferenceLock by /tokenize).
    /// </summary>
    public ITokenizer? GetTokenizer(string name)
    {
        string? path = ResolvePath(name);
        if (path is null) return null;
        lock (_tokLock)
        {
            if (_tokenizers.TryGetValue(path, out var cached)) return cached;
            var tok = Checkpointing.LoadTokenizer(path);
            _tokenizers[path] = tok;
            return tok;
        }
    }

    // Accept a name only if it's a plain file name (no separators, NUL, or other invalid chars) AND
    // "<dir>/<name>.ckpt" resolves to a file directly inside the models directory. Rejecting invalid chars first
    // makes path traversal structurally impossible and avoids Path.GetFullPath throwing on e.g. an embedded NUL.
    private string? ResolvePath(string name)
    {
        if (string.IsNullOrWhiteSpace(name) || name.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0) return null;
        string fullDir = Path.GetFullPath(_directory);
        string candidate = Path.GetFullPath(Path.Combine(_directory, name + ".ckpt"));
        bool insideDir = string.Equals(Path.GetDirectoryName(candidate), fullDir, StringComparison.OrdinalIgnoreCase);
        return insideDir && File.Exists(candidate) ? candidate : null;
    }
}

internal sealed record LoadedModel(string Name, LlamaModel Model, ModelConfig Config, ITokenizer Tokenizer, int Step);

/// <summary>One /health catalog entry. <see cref="Error"/> is set (and the numeric fields zero) when the
/// checkpoint's metadata couldn't be read.</summary>
internal sealed record ModelInfo(
    string Name, long Params, int Layers, int Ctx, int Vocab, string TokenizerKind, string Dtype, int Step,
    bool Instruct, long FileBytes, string? Error);
