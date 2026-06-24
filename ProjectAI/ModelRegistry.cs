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
