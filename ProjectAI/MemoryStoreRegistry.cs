using ProjectAI.Memory;

// Resolves and caches one FileMemoryStore per (userId, storeId), each rooted at <root>/<userId>/<storeId>. Multi-client
// by construction (the locked-in decision): a user's memories live under their OWN partition and never cross into
// another user's store, so a poisoned or private fact can't leak between clients. Segment names are validated as plain
// path segments (no separators / "." / ".." / invalid chars), which makes traversal out of the memory root structurally
// impossible — an invalid name degrades to the inert NullMemoryStore rather than erroring. Loaded stores are cached
// (the in-memory index is built once), mirroring ModelRegistry.
internal sealed class MemoryStoreRegistry
{
    private readonly string _root;
    private readonly Dictionary<(string User, string Store), IMemoryStore> _cache = new();
    private readonly object _lock = new();

    public MemoryStoreRegistry(string rootDirectory) => _root = rootDirectory;

    public string RootDirectory => _root;

    /// <summary>
    /// The store for (<paramref name="user"/>, <paramref name="store"/>); empty/absent names fall back to "default".
    /// Returns the inert <see cref="NullMemoryStore"/> when a supplied name is unsafe, so a bad request quietly gets
    /// "no memory" instead of an error or a traversal.
    /// </summary>
    public IMemoryStore Resolve(string? user, string? store)
    {
        string? u = Segment(user);
        string? s = Segment(store);
        if (u is null || s is null) return NullMemoryStore.Instance;

        lock (_lock)
        {
            var key = (u, s);
            if (_cache.TryGetValue(key, out var cached)) return cached;

            string fullRoot = Path.GetFullPath(_root);
            string dir = Path.GetFullPath(Path.Combine(_root, u, s));
            bool inside = string.Equals(dir, fullRoot, StringComparison.OrdinalIgnoreCase)
                || dir.StartsWith(fullRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase);
            IMemoryStore resolved;
            if (!inside) resolved = NullMemoryStore.Instance;
            else { try { resolved = new FileMemoryStore(dir, $"{u}/{s}"); } catch { resolved = NullMemoryStore.Instance; } }

            _cache[key] = resolved;
            return resolved;
        }
    }

    // A valid single path segment, or null if unsafe. Empty/whitespace → the "default" partition.
    private static string? Segment(string? name)
    {
        if (string.IsNullOrWhiteSpace(name)) return "default";
        if (name is "." or ".." ) return null;
        if (name.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0) return null;
        if (name != Path.GetFileName(name)) return null; // rejects any path separator
        return name;
    }
}
