using ProjectAI.Core;

// Holds the per-backend model registries for the server. The backend chosen at startup (--backend) is seeded;
// any OTHER backend the client picks is created lazily on first use and cached, so switching CPU<->GPU just
// reloads the model on the new device (each backend keeps its own model cache). The composition root owns the
// seeded backend; this registry owns — and disposes — the ones it creates here.
internal sealed class ComputeRegistry : IDisposable
{
    private readonly string _modelsDirectory;
    private readonly Dictionary<string, Entry> _byId = new(StringComparer.Ordinal);
    private readonly object _lock = new(); // guards _byId: requests are now handled on multiple threads

    private sealed record Entry(IComputeBackend Backend, ModelRegistry Models, bool Owned);

    public ComputeRegistry(string modelsDirectory, IComputeBackend defaultBackend, string defaultId)
    {
        _modelsDirectory = modelsDirectory;
        DefaultId = defaultId;
        _byId[defaultId] = new Entry(defaultBackend, new ModelRegistry(defaultBackend, modelsDirectory), Owned: false);
        AvailableBackends = Backends.Available(); // probed once at startup; availability is static for a run
    }

    public string DefaultId { get; }

    /// <summary>The directory trained checkpoints are written to and served from.</summary>
    public string ModelsDirectory => _modelsDirectory;

    /// <summary>The compute options offered to the client, tagged with which actually work on this machine.</summary>
    public IReadOnlyList<BackendStatus> AvailableBackends { get; }

    /// <summary>The available model names (backend-agnostic — just a directory scan via the default registry).</summary>
    public IReadOnlyList<string> ListModels() => Resolve(DefaultId).Models.List();

    /// <summary>The enriched model catalog (metadata-only reads; per-file failures degrade gracefully).</summary>
    public IReadOnlyList<ModelInfo> ListModelInfos() => Resolve(DefaultId).Models.ListInfos();

    /// <summary>
    /// Resolves the (backend, models) pair for an id, creating + caching the backend on first use.
    /// Throws if the id is unknown or the backend can't start on this machine.
    /// </summary>
    public (IComputeBackend Backend, ModelRegistry Models) Resolve(string id)
    {
        lock (_lock)
            if (_byId.TryGetValue(id, out var entry)) return (entry.Backend, entry.Models);

        // Create the backend OUTSIDE the lock: device init (e.g. CUDA) can be slow, and holding the lock would
        // stall a concurrent /health (which Resolves the default backend just to list models).
        var backend = Backends.Create(id); // throws if unavailable
        var models = new ModelRegistry(backend, _modelsDirectory);

        lock (_lock)
        {
            if (_byId.TryGetValue(id, out var existing)) // another thread won the race while we were creating
            {
                backend.Dispose();
                return (existing.Backend, existing.Models);
            }
            _byId[id] = new Entry(backend, models, Owned: true);
            return (backend, models);
        }
    }

    public void Dispose()
    {
        lock (_lock)
            foreach (var entry in _byId.Values)
                if (entry.Owned) entry.Backend.Dispose();
    }
}
