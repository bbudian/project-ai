using System;

// The shared client store (docs/CLIENT_DESIGN.md §4): the one place cross-view state lives, so views never
// reference each other — a subsystem that changes shared state calls a named command here, the command raises
// exactly one event, and every interested view refreshes itself. Plain C# events (not Godot signals) keep one
// eventing paradigm for state; Godot signals stay for UI controls. Passed by constructor, never an autoload.
public sealed class AppState
{
    public AppState(ClientPrefs prefs)
    {
        Prefs = prefs;
        ServerUrl = prefs.ServerUrl;
    }

    /// <summary>Client-local preferences (persisted by Main on PrefsChanged). Mutate only via <see cref="MutatePrefs"/>.</summary>
    public ClientPrefs Prefs { get; }

    /// <summary>Latest /health result (null until the first response). Views must check <c>Ok</c> before reading.</summary>
    public HealthResult Health { get; private set; }

    public string ServerUrl { get; private set; }

    /// <summary>The model/backend the user last chose in the chat composer — lets a future Models view preselect.</summary>
    public string SelectedModel { get; private set; } = "";
    public string SelectedBackend { get; private set; } = "";

    /// <summary>Latest /train/status (null until a poll lands). "Jobs" so a benchmark job can ride the same event later.</summary>
    public TrainStatus TrainStatus { get; private set; }

    public event Action HealthChanged;
    public event Action ServerUrlChanged;
    public event Action SelectionChanged;
    public event Action JobsChanged;
    public event Action PrefsChanged;

    public void SetHealth(HealthResult health)
    {
        Health = health;
        HealthChanged?.Invoke();
    }

    public void SetServerUrl(string url)
    {
        if (url == ServerUrl) return;
        ServerUrl = url;
        Prefs.ServerUrl = url;
        ServerUrlChanged?.Invoke();
        PrefsChanged?.Invoke();
    }

    public void SetSelection(string model, string backend)
    {
        if (model == SelectedModel && backend == SelectedBackend) return;
        SelectedModel = model;
        SelectedBackend = backend;
        // The last choice becomes this machine's default, so the pickers come back the same way next launch.
        Prefs.DefaultModel = model;
        Prefs.DefaultBackend = backend;
        SelectionChanged?.Invoke();
        PrefsChanged?.Invoke();
    }

    /// <summary>Applies a preference edit and notifies (Main persists on PrefsChanged, debounced).</summary>
    public void MutatePrefs(Action<ClientPrefs> mutate)
    {
        mutate(Prefs);
        PrefsChanged?.Invoke();
    }

    public void SetTrainStatus(TrainStatus status)
    {
        TrainStatus = status;
        JobsChanged?.Invoke();
    }
}
