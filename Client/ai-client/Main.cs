using Godot;

// Composition root for the ProjectAI client — wiring only. Builds the shell (NavRail | ViewHost), constructs the
// views and the shared AppState, and connects the API client's events to the store. Every behavior lives in a
// single-responsibility component: views own their logic and read shared state from AppState (never each other),
// the rail and host own navigation, and the connection panel owns the server field. Adding a destination is one
// Register + one AddDestination call here.
public partial class Main : Control
{
    private IApiClient _api;
    private AppState _state;
    private ServerController _server;
    private Godot.Timer _prefsSaveTimer;
    private bool _autoStartTried;

    public override void _Ready()
    {
        var apiNode = new ApiClient();
        AddChild(apiNode);
        _api = apiNode;

        _state = new AppState(PrefsStore.Load()); // prefs first, so every view seeds from them
        _api.BaseUrl = _state.ServerUrl;

        // Persist preference edits debounced, so typing in the URL field doesn't write a file per keystroke.
        _prefsSaveTimer = new Godot.Timer { WaitTime = 0.75, OneShot = true };
        _prefsSaveTimer.Timeout += () => PrefsStore.Save(_state.Prefs);
        AddChild(_prefsSaveTimer);
        _state.PrefsChanged += () => { _prefsSaveTimer.Stop(); _prefsSaveTimer.Start(); };

        // API results flow into the store; views subscribe to the store, never to each other.
        _api.HealthReceived += health => _state.SetHealth(health);
        _api.TrainStatusReceived += status => _state.SetTrainStatus(status);
        _state.ServerUrlChanged += () => _api.BaseUrl = _state.ServerUrl;

        BuildShell();

        _api.CheckHealth(); // attempt to connect + populate the model/backend pickers on launch
    }

    public override void _Notification(int what)
    {
        if (what != NotificationWMCloseRequest) return;
        _server?.OnAppClosing();                          // a server we spawned dies with the app (pref-controlled)
        if (_state != null) PrefsStore.Save(_state.Prefs); // flush a pending debounced save so the last edit isn't lost
    }

    private void BuildShell()
    {
        SetAnchorsPreset(LayoutPreset.FullRect);

        var background = new PanelContainer();
        background.SetAnchorsPreset(LayoutPreset.FullRect);
        Palette.StylePanel(background, Palette.AppBg);
        AddChild(background);

        var shell = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        shell.AddThemeConstantOverride("separation", 0);
        background.AddChild(shell);

        var rail = new NavRail();
        shell.AddChild(rail);

        var connection = new ConnectionPanel(_state);
        connection.UrlEdited += url => _state.SetServerUrl(url);
        connection.CheckRequested += () => _api.CheckHealth();
        rail.AddFooter(connection);

        // The app can run the server itself: Start spawns `projectai serve`, the controller polls /health until it
        // answers, and the panel narrates the lifecycle. Auto-start (pref) kicks in on the first failed health probe.
        _server = new ServerController(_state, _api);
        AddChild(_server);
        connection.StartServerRequested += _server.Start;
        connection.StopServerRequested += _server.Stop;
        _server.Changed += () => connection.SetServerState(_server.Starting, _server.OwnsRunningServer, _server.Error);
        _state.HealthChanged += () =>
        {
            if (_state.Health is { Ok: false } && _state.Prefs.AutoStartServer && !_autoStartTried
                && !_server.Starting && !_server.OwnsRunningServer)
            {
                _autoStartTried = true; // once per launch — a dead server shouldn't respawn in a loop
                _server.Start();
            }
        };

        var settings = new SettingsWindow(_state, _api);
        AddChild(settings);
        rail.SettingsRequested += settings.Open;

        var host = new ViewHost();
        shell.AddChild(host);

        host.Register(ViewIds.Chat, new ChatView(_state));
        host.Register(ViewIds.Models, new ModelsView(_state, _api));
        host.Register(ViewIds.Bench, new BenchmarkView(_state, _api));
        host.Register(ViewIds.Memory, new MemoryView(_state, _api)); // live now that the server ships /memory reads
        rail.AddDestination(ViewIds.Chat, "💬   Chat");
        rail.AddDestination(ViewIds.Models, "◫   Models");
        rail.AddDestination(ViewIds.Bench, "📊   Benchmark");
        rail.AddDestination(ViewIds.Memory, "🧠   Memory");

        rail.Navigated += host.Show;
        _state.NavigateRequested += host.Show; // cross-view jumps (e.g. a model card's "Chat with")
        host.Shown += rail.SetActive;
        host.Shown += id => _state.MutatePrefs(p => p.LastView = id); // reopen where the user left off
        host.Show(host.Has(_state.Prefs.LastView) ? _state.Prefs.LastView : ViewIds.Chat);
    }
}
