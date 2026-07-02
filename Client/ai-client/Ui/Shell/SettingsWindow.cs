using Godot;

// The Settings modal (docs/CLIENT_DESIGN.md: a Window, NOT a routed tab — it must overlay Chat mid-conversation
// without losing state). One scrolling pane grouped by section (per §4.1's guidance over a near-empty tab
// container): App (client-local, saved to prefs), Memory budgets (server-side, GET/PUT /config), and Web search
// (the write-only Tavily secret — the field clears on save and only masked status ever comes back).
public partial class SettingsWindow : Window
{
    private readonly AppState _state;
    private readonly IApiClient _api;

    private LineEdit _serverUrl;
    private SpinBox _bridgeCards, _bridgeBudget, _recallHits, _recallBudget;
    private Label _memoryStatus;
    private Label _secretState;
    private LineEdit _secretInput;
    private Label _secretStatus;

    public SettingsWindow(AppState state, IApiClient api)
    {
        _state = state;
        _api = api;
        Title = "Settings";
        Size = new Vector2I(520, 620);
        Visible = false;
        CloseRequested += Hide;
    }

    public override void _Ready()
    {
        var background = new PanelContainer();
        background.SetAnchorsPreset(Control.LayoutPreset.FullRect);
        Palette.StylePanel(background, Palette.AppBg);
        AddChild(background);

        var scroll = new ScrollContainer { HorizontalScrollMode = ScrollContainer.ScrollMode.Disabled };
        background.AddChild(scroll);

        var column = new VBoxContainer { SizeFlagsHorizontal = Control.SizeFlags.ExpandFill };
        column.AddThemeConstantOverride("separation", Palette.Space.Md);
        scroll.AddChild(Palette.Pad(column, left: 16, right: 16, top: 16, bottom: 16, expand: false));

        column.AddChild(BuildAppSection());
        column.AddChild(BuildMemorySection());
        column.AddChild(BuildSecretSection());

        _api.ConfigReceived += OnConfig;
        _api.SecretUpdated += OnSecret;
    }

    /// <summary>Opens centered and refreshes the server-side sections.</summary>
    public void Open()
    {
        _serverUrl.Text = _state.ServerUrl;
        _memoryStatus.Text = "loading…";
        _api.FetchConfig();
        PopupCentered();
    }

    // ---- App (client-local prefs) -------------------------------------------------------------------------------

    private Control BuildAppSection()
    {
        var body = new VBoxContainer { SizeFlagsHorizontal = Control.SizeFlags.ExpandFill };
        body.AddThemeConstantOverride("separation", Palette.Space.Sm);

        _serverUrl = new LineEdit { SizeFlagsHorizontal = Control.SizeFlags.ExpandFill };
        _serverUrl.TextChanged += url => _state.SetServerUrl(url); // same live wiring as the connection panel
        body.AddChild(Palette.Field("Server URL", _serverUrl));

        // Local-server lifecycle: empty fields mean "auto-discover from the repo layout".
        var exe = new LineEdit
        {
            Text = _state.Prefs.ServerExePath,
            PlaceholderText = "auto — <repo>/ProjectAI/bin/…/projectai.exe",
            SizeFlagsHorizontal = Control.SizeFlags.ExpandFill,
        };
        exe.TextChanged += v => _state.MutatePrefs(p => p.ServerExePath = v.Trim());
        body.AddChild(Palette.Field("Server exe", exe));

        var modelsDir = new LineEdit
        {
            Text = _state.Prefs.ServerModelsDir,
            PlaceholderText = "auto — <repo>/checkpoints",
            SizeFlagsHorizontal = Control.SizeFlags.ExpandFill,
        };
        modelsDir.TextChanged += v => _state.MutatePrefs(p => p.ServerModelsDir = v.Trim());
        body.AddChild(Palette.Field("Models dir", modelsDir));

        var extraArgs = new LineEdit { Text = _state.Prefs.ServerExtraArgs, SizeFlagsHorizontal = Control.SizeFlags.ExpandFill };
        extraArgs.TextChanged += v => _state.MutatePrefs(p => p.ServerExtraArgs = v);
        body.AddChild(Palette.Field("Server args", extraArgs));

        var autoStart = new CheckButton { Text = "Start the server automatically when unreachable", ButtonPressed = _state.Prefs.AutoStartServer };
        autoStart.AddThemeColorOverride("font_color", Palette.Text);
        autoStart.Toggled += on => _state.MutatePrefs(p => p.AutoStartServer = on);
        body.AddChild(autoStart);

        var stopOnExit = new CheckButton { Text = "Stop a server this app started when it closes", ButtonPressed = _state.Prefs.StopServerOnExit };
        stopOnExit.AddThemeColorOverride("font_color", Palette.Text);
        stopOnExit.Toggled += on => _state.MutatePrefs(p => p.StopServerOnExit = on);
        body.AddChild(stopOnExit);

        body.AddChild(Palette.Heading(
            "Model, backend, sampling, and text size live in the chat composer; all of it persists locally.",
            Palette.Type.Caption, Palette.Muted));
        return Palette.Card(body, "App (this machine)");
    }

    // ---- Memory budgets (server-side) ---------------------------------------------------------------------------

    private Control BuildMemorySection()
    {
        var body = new VBoxContainer { SizeFlagsHorizontal = Control.SizeFlags.ExpandFill };
        body.AddThemeConstantOverride("separation", Palette.Space.Sm);

        _bridgeCards = Spin(0, 200);
        _bridgeBudget = Spin(0, 100_000);
        _recallHits = Spin(0, 64);
        _recallBudget = Spin(0, 100_000);
        body.AddChild(Palette.Field("Bridge cards", _bridgeCards));
        body.AddChild(Palette.Field("Bridge budget", _bridgeBudget));
        body.AddChild(Palette.Field("Recall hits", _recallHits));
        body.AddChild(Palette.Field("Recall budget", _recallBudget));
        body.AddChild(Palette.Heading(
            "How much memory each turn may inject (budgets are ~tokens). Applies to every chat immediately.",
            Palette.Type.Caption, Palette.Muted));

        var save = Palette.PrimaryButton("Save memory budgets");
        save.Pressed += () =>
        {
            _memoryStatus.Text = "saving…";
            _api.SaveMemoryBudgets((int)_bridgeCards.Value, (int)_bridgeBudget.Value,
                (int)_recallHits.Value, (int)_recallBudget.Value);
        };
        body.AddChild(save);
        _memoryStatus = Palette.Heading("", Palette.Type.Caption, Palette.Muted);
        body.AddChild(_memoryStatus);
        return Palette.Card(body, "Memory injection (server)");
    }

    private static SpinBox Spin(double min, double max) =>
        new() { MinValue = min, MaxValue = max, Step = 1, CustomMinimumSize = new Vector2(120, 0) };

    private void OnConfig(ConfigInfo config)
    {
        if (!config.Ok)
        {
            _memoryStatus.AddThemeColorOverride("font_color", Palette.Bad);
            _memoryStatus.Text = $"Error: {config.Error}";
            return;
        }
        _bridgeCards.Value = config.BridgeCards;
        _bridgeBudget.Value = config.BridgeBudget;
        _recallHits.Value = config.RecallHits;
        _recallBudget.Value = config.RecallBudget;
        _memoryStatus.AddThemeColorOverride("font_color", Palette.Good);
        _memoryStatus.Text = "Loaded from the server.";
        foreach (var secret in config.Secrets)
            if (secret.Key == "tavily") RenderSecretState(secret);
    }

    // ---- Web search secret (write-only) -------------------------------------------------------------------------

    private Control BuildSecretSection()
    {
        var body = new VBoxContainer { SizeFlagsHorizontal = Control.SizeFlags.ExpandFill };
        body.AddThemeConstantOverride("separation", Palette.Space.Sm);

        _secretState = Palette.Heading("…", Palette.Type.Label, Palette.Muted);
        body.AddChild(_secretState);

        _secretInput = new LineEdit
        {
            Secret = true, // masked input
            PlaceholderText = "Paste a Tavily API key (free at tavily.com)…",
            SizeFlagsHorizontal = Control.SizeFlags.ExpandFill,
        };
        body.AddChild(Palette.Field("New key", _secretInput));

        var buttons = new HBoxContainer();
        buttons.AddThemeConstantOverride("separation", Palette.Space.Sm);
        var save = Palette.PrimaryButton("Save key");
        save.Pressed += () =>
        {
            if (string.IsNullOrWhiteSpace(_secretInput.Text)) return;
            _secretStatus.Text = "saving…";
            _api.SaveSecret("tavily", _secretInput.Text.Trim());
            _secretInput.Text = ""; // write-only: the raw key leaves the UI immediately, only masked status returns
        };
        buttons.AddChild(save);
        var clear = Palette.GhostButton("Clear stored key", Palette.Type.Label);
        clear.Pressed += () => { _secretStatus.Text = "clearing…"; _api.ClearSecret("tavily"); };
        buttons.AddChild(clear);
        body.AddChild(buttons);

        _secretStatus = Palette.Heading("", Palette.Type.Caption, Palette.Muted);
        body.AddChild(_secretStatus);
        body.AddChild(Palette.Heading(
            "The key is stored server-side (config/secrets.json, ACL-locked, git-ignored) and is never sent back — only presence and a …last-4 hint. An environment variable (TAVILY_API_KEY) takes precedence.",
            Palette.Type.Caption, Palette.Muted));
        return Palette.Card(body, "Web search (Tavily)");
    }

    private void OnSecret(SecretStatus status)
    {
        if (!status.Ok)
        {
            _secretStatus.AddThemeColorOverride("font_color", Palette.Bad);
            _secretStatus.Text = $"Error: {status.Error}";
            return;
        }
        _secretStatus.AddThemeColorOverride("font_color", Palette.Good);
        _secretStatus.Text = status.Set ? "Saved." : "Cleared.";
        RenderSecretState(status);
    }

    private void RenderSecretState(SecretStatus status)
    {
        _secretState.Text = status.Set
            ? $"Configured ({status.Hint}, from {status.Source})"
            : "Not configured — web research is unavailable until a key is set.";
        _secretState.AddThemeColorOverride("font_color", status.Set ? Palette.Good : Palette.Muted);
    }
}
