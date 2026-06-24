using Godot;

// Composition root for the ProjectAI client. Builds the Claude-desktop-style shell — sidebar | (header,
// transcript, composer) — and wires the components to the IApiClient. It holds no view logic and no HTTP: each
// concern lives in its own single-responsibility component (Sidebar, Transcript, Composer, ApiClient), so adding
// a feature means touching one piece, not this file. Main just connects their events.
public partial class Main : Control
{
    private IApiClient _api;
    private Sidebar _sidebar;
    private Transcript _transcript;
    private Composer _composer;
    private Label _headerTitle;
    private TurnCard _activeTurn;

    public override void _Ready()
    {
        var apiNode = new ApiClient();
        AddChild(apiNode);
        _api = apiNode;

        BuildLayout();

        _api.HealthReceived += OnHealth;
        _api.GenerationReceived += OnGeneration;
        _sidebar.NewChatRequested += OnNewChat;
        _sidebar.RecentSelected += prompt => _composer.SetPrompt(prompt);
        _sidebar.ServerUrlChanged += url => _api.BaseUrl = url;
        _sidebar.CheckRequested += () => { _api.BaseUrl = _sidebar.ServerUrl; _api.CheckHealth(); };
        _composer.Submitted += OnSubmit;

        _api.BaseUrl = _sidebar.ServerUrl;
        _api.CheckHealth(); // attempt to connect + populate the model picker on launch
    }

    private void BuildLayout()
    {
        SetAnchorsPreset(LayoutPreset.FullRect);

        var background = new PanelContainer();
        background.SetAnchorsPreset(LayoutPreset.FullRect);
        Palette.StylePanel(background, Palette.AppBg);
        AddChild(background);

        var shell = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        shell.AddThemeConstantOverride("separation", 0);
        background.AddChild(shell);

        _sidebar = new Sidebar();
        shell.AddChild(_sidebar);

        var main = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        main.AddThemeConstantOverride("separation", 0);
        shell.AddChild(main);

        main.AddChild(BuildHeader());
        main.AddChild(Pad(_transcript = new Transcript(), left: 24, right: 24, top: 8, bottom: 8, expand: true));
        main.AddChild(Pad(_composer = new Composer(), left: 16, right: 16, top: 8, bottom: 16, expand: false));
    }

    private PanelContainer BuildHeader()
    {
        var header = new PanelContainer();
        Palette.StylePanel(header, Palette.AppBg, pad: 14);
        var row = new HBoxContainer();
        _headerTitle = Palette.Heading("New chat", 16);
        row.AddChild(_headerTitle);
        header.AddChild(row);
        return header;
    }

    // Wraps a control in a MarginContainer so spacing lives here, not inside each component (DRY).
    private static MarginContainer Pad(Control child, int left, int right, int top, int bottom, bool expand)
    {
        var margin = new MarginContainer();
        if (expand) margin.SizeFlagsVertical = SizeFlags.ExpandFill;
        margin.AddThemeConstantOverride("margin_left", left);
        margin.AddThemeConstantOverride("margin_right", right);
        margin.AddThemeConstantOverride("margin_top", top);
        margin.AddThemeConstantOverride("margin_bottom", bottom);
        margin.AddChild(child);
        return margin;
    }

    private void OnSubmit(GenerateRequest request)
    {
        if (_api.Busy) return;
        _api.BaseUrl = _sidebar.ServerUrl;
        _activeTurn = _transcript.Begin(request.Prompt);
        _sidebar.AddRecent(request.Prompt);
        _headerTitle.Text = Title(request.Prompt);
        _composer.SetBusy(true);
        _api.Generate(request);
    }

    private void OnGeneration(GenerateResult result)
    {
        _composer.SetBusy(false);
        if (_activeTurn == null) return;
        if (result.Ok) _activeTurn.Resolve(result.Text);
        else _activeTurn.Fail(result.Error);
        _transcript.ScrollToBottom();
        _activeTurn = null;
    }

    private void OnHealth(HealthResult health)
    {
        if (!health.Ok)
        {
            _sidebar.SetStatus(health.Error, error: true);
            return;
        }
        _composer.SetModels(health.Models, health.Default);
        _sidebar.SetStatus($"Connected ✓  —  {health.Models.Length} model{(health.Models.Length == 1 ? "" : "s")}", error: false);
    }

    private void OnNewChat()
    {
        _transcript.Clear();
        _headerTitle.Text = "New chat";
    }

    private static string Title(string prompt)
    {
        string text = prompt.Replace("\n", " ").Trim();
        return text.Length > 40 ? text[..40] + "…" : text;
    }
}
