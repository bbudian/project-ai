using Godot;

// Composition root for the ProjectAI client. Builds the Claude-desktop-style shell — sidebar | (header,
// content) — where the content area toggles between a Chat view (transcript + composer) and a Train view
// (TrainPanel). It holds no view logic and no HTTP: each concern lives in its own single-responsibility component
// (Sidebar, Transcript, Composer, TrainPanel, ApiClient), so adding a feature means touching one piece. Main just
// connects their events.
public partial class Main : Control
{
    private IApiClient _api;
    private Sidebar _sidebar;
    private Transcript _transcript;
    private Composer _composer;
    private TrainPanel _trainPanel;
    private Control _chatView;
    private Control _trainView;
    private Button _modeChat;
    private Button _modeTrain;
    private Label _headerTitle;
    private Godot.Timer _pollTimer;
    private TurnCard _activeTurn;
    private ChatSocket _chat;
    private string _sessionModel = "", _sessionBackend = "";
    private bool _chatBusy;
    private bool _resetSession;
    private string _chatTitle = "New chat"; // the chat header's source of truth, so switching modes doesn't lose it

    public override void _Ready()
    {
        var apiNode = new ApiClient();
        AddChild(apiNode);
        _api = apiNode;

        // Polls /train/status while a job runs; only fires a request when the client is otherwise idle.
        _pollTimer = new Godot.Timer { WaitTime = 1.5, OneShot = false, Autostart = false };
        _pollTimer.Timeout += () => { if (!_api.Busy) _api.CheckTrainStatus(); };
        AddChild(_pollTimer);

        _chat = new ChatSocket();
        AddChild(_chat);

        BuildLayout();

        _api.HealthReceived += OnHealth;
        _chat.Token += delta => { _activeTurn?.Append(delta); _transcript.ScrollToBottom(); };
        _chat.Sources += sources => { _activeTurn?.SetSources(sources); _transcript.ScrollToBottom(); };
        _chat.Done += OnChatDone;
        _chat.ChatError += OnChatError;
        _chat.Closed += OnChatClosed;
        _api.TrainStarted += OnTrainStarted;
        _api.TrainStatusReceived += OnTrainStatus;
        _sidebar.NewChatRequested += OnNewChat;
        _sidebar.RecentSelected += prompt => _composer.SetPrompt(prompt);
        _sidebar.ServerUrlChanged += url => _api.BaseUrl = url;
        _sidebar.CheckRequested += () => { _api.BaseUrl = _sidebar.ServerUrl; _api.CheckHealth(); };
        _composer.Submitted += OnSubmit;
        _composer.Canceled += OnCancel;
        _composer.FontSizeChanged += size => _transcript.SetFontSize(size);
        _trainPanel.TrainRequested += OnTrainRequested;

        SwitchMode(train: false);
        _api.BaseUrl = _sidebar.ServerUrl;
        _api.CheckHealth(); // attempt to connect + populate the model/backend pickers on launch
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

        // Chat view: transcript (fills) + composer (fixed).
        _chatView = new VBoxContainer { SizeFlagsVertical = SizeFlags.ExpandFill };
        _chatView.AddThemeConstantOverride("separation", 0);
        _chatView.AddChild(Pad(_transcript = new Transcript(), left: 24, right: 24, top: 8, bottom: 8, expand: true));
        _chatView.AddChild(Pad(_composer = new Composer(), left: 16, right: 16, top: 8, bottom: 16, expand: false));
        main.AddChild(_chatView);

        // Train view: the TrainPanel pinned to the top, with a spacer filling the rest.
        _trainView = new VBoxContainer { SizeFlagsVertical = SizeFlags.ExpandFill };
        _trainView.AddChild(Pad(_trainPanel = new TrainPanel(), left: 24, right: 24, top: 16, bottom: 8, expand: false));
        _trainView.AddChild(new Control { SizeFlagsVertical = SizeFlags.ExpandFill });
        main.AddChild(_trainView);
    }

    private PanelContainer BuildHeader()
    {
        var header = new PanelContainer();
        Palette.StylePanel(header, Palette.AppBg, pad: 14);
        var row = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _headerTitle = Palette.Heading("New chat", 16);
        row.AddChild(_headerTitle);
        row.AddChild(new Control { SizeFlagsHorizontal = SizeFlags.ExpandFill }); // push the toggle to the right

        _modeChat = Palette.GhostButton("Chat");
        _modeChat.Pressed += () => SwitchMode(train: false);
        row.AddChild(_modeChat);
        _modeTrain = Palette.GhostButton("Train");
        _modeTrain.Pressed += () => SwitchMode(train: true);
        row.AddChild(_modeTrain);

        header.AddChild(row);
        return header;
    }

    private void SwitchMode(bool train)
    {
        _trainView.Visible = train;
        _chatView.Visible = !train;
        _modeTrain.Disabled = train;   // disabling the active mode's button marks it as selected
        _modeChat.Disabled = !train;
        _headerTitle.Text = train ? "Train a model" : _chatTitle;
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
        if (_chatBusy) return;
        _activeTurn = _transcript.Begin(request.Prompt);
        _sidebar.AddRecent(request.Prompt);
        _chatTitle = Format.Ellipsize(request.Prompt, 40);
        _headerTitle.Text = _chatTitle;
        _composer.SetBusy(true);
        _chatBusy = true;

        // One persistent /chat connection keeps the server's KV cache warm across turns. (Re)start the session when
        // first connecting, when the model/backend changed, or after "New chat" cleared the conversation.
        if (!_chat.IsActive) { _chat.Connect(_sidebar.ServerUrl); _resetSession = true; }
        if (_resetSession || request.Model != _sessionModel || request.Backend != _sessionBackend)
        {
            _chat.SendStart(request.Model, request.Backend);
            _sessionModel = request.Model;
            _sessionBackend = request.Backend;
            _resetSession = false;
        }
        _chat.SendMessage(request);
    }

    // Stop button: ask the server to halt generation. It stops at the next token and replies with a "done"
    // (stop=canceled), which flows through OnChatDone to keep the partial reply and re-enable the composer.
    private void OnCancel()
    {
        if (_chatBusy) _chat.Cancel();
    }

    private void OnChatDone(string stop)
    {
        _activeTurn?.Complete(stop);
        _transcript.ScrollToBottom();
        _activeTurn = null;
        _chatBusy = false;
        _composer.SetBusy(false);
    }

    private void OnChatError(string error)
    {
        _activeTurn?.Fail(error);
        _activeTurn = null;
        _chatBusy = false;
        _composer.SetBusy(false);
    }

    private void OnChatClosed()
    {
        // Connection dropped: fail any in-flight turn and force a fresh session (new warm cache) on the next message.
        if (_chatBusy)
        {
            _activeTurn?.Fail("Chat connection closed — is `projectai serve` running?");
            _activeTurn = null;
            _chatBusy = false;
            _composer.SetBusy(false);
        }
        _resetSession = true;
    }

    private void OnTrainRequested(TrainRequest request)
    {
        _api.BaseUrl = _sidebar.ServerUrl;
        _trainPanel.SetBusy(true);
        _api.StartTraining(request);
    }

    private void OnTrainStarted(TrainStartResult result)
    {
        if (!result.Ok)
        {
            _trainPanel.UpdateStatus(new TrainStatus("error", "", 0, 0, 0f, result.Error));
            _trainPanel.SetBusy(false);
            return;
        }
        _pollTimer.Start(); // begin polling progress
    }

    private void OnTrainStatus(TrainStatus status)
    {
        _trainPanel.UpdateStatus(status);
        // Once we're polling, "idle" means the job vanished (e.g. the server restarted) — treat it as terminal too,
        // otherwise the panel stays "Training…" and polls forever.
        if (status.State is "done" or "error" or "idle")
        {
            _pollTimer.Stop();
            _trainPanel.SetBusy(false);
            _api.CheckHealth(); // refresh the model list so a finished model appears in the chat picker
        }
    }

    private void OnHealth(HealthResult health)
    {
        if (!health.Ok)
        {
            _sidebar.SetStatus(health.Error, error: true);
            return;
        }
        _composer.SetModels(health.Models, health.Default);
        _composer.SetBackends(health.Backends, health.DefaultBackend);
        _trainPanel.SetBackends(health.Backends, health.DefaultBackend);
        _trainPanel.SetSizes(health.Sizes);
        int backendCount = 0;
        foreach (var b in health.Backends) if (b.Available) backendCount++;
        _sidebar.SetStatus(
            $"Connected ✓  —  {health.Models.Length} model{(health.Models.Length == 1 ? "" : "s")}, {backendCount} backend{(backendCount == 1 ? "" : "s")}",
            error: false);
    }

    private void OnNewChat()
    {
        _transcript.Clear();
        _chatTitle = "New chat";
        _resetSession = true; // next message starts a fresh server session (drops the warm cache)
        SwitchMode(train: false); // also resets the header to _chatTitle
    }
}
