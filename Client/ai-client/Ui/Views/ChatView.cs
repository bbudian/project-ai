using Godot;

// The Chat destination: recents context panel | (title header, transcript, composer). Owns the chat session
// state machine and its streaming transport (ChatSocket) — moved out of Main so the shell stays wiring-only.
// The socket and transcript live as long as the app: switching destinations hides this view without dropping
// the server's warm KV cache or the conversation on screen.
public partial class ChatView : HBoxContainer, IView
{
    private readonly AppState _state;

    private RecentsPanel _recents;
    private Transcript _transcript;
    private Composer _composer;
    private Label _title;
    private Label _meter;      // right side of the header: context usage + last-turn stats
    private Control _instruct; // "instruct" badge, shown when the server detects a chat-templated model
    private ChatSocket _chat;
    private TurnCard _activeTurn;
    private string _sessionModel = "", _sessionBackend = "";
    private bool _sessionMemory;
    private bool _chatBusy;
    private bool _resetSession;

    public ChatView(AppState state) => _state = state;

    public Control Root => this;
    public void OnShown() { }  // nothing to start: streaming continues while hidden so a reply isn't lost
    public void OnHidden() { }

    public override void _Ready()
    {
        SizeFlagsHorizontal = SizeFlags.ExpandFill;
        SizeFlagsVertical = SizeFlags.ExpandFill;
        AddThemeConstantOverride("separation", 0);

        _recents = new RecentsPanel();
        AddChild(_recents);

        var column = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        column.AddThemeConstantOverride("separation", 0);
        AddChild(column);

        var header = new PanelContainer();
        Palette.StylePanel(header, Palette.AppBg, pad: 14);
        var headerRow = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        headerRow.AddThemeConstantOverride("separation", Palette.Space.Sm);
        _title = Palette.Heading("New chat", Palette.Type.H3);
        _title.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        headerRow.AddChild(_title);
        _instruct = Palette.Badge("instruct", Palette.Tone.Accent);
        _instruct.Visible = false;
        _instruct.TooltipText = "The server detected a chat-templated (instruct) model for this session.";
        headerRow.AddChild(_instruct);
        _meter = Palette.Heading("", Palette.Type.Caption, Palette.Muted);
        headerRow.AddChild(_meter);
        var clear = Palette.GhostButton("Clear", Palette.Type.Label);
        clear.TooltipText = "Clear the conversation (the next message starts a fresh session).";
        clear.Pressed += OnNewChat;
        headerRow.AddChild(clear);
        header.AddChild(headerRow);
        column.AddChild(header);

        column.AddChild(Palette.Pad(_transcript = new Transcript(), left: 24, right: 24, top: 8, bottom: 8, expand: true));
        column.AddChild(Palette.Pad(_composer = new Composer(), left: 16, right: 16, top: 8, bottom: 16));

        _chat = new ChatSocket();
        AddChild(_chat);

        _chat.Token += delta => { _activeTurn?.Append(delta); _transcript.ScrollToBottom(); };
        _chat.Sources += sources => { _activeTurn?.SetSources(sources); _transcript.ScrollToBottom(); };
        _chat.SessionReady += OnSessionReady;
        _chat.Done += OnChatDone;
        _chat.ChatError += OnChatError;
        _chat.Closed += OnChatClosed;

        _composer.Submitted += OnSubmit;
        _composer.Canceled += OnCancel;
        _composer.FontSizeChanged += size =>
        {
            _transcript.SetFontSize(size);
            _state.MutatePrefs(p => p.FontSize = size);
        };
        _composer.SettingsChanged += PersistComposerSettings;

        _recents.NewChatRequested += OnNewChat;
        _recents.RecentSelected += prompt => _composer.SetPrompt(prompt);

        _composer.ApplyPrefs(_state.Prefs);
        _transcript.SetFontSize(_state.Prefs.FontSize);

        _state.HealthChanged += OnHealth;
        // Another view chose a model (e.g. "Chat with" on a card) — reflect it in the picker.
        _state.SelectionChanged += () =>
        {
            if (!string.IsNullOrEmpty(_state.SelectedModel)) _composer.SelectModel(_state.SelectedModel);
        };
        OnHealth(); // render whatever state already exists — a view registered after the first /health must not start empty
    }

    private void OnHealth()
    {
        if (_state.Health is not { Ok: true } health) return;
        // Prefer this machine's remembered model/backend when the server still offers them, else the server default.
        string preferredModel = System.Array.IndexOf(health.Models, _state.Prefs.DefaultModel) >= 0
            ? _state.Prefs.DefaultModel : health.Default;
        string preferredBackend = health.DefaultBackend;
        foreach (var b in health.Backends)
            if (b.Id == _state.Prefs.DefaultBackend && b.Available) { preferredBackend = b.Id; break; }
        _composer.SetModels(health.Models, preferredModel);
        _composer.SetBackends(health.Backends, preferredBackend);
    }

    // Any popup change (model, backend, sampling, length, research) becomes this machine's new default.
    private void PersistComposerSettings()
    {
        var s = _composer.Snapshot();
        _state.MutatePrefs(p =>
        {
            p.DefaultModel = s.Model;
            p.DefaultBackend = s.Backend;
            p.Sample = s.Sample;
            p.Temperature = s.Temperature;
            p.TopK = s.TopK;
            p.TopP = s.TopP;
            p.MaxTokens = s.MaxTokens;
            p.Research = s.Research;
            p.Memory = s.Memory;
            p.Seed = s.Seed;
        });
    }

    private void OnSubmit(GenerateRequest request)
    {
        if (_chatBusy) return;
        _activeTurn = _transcript.Begin(request.Prompt);
        _recents.AddRecent(request.Prompt);
        _title.Text = Format.Ellipsize(request.Prompt, 40);
        _composer.SetBusy(true);
        _chatBusy = true;
        _state.SetSelection(request.Model, request.Backend);

        // One persistent /chat connection keeps the server's KV cache warm across turns. (Re)start the session when
        // first connecting, when the model/backend/memory choice changed, or after "New chat" cleared the
        // conversation. Memory rides the start frame only (baked into the warm cache), hence the restart on toggle.
        if (!_chat.IsActive) { _chat.Connect(_state.ServerUrl); _resetSession = true; }
        if (_resetSession || request.Model != _sessionModel || request.Backend != _sessionBackend
            || request.Memory != _sessionMemory)
        {
            _chat.SendStart(request.Model, request.Backend, request.Memory);
            _sessionModel = request.Model;
            _sessionBackend = request.Backend;
            _sessionMemory = request.Memory;
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

    private void OnSessionReady(SessionInfo info)
    {
        _instruct.Visible = info.Instruct;
        if (info.ContextLimit > 0) _meter.Text = $"ctx {info.ContextLimit:N0}";
    }

    private void OnChatDone(TurnStats stats)
    {
        _activeTurn?.Complete(stats.Stop);
        _transcript.ScrollToBottom();
        _activeTurn = null;
        _chatBusy = false;
        _composer.SetBusy(false);
        // The short done form (research canceled mid-search) has no accounting — keep the previous meter then.
        if (stats.ContextLimit > 0)
            _meter.Text = $"{stats.Position:N0} / {stats.ContextLimit:N0} ctx   ·   {stats.GeneratedTokens} tok in {stats.Seconds:0.0}s";
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

    private void OnNewChat()
    {
        _transcript.Clear();
        _title.Text = "New chat";
        _meter.Text = "";
        _resetSession = true; // next message starts a fresh server session (drops the warm cache)
    }
}
