using System;
using Godot;

// The bottom input panel: a prompt box and a compact action row (a "settings" button that opens a popup, plus
// Send). Model, compute backend, and the sampling/length controls live in the popup — Claude-style — so the
// composer stays uncluttered and the Send button is always visible. It emits a single Submitted(GenerateRequest);
// the rest of the app never touches temperature/top-k/etc. SetBusy/SetPrompt are the only ways outside code drives it.
public partial class Composer : PanelContainer
{
    public event Action<GenerateRequest> Submitted;
    public event Action<int> FontSizeChanged;
    public event Action Canceled; // the Send button doubles as Stop while a reply streams
    public event Action SettingsChanged; // any popup setting changed — the host persists a Snapshot() to prefs

    private TextEdit _prompt;
    private Button _settingsButton;
    private Button _send;
    private bool _busy;

    private PopupPanel _settingsPopup;
    private OptionButton _model;
    private BackendPicker _backend;
    private CheckButton _sample, _capLength, _research, _memory;
    private SpinBox _temperature, _topK, _topP, _maxTokens, _fontSize, _seed;

    public override void _Ready()
    {
        Palette.StylePanel(this, Palette.InputBg, radius: 14, pad: 10, border: 1, borderColor: Palette.Border);

        var column = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        column.AddThemeConstantOverride("separation", 8);
        AddChild(column);

        _prompt = new TextEdit
        {
            PlaceholderText = "Message ProjectAI…  (Enter to send, Shift+Enter for newline)",
            CustomMinimumSize = new Vector2(0, 72),
            WrapMode = TextEdit.LineWrappingMode.Boundary,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        _prompt.AddThemeStyleboxOverride("normal", Palette.Box(Palette.Transparent));
        _prompt.AddThemeStyleboxOverride("focus", Palette.Box(Palette.Transparent));
        _prompt.GuiInput += OnPromptInput;
        column.AddChild(_prompt);

        var actions = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        actions.AddThemeConstantOverride("separation", 8);
        column.AddChild(actions);

        // Opens the model/backend/sampling popup; its label reflects the current model + backend (Claude-style).
        _settingsButton = Palette.GhostButton("⚙  Model & settings");
        _settingsButton.Pressed += OpenSettings;
        actions.AddChild(_settingsButton);

        actions.AddChild(new Control { SizeFlagsHorizontal = SizeFlags.ExpandFill }); // push Send to the right

        _send = Palette.PrimaryButton("Send  ↵");
        _send.Pressed += OnSendPressed;
        actions.AddChild(_send);

        BuildSettingsPopup();
        UpdateSettingsButton();
    }

    // The settings popup: model + backend pickers, a sampling toggle with its decoding controls, and max length.
    private void BuildSettingsPopup()
    {
        _settingsPopup = new PopupPanel();
        _settingsPopup.AddThemeStyleboxOverride("panel",
            Palette.Box(Palette.PanelBg, radius: 12, pad: 16, border: 1, borderColor: Palette.Border));
        AddChild(_settingsPopup);

        var col = new VBoxContainer { CustomMinimumSize = new Vector2(360, 0) };
        col.AddThemeConstantOverride("separation", 10);
        _settingsPopup.AddChild(col);

        col.AddChild(Palette.Heading("Model", 12, Palette.Muted));
        _model = new OptionButton { TooltipText = "Model", SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _model.ItemSelected += _ => { UpdateSettingsButton(); SettingsChanged?.Invoke(); };
        col.AddChild(_model);

        col.AddChild(Palette.Heading("Compute backend", 12, Palette.Muted));
        _backend = new BackendPicker { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _backend.ItemSelected += _ => { UpdateSettingsButton(); SettingsChanged?.Invoke(); };
        col.AddChild(_backend);

        col.AddChild(new HSeparator());

        _sample = new CheckButton { Text = "Sample  (off = greedy / deterministic)" };
        _sample.AddThemeColorOverride("font_color", Palette.Text);
        _sample.Toggled += on => { OnSampleToggled(on); SettingsChanged?.Invoke(); };
        col.AddChild(_sample);

        _temperature = Param(col, "Temperature", 0, 2, 0.05, 0.8);
        _topK = Param(col, "Top-K", 0, 200, 1, 40);
        _topP = Param(col, "Top-P", 0.05, 1, 0.05, 0.9);
        _seed = Param(col, "Seed", 0, uint.MaxValue, 1, 0);
        foreach (var spin in new[] { _temperature, _topK, _topP, _seed })
            spin.ValueChanged += _ => SettingsChanged?.Invoke();

        col.AddChild(new HSeparator());
        // Response length. Off (default) sends maxTokens 0, which the server reads as "dynamic" — stream until the
        // model stops (EOS / <|im_end|>) or its context window fills. Turn it on to impose an explicit cap; the
        // server clamps that to the loaded model's context, so the spinbox max is just a generous upper bound.
        _capLength = new CheckButton { Text = "Limit response length  (off = until the model stops)" };
        _capLength.AddThemeColorOverride("font_color", Palette.Text);
        _capLength.Toggled += on => { OnCapToggled(on); SettingsChanged?.Invoke(); };
        col.AddChild(_capLength);
        _maxTokens = Param(col, "Max tokens", 1, 8192, 1, 1024);
        _maxTokens.ValueChanged += _ => SettingsChanged?.Invoke();

        col.AddChild(new HSeparator());
        col.AddChild(Palette.Heading("Web research", 12, Palette.Muted));
        // When on, the server runs a live web search for the prompt and grounds the answer in the results (RAG),
        // returning the sources for citation. Needs TAVILY_API_KEY set on the server, else the turn errors clearly.
        _research = new CheckButton { Text = "🌐  Search the web and answer from current results" };
        _research.AddThemeColorOverride("font_color", Palette.Text);
        _research.Toggled += _ => SettingsChanged?.Invoke();
        col.AddChild(_research);

        col.AddChild(new HSeparator());
        col.AddChild(Palette.Heading("Memory", 12, Palette.Muted));
        // When on, the chat session attaches this model's server-side memory store: pinned facts are baked into the
        // warm cache at session start, and each message recalls relevant memories. Toggling it restarts the session.
        _memory = new CheckButton { Text = "🧠  Use long-term memory for this conversation" };
        _memory.AddThemeColorOverride("font_color", Palette.Text);
        _memory.Toggled += _ => SettingsChanged?.Invoke();
        col.AddChild(_memory);

        col.AddChild(new HSeparator());
        col.AddChild(Palette.Heading("Appearance", 12, Palette.Muted));
        _fontSize = Param(col, "Text size", 11, 28, 1, Palette.DefaultFontSize); // resizes the conversation text live
        _fontSize.ValueChanged += v => FontSizeChanged?.Invoke((int)v);

        OnSampleToggled(false); // greedy by default → decoding controls start disabled
        OnCapToggled(false);    // dynamic length by default → the cap spinbox starts disabled
    }

    /// <summary>Seeds every popup control from persisted prefs (called once by the host view after construction).
    /// Values are set directly — the resulting change events just re-persist identical values, which is harmless.</summary>
    public void ApplyPrefs(ClientPrefs p)
    {
        _sample.SetPressedNoSignal(p.Sample);
        OnSampleToggled(p.Sample);
        _temperature.Value = p.Temperature;
        _topK.Value = p.TopK;
        _topP.Value = p.TopP;
        _seed.Value = p.Seed;
        _capLength.SetPressedNoSignal(p.MaxTokens > 0);
        OnCapToggled(p.MaxTokens > 0);
        if (p.MaxTokens > 0) _maxTokens.Value = p.MaxTokens;
        _research.SetPressedNoSignal(p.Research);
        _memory.SetPressedNoSignal(p.Memory);
        _fontSize.Value = p.FontSize;
    }

    /// <summary>The current popup settings as a request with an empty prompt — what the host persists to prefs.</summary>
    public GenerateRequest Snapshot() => BuildRequest("");

    public void SetBusy(bool busy)
    {
        _busy = busy;
        _send.Disabled = false;                 // stays clickable while busy — it's the Stop button now
        _send.Text = busy ? "Stop  ◼" : "Send  ↵";
    }

    // The primary button sends a prompt when idle and cancels the streaming reply while busy (Claude-style).
    private void OnSendPressed()
    {
        if (!_busy) { Submit(); return; }
        _send.Disabled = true;                  // disabled until the server's "done" lands → SetBusy(false) re-enables
        _send.Text = "Stopping…";
        Canceled?.Invoke();
    }

    public void SetPrompt(string text)
    {
        _prompt.Text = text;
        _prompt.GrabFocus();
    }

    /// <summary>Populates the model picker, keeping the current choice if it still exists, else selecting the default.</summary>
    public void SetModels(string[] models, string defaultModel)
    {
        string previous = SelectedModel();
        _model.Clear();
        foreach (var name in models) _model.AddItem(name);

        int keep = System.Array.IndexOf(models, previous);
        int fallback = System.Array.IndexOf(models, defaultModel);
        _model.Selected = Selection.FirstValid(keep, fallback, models.Length > 0 ? 0 : -1);
        UpdateSettingsButton();
    }

    private string SelectedModel() => _model is { Selected: >= 0 } ? _model.GetItemText(_model.Selected) : "";

    /// <summary>Programmatically selects a model (e.g. a model card's "Chat with"). No SettingsChanged is raised —
    /// the caller already updated the shared selection state.</summary>
    public void SelectModel(string name)
    {
        for (int i = 0; i < _model.ItemCount; i++)
            if (_model.GetItemText(i) == name)
            {
                _model.Selected = i;
                UpdateSettingsButton();
                return;
            }
    }

    /// <summary>Populates the backend picker (shared <see cref="BackendPicker"/> logic), then refreshes the trigger label.</summary>
    public void SetBackends(BackendOption[] backends, string defaultId)
    {
        _backend.SetBackends(backends, defaultId);
        UpdateSettingsButton();
    }

    // Reflects the current model + backend on the trigger button, e.g. "⚙  smol  ·  GPU (CUDA)".
    private void UpdateSettingsButton()
    {
        string model = SelectedModel();
        string backendLabel = _backend is { Selected: >= 0 } ? _backend.GetItemText(_backend.Selected) : "";
        string label = string.IsNullOrEmpty(model) ? "Model & settings" : model;
        if (!string.IsNullOrEmpty(backendLabel)) label += $"  ·  {backendLabel}";
        _settingsButton.Text = $"⚙  {label}";
    }

    private void OpenSettings()
    {
        // The composer sits at the bottom, so open the popup above the button; fall below if there's no room.
        var rect = _settingsButton.GetGlobalRect();
        Vector2I size = (Vector2I)_settingsPopup.Size;
        if (size.X < 50 || size.Y < 50) size = new Vector2I(392, 380); // first open: popup hasn't been sized yet
        int y = (int)rect.Position.Y - size.Y - 8;
        if (y < 8) y = (int)(rect.Position.Y + rect.Size.Y + 8);
        _settingsPopup.Popup(new Rect2I(new Vector2I((int)rect.Position.X, y), size));
    }

    private void OnSampleToggled(bool on)
    {
        _temperature.Editable = on;
        _topK.Editable = on;
        _topP.Editable = on;
        if (_seed != null) _seed.Editable = on; // built after the toggle's first programmatic fire
    }

    private void OnCapToggled(bool on) => _maxTokens.Editable = on; // off → dynamic length (Submit sends maxTokens 0)

    private void OnPromptInput(InputEvent @event)
    {
        // Enter sends; Shift+Enter inserts a newline (chat-style). Ctrl+Enter also sends.
        if (@event is InputEventKey { Pressed: true, Keycode: Key.Enter or Key.KpEnter } key && !key.ShiftPressed)
        {
            Submit();
            _prompt.AcceptEvent();
        }
    }

    private void Submit()
    {
        if (_busy || string.IsNullOrWhiteSpace(_prompt.Text)) return;
        Submitted?.Invoke(BuildRequest(_prompt.Text));
        _prompt.Text = ""; // clear the input for the next message (the request already captured the text)
    }

    private GenerateRequest BuildRequest(string prompt) => new(
        prompt,
        SelectedModel(),
        _backend.SelectedId,
        _sample.ButtonPressed,
        (float)_temperature.Value,
        (int)_topK.Value,
        (float)_topP.Value,
        _capLength.ButtonPressed ? (int)_maxTokens.Value : 0, // 0 = dynamic (until the model stops / context fills)
        _research.ButtonPressed,
        _memory.ButtonPressed,
        (ulong)_seed.Value);

    // A labelled spinbox row for the vertical popup layout: caption on the left, control on the right.
    private static SpinBox Param(VBoxContainer col, string label, double min, double max, double step, double value)
    {
        var row = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        row.AddThemeConstantOverride("separation", 10);

        var caption = Palette.Heading(label, 13, Palette.Muted);
        caption.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        row.AddChild(caption);

        var spin = new SpinBox
        {
            MinValue = min,
            MaxValue = max,
            Step = step,
            Value = value,
            CustomMinimumSize = new Vector2(110, 0),
        };
        row.AddChild(spin);

        col.AddChild(row);
        return spin;
    }
}
