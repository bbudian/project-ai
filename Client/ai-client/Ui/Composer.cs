using System;
using Godot;

// The bottom input panel, matching the design harness's composer (prototype/js/views/chat.js): the Model and
// Backend pickers sit VISIBLE in the card (switching models is the product, not an advanced setting), Memory and
// Web are toggle chips, then the prompt box and a compact action row — an "Advanced" popup for sampling / response
// length / seed / text size, and Send (which doubles as Stop while a reply streams). It emits a single
// Submitted(GenerateRequest); SettingsChanged fires on any setting change so the host persists a Snapshot() to prefs.
public partial class Composer : PanelContainer
{
    public event Action<GenerateRequest> Submitted;
    public event Action<int> FontSizeChanged;
    public event Action Canceled; // the Send button doubles as Stop while a reply streams
    public event Action SettingsChanged; // any setting changed — the host persists a Snapshot() to prefs

    private TextEdit _prompt;
    private Button _send;
    private Button _advanced;
    private Label _status;
    private bool _busy;

    private OptionButton _model;
    private BackendPicker _backend;
    private Button _memory, _research; // toggle chips

    private PopupPanel _advancedPopup;
    private CheckButton _sample, _capLength;
    private SpinBox _temperature, _topK, _topP, _maxTokens, _fontSize, _seed;

    public override void _Ready()
    {
        Palette.StylePanel(this, Palette.InputBg, radius: Palette.Radius.Lg, pad: 12, border: 1, borderColor: Palette.Border);

        var column = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        column.AddThemeConstantOverride("separation", Palette.Space.Sm);
        AddChild(column);

        // Pickers, visible like the mockup's 2-column grid.
        var pickers = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        pickers.AddThemeConstantOverride("separation", Palette.Space.Sm);
        _model = new OptionButton
        {
            TooltipText = "Model",
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
            SizeFlagsStretchRatio = 3,
        };
        _model.ItemSelected += _ => SettingsChanged?.Invoke();
        pickers.AddChild(_model);
        _backend = new BackendPicker { SizeFlagsHorizontal = SizeFlags.ExpandFill, SizeFlagsStretchRatio = 2 };
        _backend.ItemSelected += _ => SettingsChanged?.Invoke();
        pickers.AddChild(_backend);
        column.AddChild(pickers);

        // Toggle chips + a live status span (mockup: brain + world-search chips).
        var chips = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        chips.AddThemeConstantOverride("separation", Palette.Space.Sm);
        _memory = Palette.Chip("🧠  Memory");
        _memory.TooltipText = "Attach this model's long-term memory store to the conversation (toggling restarts the session).";
        _memory.Toggled += _ => SettingsChanged?.Invoke();
        chips.AddChild(_memory);
        _research = Palette.Chip("🌐  Web");
        _research.TooltipText = "Ground answers in a live web search with citations (needs a Tavily key — Settings → Web search).";
        _research.Toggled += _ => SettingsChanged?.Invoke();
        chips.AddChild(_research);
        _status = Palette.Heading("", Palette.Type.Caption, Palette.Muted);
        _status.SizeFlagsVertical = SizeFlags.ShrinkCenter;
        chips.AddChild(_status);
        column.AddChild(chips);

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
        actions.AddThemeConstantOverride("separation", Palette.Space.Sm);
        _advanced = Palette.GhostButton("⚙  Advanced", Palette.Type.Label);
        _advanced.TooltipText = "Sampling, response length, seed, and text size";
        _advanced.Pressed += OpenAdvanced;
        actions.AddChild(_advanced);
        actions.AddChild(new Control { SizeFlagsHorizontal = SizeFlags.ExpandFill }); // push Send to the right
        _send = Palette.PrimaryButton("Send  ↵");
        _send.Pressed += OnSendPressed;
        actions.AddChild(_send);
        column.AddChild(actions);

        BuildAdvancedPopup();
    }

    // Sampling / length / appearance — genuinely advanced settings; everything product-level lives in the card.
    private void BuildAdvancedPopup()
    {
        _advancedPopup = new PopupPanel();
        _advancedPopup.AddThemeStyleboxOverride("panel",
            Palette.Box(Palette.PanelBg, radius: Palette.Radius.Md, pad: Palette.Space.Lg, border: 1, borderColor: Palette.Border));
        AddChild(_advancedPopup);

        var col = new VBoxContainer { CustomMinimumSize = new Vector2(360, 0) };
        col.AddThemeConstantOverride("separation", 10);
        _advancedPopup.AddChild(col);

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
        col.AddChild(Palette.Heading("Appearance", Palette.Type.Caption, Palette.Muted));
        _fontSize = Param(col, "Text size", 11, 28, 1, Palette.DefaultFontSize); // resizes the conversation text live
        _fontSize.ValueChanged += v => FontSizeChanged?.Invoke((int)v);

        OnSampleToggled(false); // greedy by default → decoding controls start disabled
        OnCapToggled(false);    // dynamic length by default → the cap spinbox starts disabled
    }

    /// <summary>Seeds every control from persisted prefs (called once by the host view after construction).
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

    /// <summary>The current settings as a request with an empty prompt — what the host persists to prefs.</summary>
    public GenerateRequest Snapshot() => BuildRequest("");

    public void SetBusy(bool busy)
    {
        _busy = busy;
        _send.Disabled = false;                 // stays clickable while busy — it's the Stop button now
        _send.Text = busy ? "Stop  ◼" : "Send  ↵";
        _status.Text = busy ? "Generating…" : "";
    }

    // The primary button sends a prompt when idle and cancels the streaming reply while busy (mockup-style).
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
                return;
            }
    }

    /// <summary>Populates the backend picker (shared <see cref="BackendPicker"/> logic).</summary>
    public void SetBackends(BackendOption[] backends, string defaultId) => _backend.SetBackends(backends, defaultId);

    private void OpenAdvanced()
    {
        // The composer sits at the bottom, so open the popup above the button; fall below if there's no room.
        var rect = _advanced.GetGlobalRect();
        Vector2I size = (Vector2I)_advancedPopup.Size;
        if (size.X < 50 || size.Y < 50) size = new Vector2I(392, 360); // first open: popup hasn't been sized yet
        int y = (int)rect.Position.Y - size.Y - 8;
        if (y < 8) y = (int)(rect.Position.Y + rect.Size.Y + 8);
        _advancedPopup.Popup(new Rect2I(new Vector2I((int)rect.Position.X, y), size));
    }

    private void OnSampleToggled(bool on)
    {
        _temperature.Editable = on;
        _topK.Editable = on;
        _topP.Editable = on;
        if (_seed != null) _seed.Editable = on;
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

        var caption = Palette.Heading(label, Palette.Type.Label, Palette.Muted);
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
