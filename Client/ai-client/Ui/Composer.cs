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

    private TextEdit _prompt;
    private Button _settingsButton;
    private Button _send;
    private bool _busy;

    private PopupPanel _settingsPopup;
    private OptionButton _model;
    private BackendPicker _backend;
    private CheckButton _sample, _capLength;
    private SpinBox _temperature, _topK, _topP, _maxTokens;

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
        _model.ItemSelected += _ => UpdateSettingsButton();
        col.AddChild(_model);

        col.AddChild(Palette.Heading("Compute backend", 12, Palette.Muted));
        _backend = new BackendPicker { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _backend.ItemSelected += _ => UpdateSettingsButton();
        col.AddChild(_backend);

        col.AddChild(new HSeparator());

        _sample = new CheckButton { Text = "Sample  (off = greedy / deterministic)" };
        _sample.AddThemeColorOverride("font_color", Palette.Text);
        _sample.Toggled += OnSampleToggled;
        col.AddChild(_sample);

        _temperature = Param(col, "Temperature", 0, 2, 0.05, 0.8);
        _topK = Param(col, "Top-K", 0, 200, 1, 40);
        _topP = Param(col, "Top-P", 0.05, 1, 0.05, 0.9);

        col.AddChild(new HSeparator());
        // Response length. Off (default) sends maxTokens 0, which the server reads as "dynamic" — stream until the
        // model stops (EOS / <|im_end|>) or its context window fills. Turn it on to impose an explicit cap; the
        // server clamps that to the loaded model's context, so the spinbox max is just a generous upper bound.
        _capLength = new CheckButton { Text = "Limit response length  (off = until the model stops)" };
        _capLength.AddThemeColorOverride("font_color", Palette.Text);
        _capLength.Toggled += OnCapToggled;
        col.AddChild(_capLength);
        _maxTokens = Param(col, "Max tokens", 1, 8192, 1, 1024);

        col.AddChild(new HSeparator());
        col.AddChild(Palette.Heading("Appearance", 12, Palette.Muted));
        var fontSize = Param(col, "Text size", 11, 28, 1, Palette.DefaultFontSize); // resizes the conversation text live
        fontSize.ValueChanged += v => FontSizeChanged?.Invoke((int)v);

        OnSampleToggled(false); // greedy by default → decoding controls start disabled
        OnCapToggled(false);    // dynamic length by default → the cap spinbox starts disabled
    }

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
        Submitted?.Invoke(new GenerateRequest(
            _prompt.Text,
            SelectedModel(),
            _backend.SelectedId,
            _sample.ButtonPressed,
            (float)_temperature.Value,
            (int)_topK.Value,
            (float)_topP.Value,
            _capLength.ButtonPressed ? (int)_maxTokens.Value : 0)); // 0 = dynamic (until the model stops / context fills)
        _prompt.Text = ""; // clear the input for the next message (the request already captured the text)
    }

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
