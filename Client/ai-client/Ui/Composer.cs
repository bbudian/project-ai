using System;
using Godot;

// The bottom input panel: a prompt box plus the decoding controls and a Send button, styled like a chat
// composer. It encapsulates the sampling widgets and emits a single Submitted(GenerateRequest) — the rest of
// the app never touches temperature/top-k/etc. SetBusy/SetPrompt are the only ways outside code drives it.
public partial class Composer : PanelContainer
{
    public event Action<GenerateRequest> Submitted;

    private TextEdit _prompt;
    private OptionButton _model;
    private CheckButton _sample;
    private SpinBox _temperature, _topK, _topP, _maxTokens;
    private Button _send;

    public override void _Ready()
    {
        Palette.StylePanel(this, Palette.InputBg, radius: 14, pad: 10, border: 1, borderColor: Palette.Border);

        var column = new VBoxContainer();
        column.AddThemeConstantOverride("separation", 8);
        AddChild(column);

        _prompt = new TextEdit
        {
            PlaceholderText = "Message ProjectAI…  (Ctrl+Enter to send)",
            Text = "roses are ",
            CustomMinimumSize = new Vector2(0, 64),
            WrapMode = TextEdit.LineWrappingMode.Boundary,
        };
        _prompt.AddThemeStyleboxOverride("normal", Palette.Box(Palette.Transparent));
        _prompt.AddThemeStyleboxOverride("focus", Palette.Box(Palette.Transparent));
        _prompt.GuiInput += OnPromptInput;
        column.AddChild(_prompt);

        var controls = new HBoxContainer();
        controls.AddThemeConstantOverride("separation", 10);

        // Model picker (like Claude's model selector) — populated from the server's available checkpoints.
        _model = new OptionButton { TooltipText = "Model", CustomMinimumSize = new Vector2(150, 0) };
        controls.AddChild(_model);

        _sample = new CheckButton { Text = "Sample" };
        _sample.AddThemeColorOverride("font_color", Palette.Muted);
        controls.AddChild(_sample);

        _temperature = Param(controls, "Temp", 0, 2, 0.05, 0.8);
        _topK = Param(controls, "Top-K", 0, 200, 1, 40);
        _topP = Param(controls, "Top-P", 0.05, 1, 0.05, 0.9);
        _maxTokens = Param(controls, "Max", 1, 512, 1, 64);

        controls.AddChild(new Control { SizeFlagsHorizontal = SizeFlags.ExpandFill }); // push Send to the right

        _send = Palette.PrimaryButton("Send  ↵");
        _send.Pressed += Submit;
        controls.AddChild(_send);

        column.AddChild(controls);
    }

    public void SetBusy(bool busy)
    {
        _send.Disabled = busy;
        _send.Text = busy ? "Sending…" : "Send  ↵";
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
        _model.Selected = keep >= 0 ? keep : (fallback >= 0 ? fallback : (models.Length > 0 ? 0 : -1));
    }

    private string SelectedModel() => _model.Selected >= 0 ? _model.GetItemText(_model.Selected) : "";

    private void OnPromptInput(InputEvent @event)
    {
        if (@event is InputEventKey { Pressed: true, Keycode: Key.Enter, CtrlPressed: true })
        {
            Submit();
            _prompt.AcceptEvent();
        }
    }

    private void Submit()
    {
        if (_send.Disabled || string.IsNullOrWhiteSpace(_prompt.Text)) return;
        Submitted?.Invoke(new GenerateRequest(
            _prompt.Text,
            SelectedModel(),
            _sample.ButtonPressed,
            (float)_temperature.Value,
            (int)_topK.Value,
            (float)_topP.Value,
            (int)_maxTokens.Value));
    }

    private static SpinBox Param(Control row, string label, double min, double max, double step, double value)
    {
        var caption = Palette.Heading(label, 12, Palette.Muted);
        row.AddChild(caption);
        var spin = new SpinBox
        {
            MinValue = min,
            MaxValue = max,
            Step = step,
            Value = value,
            CustomMinimumSize = new Vector2(64, 0),
        };
        row.AddChild(spin);
        return spin;
    }
}
