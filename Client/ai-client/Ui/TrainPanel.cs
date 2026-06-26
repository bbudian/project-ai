using System;
using Godot;

// The Train view: pick a text file, name the model, choose a size / backend / step count, and train it on the
// server (on the GPU when available). Emits TrainRequested with the chosen settings + the file's text; the host
// (Main) sends it via the API and feeds progress back through UpdateStatus. Mirrors Composer's component style.
public partial class TrainPanel : PanelContainer
{
    public event Action<TrainRequest> TrainRequested;

    private LineEdit _name;
    private OptionButton _size;
    private BackendPicker _backend;
    private SpinBox _steps;
    private Button _train;
    private Label _fileLabel;
    private Label _statusLabel;
    private ProgressBar _progress;
    private FileDialog _fileDialog;
    private string _text = "";

    public override void _Ready()
    {
        Palette.StylePanel(this, Palette.InputBg, radius: 14, pad: 16, border: 1, borderColor: Palette.Border);

        var col = new VBoxContainer();
        col.AddThemeConstantOverride("separation", 12);
        AddChild(col);

        col.AddChild(Palette.Heading("Train a new model", 18));
        col.AddChild(Palette.Heading("Pick a text file, choose a size, and train on your GPU. It appears in the chat model picker when done.", 12, Palette.Muted));

        var fileRow = new HBoxContainer();
        fileRow.AddThemeConstantOverride("separation", 10);
        var choose = Palette.GhostButton("Choose text file…");
        choose.Pressed += () => _fileDialog.PopupCentered(new Vector2I(720, 500));
        fileRow.AddChild(choose);
        _fileLabel = Palette.Heading("no file selected", 13, Palette.Muted);
        fileRow.AddChild(_fileLabel);
        col.AddChild(fileRow);

        var settings = new HBoxContainer();
        settings.AddThemeConstantOverride("separation", 10);
        settings.AddChild(Palette.Heading("Name", 12, Palette.Muted));
        _name = new LineEdit { Text = "mymodel", CustomMinimumSize = new Vector2(150, 0) };
        settings.AddChild(_name);
        settings.AddChild(Palette.Heading("Size", 12, Palette.Muted));
        _size = new OptionButton { CustomMinimumSize = new Vector2(220, 0) }; // filled from /health (SetSizes)
        settings.AddChild(_size);
        settings.AddChild(Palette.Heading("Steps", 12, Palette.Muted));
        _steps = new SpinBox { MinValue = 1, MaxValue = 100_000, Step = 50, Value = 300, CustomMinimumSize = new Vector2(100, 0) };
        settings.AddChild(_steps);
        col.AddChild(settings);

        var backendRow = new HBoxContainer();
        backendRow.AddThemeConstantOverride("separation", 10);
        backendRow.AddChild(Palette.Heading("Backend", 12, Palette.Muted));
        _backend = new BackendPicker { CustomMinimumSize = new Vector2(160, 0) };
        backendRow.AddChild(_backend);
        col.AddChild(backendRow);

        _train = Palette.PrimaryButton("Train");
        _train.Pressed += OnTrain;
        col.AddChild(_train);

        _progress = new ProgressBar { MinValue = 0, MaxValue = 100, Value = 0, ShowPercentage = false, CustomMinimumSize = new Vector2(0, 16) };
        col.AddChild(_progress);
        _statusLabel = Palette.Heading("", 13, Palette.Muted);
        col.AddChild(_statusLabel);

        _fileDialog = new FileDialog
        {
            FileMode = FileDialog.FileModeEnum.OpenFile,
            Access = FileDialog.AccessEnum.Filesystem,
            Title = "Choose a text file to train on",
            Filters = ["*.txt,*.md,*.json,*.csv ; Text files", "* ; All files"],
        };
        _fileDialog.FileSelected += OnFileSelected;
        AddChild(_fileDialog);
    }

    /// <summary>Populates the compute-backend picker (shared <see cref="BackendPicker"/> logic — same as the chat composer's).</summary>
    public void SetBackends(BackendOption[] backends, string defaultId) => _backend.SetBackends(backends, defaultId);

    /// <summary>
    /// Populates the size picker from the server's presets (/health — the single source of truth for which sizes
    /// exist). Display is "id — description"; the request sends the id. Keeps the current pick across refreshes,
    /// else prefers "small", else the first.
    /// </summary>
    public void SetSizes(SizeOption[] sizes)
    {
        string previous = SelectedSize();
        _size.Clear();
        int keep = -1, small = -1;
        for (int i = 0; i < sizes.Length; i++)
        {
            _size.AddItem(string.IsNullOrEmpty(sizes[i].Label) ? sizes[i].Id : $"{sizes[i].Id} — {sizes[i].Label}");
            _size.SetItemMetadata(i, sizes[i].Id);
            if (sizes[i].Id == previous) keep = i;
            if (sizes[i].Id == "small") small = i;
        }
        int select = Selection.FirstValid(keep, small, sizes.Length > 0 ? 0 : -1);
        if (select >= 0) _size.Selected = select;
    }

    public void SetBusy(bool busy)
    {
        _train.Disabled = busy;
        _train.Text = busy ? "Training…" : "Train";
    }

    /// <summary>Drives the progress bar + status line from a /train/status poll.</summary>
    public void UpdateStatus(TrainStatus s)
    {
        switch (s.State)
        {
            case "running":
                _progress.Value = s.TotalSteps > 0 ? 100.0 * s.Step / s.TotalSteps : 0;
                _statusLabel.AddThemeColorOverride("font_color", Palette.Muted);
                _statusLabel.Text = $"Training '{s.Name}'…  step {s.Step}/{s.TotalSteps},  loss {s.Loss:0.000}";
                break;
            case "done":
                _progress.Value = 100;
                _statusLabel.AddThemeColorOverride("font_color", Palette.Good);
                _statusLabel.Text = $"Done — '{s.Name}' trained (loss {s.Loss:0.000}). Switch to Chat and pick it.";
                break;
            case "error":
                _statusLabel.AddThemeColorOverride("font_color", Palette.Bad);
                _statusLabel.Text = $"Error: {s.Error}";
                break;
            default: // "idle" (e.g. the server restarted and lost the job) or any unrecognized state
                _statusLabel.AddThemeColorOverride("font_color", Palette.Muted);
                _statusLabel.Text = string.IsNullOrEmpty(s.Error)
                    ? "Training is no longer running on the server."
                    : $"Error: {s.Error}";
                break;
        }
    }

    private void OnFileSelected(string path)
    {
        using var file = Godot.FileAccess.Open(path, Godot.FileAccess.ModeFlags.Read);
        if (file is null)
        {
            _statusLabel.AddThemeColorOverride("font_color", Palette.Bad);
            _statusLabel.Text = "Could not read that file.";
            return;
        }
        _text = file.GetAsText();
        _fileLabel.AddThemeColorOverride("font_color", Palette.Text);
        _fileLabel.Text = $"{System.IO.Path.GetFileName(path)}  ({_text.Length:N0} chars)";
        if (_name.Text is "" or "mymodel") _name.Text = Sanitize(System.IO.Path.GetFileNameWithoutExtension(path));
    }

    private void OnTrain()
    {
        if (string.IsNullOrEmpty(_text)) { Warn("Pick a text file first."); return; }
        string name = _name.Text.Trim();
        if (string.IsNullOrEmpty(name)) { Warn("Enter a model name."); return; }
        // The size list comes from /health; if it never loaded, don't silently train an unrequested default.
        if (_size.Selected < 0) { Warn("Connect to the server first — no sizes loaded yet."); return; }
        TrainRequested?.Invoke(new TrainRequest(name, _text, SelectedSize(), (int)_steps.Value, _backend.SelectedId));
    }

    private void Warn(string message)
    {
        _statusLabel.AddThemeColorOverride("font_color", Palette.Bad);
        _statusLabel.Text = message;
    }

    private string SelectedSize() => _size.Selected >= 0 ? _size.GetItemMetadata(_size.Selected).AsString() : "small";

    private static string Sanitize(string s)
    {
        var sb = new System.Text.StringBuilder();
        foreach (char c in s)
            if (char.IsLetterOrDigit(c) || c is '-' or '_') sb.Append(c);
        return sb.Length > 0 ? sb.ToString() : "mymodel";
    }
}
