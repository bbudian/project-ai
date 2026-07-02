using Godot;

// The Models destination (docs/CLIENT_DESIGN.md §2.2): a card grid over the server's enriched /health catalog —
// params, layers, context, tokenizer kind, precision, step, instruct — with per-card actions (Chat with,
// Tokenize probe) and "+ Train new model" re-hosting the TrainPanel. This view also owns the training
// orchestration that used to live in Main: start the job, poll /train/status while it runs, and refresh /health
// on a terminal state so the finished model appears in every picker at once.
public partial class ModelsView : VBoxContainer, IView
{
    private readonly AppState _state;
    private readonly IApiClient _api;

    private GridContainer _grid;
    private Control _empty;
    private Button _trainToggle;
    private Control _trainSection;
    private TrainPanel _trainPanel;
    private Godot.Timer _pollTimer;
    private PopupPanel _tokenizePopup;
    private LineEdit _tokenizeInput;
    private Label _tokenizeResult;
    private string _tokenizeModel = "";

    public ModelsView(AppState state, IApiClient api)
    {
        _state = state;
        _api = api;
    }

    public Control Root => this;
    public void OnShown() { }  // the poll timer tracks the job, not the view: progress keeps updating while hidden
    public void OnHidden() { }

    public override void _Ready()
    {
        SizeFlagsHorizontal = SizeFlags.ExpandFill;
        SizeFlagsVertical = SizeFlags.ExpandFill;
        AddThemeConstantOverride("separation", 0);

        var header = new PanelContainer();
        Palette.StylePanel(header, Palette.AppBg, pad: 14);
        var headerRow = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        var title = Palette.Heading("Models", Palette.Type.H3);
        title.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        headerRow.AddChild(title);
        _trainToggle = Palette.GhostButton("＋   Train new model");
        _trainToggle.Pressed += () => SetTrainVisible(!_trainSection.Visible);
        headerRow.AddChild(_trainToggle);
        header.AddChild(headerRow);
        AddChild(header);

        var scroll = new ScrollContainer
        {
            SizeFlagsVertical = SizeFlags.ExpandFill,
            HorizontalScrollMode = ScrollContainer.ScrollMode.Disabled,
        };
        var column = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        column.AddThemeConstantOverride("separation", Palette.Space.Md);
        scroll.AddChild(column);
        AddChild(Palette.Pad(scroll, left: 24, right: 24, top: 12, bottom: 12, expand: true));

        _grid = new GridContainer { Columns = 2, SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _grid.AddThemeConstantOverride("h_separation", Palette.Space.Md);
        _grid.AddThemeConstantOverride("v_separation", Palette.Space.Md);
        column.AddChild(_grid);

        _empty = Palette.EmptyState("◫", "No models yet",
            "Train one below, or convert a HuggingFace model with `projectai convert` and point --models at it.");
        _empty.Visible = false;
        column.AddChild(_empty);

        // The re-hosted TrainPanel, collapsed until requested (or a job is running).
        _trainSection = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill, Visible = false };
        _trainSection.AddChild(_trainPanel = new TrainPanel());
        column.AddChild(_trainSection);

        BuildTokenizePopup();

        // Polls /train/status while a job runs. CheckTrainStatus is a self-deduping poll (ApiClient drops it while
        // one is already in flight), so no busy guard is needed here.
        _pollTimer = new Godot.Timer { WaitTime = 1.5, OneShot = false, Autostart = false };
        _pollTimer.Timeout += () => _api.CheckTrainStatus();
        AddChild(_pollTimer);

        _trainPanel.TrainRequested += OnTrainRequested;
        _api.TrainStarted += OnTrainStarted;
        _api.TokenizeReceived += OnTokenizeResult;
        _state.JobsChanged += OnTrainStatus;
        _state.HealthChanged += Rebuild;
        Rebuild(); // render whatever state already exists — a view registered after the first /health must not start empty
    }

    private void SetTrainVisible(bool visible)
    {
        _trainSection.Visible = visible;
        _trainToggle.Text = visible ? "－   Hide training" : "＋   Train new model";
    }

    // ---- the card grid ----------------------------------------------------------------------------------------

    private void Rebuild()
    {
        foreach (var child in _grid.GetChildren()) child.QueueFree();
        if (_state.Health is not { Ok: true } health) return;

        _trainPanel.SetBackends(health.Backends, health.DefaultBackend);
        _trainPanel.SetSizes(health.Sizes);

        _empty.Visible = health.Infos.Length == 0;
        foreach (var info in health.Infos) _grid.AddChild(BuildCard(info, info.Name == health.Default));
    }

    private Control BuildCard(ModelInfo info, bool isDefault)
    {
        var body = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        body.AddThemeConstantOverride("separation", Palette.Space.Sm);

        var titleRow = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        titleRow.AddThemeConstantOverride("separation", Palette.Space.Sm);
        var name = Palette.Heading(info.Name, Palette.Type.Body);
        name.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        titleRow.AddChild(name);
        if (isDefault) titleRow.AddChild(Palette.Badge("default", Palette.Tone.Accent));
        if (info.Instruct) titleRow.AddChild(Palette.Badge("instruct", Palette.Tone.Good));
        body.AddChild(titleRow);

        var meta = Palette.Heading(DescribeMeta(info), Palette.Type.Caption, Palette.Muted);
        meta.AutowrapMode = TextServer.AutowrapMode.WordSmart;
        body.AddChild(meta);
        if (info.Error is not null)
        {
            var error = Palette.Heading($"metadata unreadable: {info.Error}", Palette.Type.Caption, Palette.Bad);
            error.AutowrapMode = TextServer.AutowrapMode.WordSmart;
            body.AddChild(error);
        }

        var actions = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        actions.AddThemeConstantOverride("separation", Palette.Space.Sm);
        var chat = Palette.GhostButton("💬  Chat with", Palette.Type.Label);
        chat.Pressed += () =>
        {
            _state.SetSelection(info.Name, _state.SelectedBackend);
            _state.RequestNavigate(ViewIds.Chat);
        };
        actions.AddChild(chat);
        var tokenize = Palette.GhostButton("⊟  Tokenize…", Palette.Type.Label);
        tokenize.Pressed += () => OpenTokenize(info.Name);
        actions.AddChild(tokenize);
        body.AddChild(actions);

        var card = Palette.Card(body);
        card.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        return card;
    }

    private static string DescribeMeta(ModelInfo info)
    {
        if (info.Params <= 0) return info.FileBytes > 0 ? $"{FormatBytes(info.FileBytes)} on disk" : "";
        return $"{FormatParams(info.Params)} params  ·  {info.Layers} layers  ·  ctx {info.Ctx:N0}  ·  " +
               $"{info.Tokenizer} tokenizer  ·  {info.Dtype}  ·  step {info.Step:N0}  ·  {FormatBytes(info.FileBytes)}";
    }

    private static string FormatParams(long p) => p switch
    {
        >= 1_000_000_000 => $"{p / 1e9:0.0}B",
        >= 1_000_000 => $"{p / 1e6:0}M",
        _ => p.ToString("N0"),
    };

    private static string FormatBytes(long b) => b switch
    {
        >= 1L << 30 => $"{b / (double)(1L << 30):0.0} GB",
        >= 1L << 20 => $"{b / (double)(1L << 20):0} MB",
        _ => $"{b:N0} B",
    };

    // ---- tokenize probe ---------------------------------------------------------------------------------------

    private void BuildTokenizePopup()
    {
        _tokenizePopup = new PopupPanel();
        _tokenizePopup.AddThemeStyleboxOverride("panel",
            Palette.Box(Palette.PanelBg, radius: Palette.Radius.Md, pad: Palette.Space.Lg, border: 1, borderColor: Palette.Border));
        AddChild(_tokenizePopup);

        var col = new VBoxContainer { CustomMinimumSize = new Vector2(480, 0) };
        col.AddThemeConstantOverride("separation", Palette.Space.Sm);
        _tokenizePopup.AddChild(col);

        col.AddChild(Palette.Heading("Tokenize probe", Palette.Type.Caption, Palette.Muted));
        _tokenizeInput = new LineEdit { PlaceholderText = "Type text, press Enter…", SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _tokenizeInput.TextSubmitted += _ => RunTokenize();
        col.AddChild(_tokenizeInput);
        _tokenizeResult = Palette.Heading("", Palette.Type.Label, Palette.Muted);
        _tokenizeResult.AutowrapMode = TextServer.AutowrapMode.WordSmart;
        _tokenizeResult.CustomMinimumSize = new Vector2(480, 0);
        col.AddChild(_tokenizeResult);
    }

    private void OpenTokenize(string model)
    {
        _tokenizeModel = model;
        _tokenizeResult.Text = $"model: {model}";
        _tokenizePopup.PopupCentered();
        _tokenizeInput.GrabFocus();
    }

    private void RunTokenize()
    {
        if (string.IsNullOrWhiteSpace(_tokenizeInput.Text)) return;
        _tokenizeResult.Text = "tokenizing…";
        _api.Tokenize(_tokenizeModel, _tokenizeInput.Text);
    }

    private void OnTokenizeResult(TokenizeResult result)
    {
        if (!result.Ok) { _tokenizeResult.Text = $"Error: {result.Error}"; return; }
        string joined = string.Join(" | ", result.Pieces is { Length: > 64 } ? result.Pieces[..64] : result.Pieces);
        if (joined.Length > 700) joined = joined[..700] + "…";
        _tokenizeResult.Text = $"{result.Count} tokens\n{joined}";
    }

    // ---- training orchestration -------------------------------------------------------------------------------

    private void OnTrainRequested(TrainRequest request)
    {
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

    private void OnTrainStatus()
    {
        if (_state.TrainStatus is not { } status) return;
        if (status.State == "running" && !_trainSection.Visible) SetTrainVisible(true); // surface live progress
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
}
