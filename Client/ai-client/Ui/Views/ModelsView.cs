using Godot;

// The Models destination. The model-card grid arrives with the Models hub milestone; today this view hosts the
// re-homed TrainPanel ("Train new model" is a Models action per docs/CLIENT_DESIGN.md — not a top-level tab) and
// owns the training orchestration that used to live in Main: start the job, poll /train/status while it runs,
// and refresh /health on a terminal state so the finished model appears in every picker at once.
public partial class ModelsView : VBoxContainer, IView
{
    private readonly AppState _state;
    private readonly IApiClient _api;

    private TrainPanel _trainPanel;
    private Godot.Timer _pollTimer;

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
        var headerColumn = new VBoxContainer();
        headerColumn.AddThemeConstantOverride("separation", Palette.Space.Xs);
        headerColumn.AddChild(Palette.Heading("Models", Palette.Type.H3));
        headerColumn.AddChild(Palette.Heading(
            "Model cards are on the way. Train a new model below — it appears in the chat picker when it finishes.",
            Palette.Type.Caption, Palette.Muted));
        header.AddChild(headerColumn);
        AddChild(header);

        AddChild(Palette.Pad(_trainPanel = new TrainPanel(), left: 24, right: 24, top: 16, bottom: 8));
        AddChild(new Control { SizeFlagsVertical = SizeFlags.ExpandFill });

        // Polls /train/status while a job runs. CheckTrainStatus is a self-deduping poll (ApiClient drops it while
        // one is already in flight), so no busy guard is needed here.
        _pollTimer = new Godot.Timer { WaitTime = 1.5, OneShot = false, Autostart = false };
        _pollTimer.Timeout += () => _api.CheckTrainStatus();
        AddChild(_pollTimer);

        _trainPanel.TrainRequested += OnTrainRequested;
        _api.TrainStarted += OnTrainStarted;
        _state.JobsChanged += OnTrainStatus;
        _state.HealthChanged += OnHealth;
        OnHealth(); // render whatever state already exists — a view registered after the first /health must not start empty
    }

    private void OnHealth()
    {
        if (_state.Health is not { Ok: true } health) return;
        _trainPanel.SetBackends(health.Backends, health.DefaultBackend);
        _trainPanel.SetSizes(health.Sizes);
    }

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
