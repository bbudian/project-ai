using System.Linq;
using Godot;

// The Benchmark destination (docs/CLIENT_DESIGN.md §2.3): Define (pick suite/models/backend, run) → progress with
// cancel → Compare (aggregates + per-case grid, row-activate opens the side-by-side outputs) → Reports (past runs).
// Rigor defaults are baked in: greedy decoding, fixed seed, repeats with a warmup discard — the server enforces
// them; this view only ever displays what the run records actually say (every rate carries its n).
public partial class BenchmarkView : VBoxContainer, IView
{
    private readonly AppState _state;
    private readonly IApiClient _api;

    private Button _tabDefine, _tabCompare, _tabReports;
    private Control _defineTab, _compareTab, _reportsTab;

    // Define
    private OptionButton _suite;
    private ItemList _models;
    private BackendPicker _backend;
    private SpinBox _repeats;
    private Button _run, _cancel;
    private ProgressRow _progress;
    private BenchSuiteInfo[] _suites = [];

    // Compare
    private Label _compareTitle;
    private DataTable _aggregates, _cases;
    private BenchRunDetail _current;
    private string[] _caseIds = [];
    private PopupPanel _diffPopup;
    private VBoxContainer _diffBody;

    // Reports
    private VBoxContainer _runsList;

    private Godot.Timer _pollTimer;
    private bool _running;

    public BenchmarkView(AppState state, IApiClient api)
    {
        _state = state;
        _api = api;
    }

    public Control Root => this;

    public void OnShown()
    {
        _api.FetchBenchSuites();
        _api.FetchBenchRuns();
        _api.CheckBenchStatus(); // a run may already be live (started elsewhere)
    }

    public void OnHidden() { }

    public override void _Ready()
    {
        SizeFlagsHorizontal = SizeFlags.ExpandFill;
        SizeFlagsVertical = SizeFlags.ExpandFill;
        AddThemeConstantOverride("separation", 0);

        var header = new PanelContainer();
        Palette.StylePanel(header, Palette.AppBg, pad: 14);
        var headerRow = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        headerRow.AddThemeConstantOverride("separation", Palette.Space.Sm);
        var title = Palette.Heading("Benchmark", Palette.Type.H3);
        title.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        headerRow.AddChild(title);
        _tabDefine = TabButton("Define");
        _tabCompare = TabButton("Compare");
        _tabReports = TabButton("Reports");
        _tabDefine.Pressed += () => ShowTab(0);
        _tabCompare.Pressed += () => ShowTab(1);
        _tabReports.Pressed += () => ShowTab(2);
        headerRow.AddChild(_tabDefine);
        headerRow.AddChild(_tabCompare);
        headerRow.AddChild(_tabReports);
        header.AddChild(headerRow);
        AddChild(header);

        var scroll = new ScrollContainer
        {
            SizeFlagsVertical = SizeFlags.ExpandFill,
            HorizontalScrollMode = ScrollContainer.ScrollMode.Disabled,
        };
        var content = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        content.AddThemeConstantOverride("separation", Palette.Space.Md);
        scroll.AddChild(content);
        AddChild(Palette.Pad(scroll, left: 24, right: 24, top: 12, bottom: 12, expand: true));

        content.AddChild(_defineTab = BuildDefineTab());
        content.AddChild(_compareTab = BuildCompareTab());
        content.AddChild(_reportsTab = BuildReportsTab());

        _pollTimer = new Godot.Timer { WaitTime = 1.5, OneShot = false, Autostart = false };
        _pollTimer.Timeout += () => _api.CheckBenchStatus();
        AddChild(_pollTimer);

        _api.BenchStarted += OnBenchStarted;
        _api.BenchStatusReceived += OnBenchStatus;
        _api.BenchSuitesReceived += OnSuites;
        _api.BenchRunsReceived += OnRuns;
        _api.BenchRunReceived += OnRunDetail;
        _state.HealthChanged += OnHealth;

        ShowTab(0);
        OnHealth();
    }

    private static Button TabButton(string label) => Palette.GhostButton(label, Palette.Type.Label);

    private void ShowTab(int index)
    {
        _defineTab.Visible = index == 0;
        _compareTab.Visible = index == 1;
        _reportsTab.Visible = index == 2;
        foreach (var (button, i) in new[] { (_tabDefine, 0), (_tabCompare, 1), (_tabReports, 2) })
            button.AddThemeColorOverride("font_color", i == index ? Palette.Accent : Palette.Text);
    }

    private void OnHealth()
    {
        if (_state.Health is not { Ok: true } health) return;
        string previous = string.Join(",", SelectedModels());
        _models.Clear();
        foreach (var info in health.Infos)
        {
            int idx = _models.AddItem(info.Name);
            if (previous.Contains(info.Name) || (previous.Length == 0 && info.Name == _state.SelectedModel))
                _models.Select(idx, single: false);
        }
        _backend.SetBackends(health.Backends, health.DefaultBackend);
    }

    // ---- Define -----------------------------------------------------------------------------------------------

    private Control BuildDefineTab()
    {
        var body = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        body.AddThemeConstantOverride("separation", Palette.Space.Sm);

        _suite = new OptionButton();
        body.AddChild(Palette.Field("Suite", _suite));

        _models = new ItemList
        {
            SelectMode = ItemList.SelectModeEnum.Multi,
            CustomMinimumSize = new Vector2(0, 110),
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        _models.TooltipText = "Ctrl/Shift-click to select several models — the whole point is comparing.";
        body.AddChild(Palette.Field("Models", _models));

        _backend = new BackendPicker();
        body.AddChild(Palette.Field("Backend", _backend));

        _repeats = new SpinBox { MinValue = 1, MaxValue = 20, Value = 3, CustomMinimumSize = new Vector2(110, 0) };
        body.AddChild(Palette.Field("Repeats", _repeats));
        body.AddChild(Palette.Heading(
            "Greedy decoding, fixed seed, median over repeats with one discarded warmup — held constant across models.",
            Palette.Type.Caption, Palette.Muted));

        var buttons = new HBoxContainer();
        buttons.AddThemeConstantOverride("separation", Palette.Space.Sm);
        _run = Palette.PrimaryButton("Run benchmark");
        _run.Pressed += OnRun;
        buttons.AddChild(_run);
        _cancel = Palette.GhostButton("Cancel", Palette.Type.Label);
        _cancel.Visible = false;
        _cancel.Pressed += () => _api.CancelBenchmark();
        buttons.AddChild(_cancel);
        body.AddChild(buttons);

        _progress = new ProgressRow();
        body.AddChild(_progress);

        return Palette.Card(body, "Run configuration");
    }

    private string[] SelectedModels()
    {
        if (_models is null) return [];
        var selected = _models.GetSelectedItems();
        var names = new string[selected.Length];
        for (int i = 0; i < selected.Length; i++) names[i] = _models.GetItemText(selected[i]);
        return names;
    }

    private void OnRun()
    {
        var models = SelectedModels();
        if (models.Length == 0) { _progress.SetStatus("Select at least one model.", Palette.Bad); return; }
        string suite = _suite is { Selected: >= 0 } && _suites.Length > 0 ? _suites[_suite.Selected].Id : "baseline";
        _run.Disabled = true;
        _progress.SetPercent(0);
        _progress.SetStatus("Starting…", Palette.Muted);
        _api.StartBenchmark(new BenchStartRequest(suite, models, _backend.SelectedId, (int)_repeats.Value));
    }

    private void OnBenchStarted(BenchStartResult result)
    {
        if (!result.Ok)
        {
            _run.Disabled = false;
            _progress.SetStatus($"Error: {result.Error}", Palette.Bad);
            return;
        }
        _running = true;
        _cancel.Visible = true;
        _progress.SetStatus($"Running {result.RunId} — 0/{result.Total}", Palette.Muted);
        _pollTimer.Start();
    }

    private void OnBenchStatus(BenchStatusInfo status)
    {
        switch (status.State)
        {
            case "running":
                if (!_running) { _running = true; _run.Disabled = true; _cancel.Visible = true; _pollTimer.Start(); }
                _progress.SetPercent(status.Total > 0 ? 100.0 * status.Done / status.Total : 0);
                _progress.SetStatus($"{status.Done}/{status.Total}  ·  {status.CurrentModel} · {status.CurrentCase}", Palette.Muted);
                break;
            case "done" or "canceled" or "error" when _running:
                _running = false;
                _pollTimer.Stop();
                _run.Disabled = false;
                _cancel.Visible = false;
                _progress.SetPercent(status.State == "done" ? 100 : 0);
                _progress.SetStatus(status.State == "error" ? $"Error: {status.Error}" : $"Run {status.State}.",
                    status.State == "error" ? Palette.Bad : Palette.Good);
                _api.FetchBenchRuns();
                if (status.State == "done" && status.RunId.Length > 0)
                {
                    _api.FetchBenchRun(status.RunId); // auto-open the finished run
                    ShowTab(1);
                }
                break;
        }
    }

    private void OnSuites(BenchSuiteInfo[] suites)
    {
        _suites = suites;
        string previous = _suite is { Selected: >= 0 } && _suite.ItemCount > 0 ? _suite.GetItemText(_suite.Selected) : "";
        _suite.Clear();
        int keep = -1;
        for (int i = 0; i < suites.Length; i++)
        {
            _suite.AddItem($"{suites[i].Id} — {suites[i].CaseCount} cases{(suites[i].HasCorpus ? " + bpb" : "")}");
            if (_suite.GetItemText(i) == previous) keep = i;
        }
        if (suites.Length > 0) _suite.Selected = keep >= 0 ? keep : 0;
    }

    // ---- Compare ----------------------------------------------------------------------------------------------

    private Control BuildCompareTab()
    {
        var body = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        body.AddThemeConstantOverride("separation", Palette.Space.Md);
        _compareTitle = Palette.Heading("No run loaded — run a benchmark or open one from Reports.", Palette.Type.Label, Palette.Muted);
        body.AddChild(_compareTitle);

        _aggregates = new DataTable();
        body.AddChild(Palette.Card(_aggregates, "Aggregates (bpb ↓ is the quality headline; every rate shows its n)"));

        _cases = new DataTable();
        _cases.RowActivated += OnCaseActivated;
        body.AddChild(Palette.Card(_cases, "Cases (click a row for the side-by-side outputs)"));

        _diffPopup = new PopupPanel();
        _diffPopup.AddThemeStyleboxOverride("panel",
            Palette.Box(Palette.PanelBg, radius: Palette.Radius.Md, pad: Palette.Space.Lg, border: 1, borderColor: Palette.Border));
        var diffScroll = new ScrollContainer { CustomMinimumSize = new Vector2(640, 420) };
        _diffBody = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _diffBody.AddThemeConstantOverride("separation", Palette.Space.Md);
        diffScroll.AddChild(_diffBody);
        _diffPopup.AddChild(diffScroll);
        AddChild(_diffPopup);

        return body;
    }

    private void OnRunDetail(BenchRunDetail run)
    {
        if (!run.Ok)
        {
            _compareTitle.Text = $"Could not load run: {run.Error}";
            return;
        }
        _current = run;
        _compareTitle.Text = $"Run {run.Id}  ·  suite {run.SuiteId}  ·  {run.State}";

        double bestBpb = run.Aggregates.Where(a => !double.IsNaN(a.Bpb)).Select(a => a.Bpb).DefaultIfEmpty(double.NaN).Min();
        double bestTok = run.Aggregates.Select(a => a.MedianTokPerSec).DefaultIfEmpty(0).Max();
        _aggregates.SetData(
            ["model", "bpb ↓", "median tok/s", "check pass", "n"],
            run.Aggregates.Select(a => new DataTable.Cell[]
            {
                new(a.Model, Bold: true),
                new(double.IsNaN(a.Bpb) ? "—" : a.Bpb.ToString("0.0000"),
                    !double.IsNaN(a.Bpb) && a.Bpb == bestBpb ? Palette.Tone.Good : null),
                new(a.MedianTokPerSec.ToString("0.00"), a.MedianTokPerSec == bestTok && bestTok > 0 ? Palette.Tone.Good : null),
                new(a.CheckPassRate.ToString("P0")),
                new(a.N.ToString()),
            }).ToArray());

        _caseIds = run.Cells.Select(c => c.CaseId).Distinct().ToArray();
        var models = run.Aggregates.Select(a => a.Model).ToArray();
        var rows = new DataTable.Cell[_caseIds.Length][];
        for (int i = 0; i < _caseIds.Length; i++)
        {
            var row = new DataTable.Cell[models.Length + 1];
            row[0] = new DataTable.Cell(_caseIds[i], Bold: true);
            for (int m = 0; m < models.Length; m++)
            {
                var cell = run.Cells.FirstOrDefault(c => c.CaseId == _caseIds[i] && c.Model == models[m]);
                row[m + 1] = cell is null
                    ? new DataTable.Cell("—", Palette.Tone.Neutral)
                    : cell.Error is not null
                        ? new DataTable.Cell("⚠ error", Palette.Tone.Bad)
                        : new DataTable.Cell($"{cell.CheckPassRate:P0} · {cell.MedianTokPerSec:0.0} tok/s · {cell.Stop}",
                            cell.CheckPassRate >= 1 ? Palette.Tone.Good : cell.CheckPassRate <= 0 ? Palette.Tone.Bad : null);
            }
            rows[i] = row;
        }
        _cases.SetData([.. new[] { "case" }.Concat(models)], rows);
    }

    private void OnCaseActivated(int rowIndex)
    {
        if (_current is null || rowIndex >= _caseIds.Length) return;
        string caseId = _caseIds[rowIndex];
        foreach (var child in _diffBody.GetChildren()) child.QueueFree();
        _diffBody.AddChild(Palette.Heading($"Case: {caseId}", Palette.Type.H3));
        foreach (var cell in _current.Cells.Where(c => c.CaseId == caseId))
        {
            var block = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
            block.AddThemeConstantOverride("separation", Palette.Space.Xs);
            block.AddChild(Palette.Heading(
                $"{cell.Model}  ·  {cell.CheckPassRate:P0} checks  ·  {cell.GeneratedTokens} tok  ·  stop {cell.Stop}",
                Palette.Type.Caption, Palette.Muted));
            var output = Palette.Heading(
                cell.Error is not null ? $"error: {cell.Error}" : cell.Output.Length > 0 ? cell.Output : "(no output)",
                Palette.Type.Label);
            output.AutowrapMode = TextServer.AutowrapMode.WordSmart;
            output.CustomMinimumSize = new Vector2(600, 0);
            block.AddChild(output);
            _diffBody.AddChild(Palette.Card(block, pad: Palette.Space.Md));
        }
        _diffPopup.PopupCentered();
    }

    // ---- Reports ----------------------------------------------------------------------------------------------

    private Control BuildReportsTab()
    {
        var body = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        body.AddThemeConstantOverride("separation", Palette.Space.Sm);
        _runsList = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _runsList.AddThemeConstantOverride("separation", Palette.Space.Sm);
        body.AddChild(_runsList);
        return Palette.Card(body, "Past runs");
    }

    private void OnRuns(BenchRunSummary[] runs)
    {
        foreach (var child in _runsList.GetChildren()) child.QueueFree();
        if (runs.Length == 0)
        {
            _runsList.AddChild(Palette.Heading("No runs yet — define one and hit Run.", Palette.Type.Label, Palette.Muted));
            return;
        }
        foreach (var run in runs)
        {
            var button = Palette.GhostButton(
                $"{run.Id}   ·   {run.SuiteId}   ·   {string.Join(", ", run.Models)}   ·   {run.Backend}   ·   {run.State}",
                Palette.Type.Label);
            string id = run.Id;
            button.Pressed += () => { _api.FetchBenchRun(id); ShowTab(1); };
            _runsList.AddChild(button);
        }
    }
}
