using Godot;

// The Memory destination (registered now that the server ships its read endpoints — the CLIENT_DESIGN reservation
// is satisfied): browse/search a store's memory catalog, preview exactly what would be injected for a message
// (bridge + recall via GET /memory/render), and manually inject a memory (PUT /memory) — the write path's first
// real UI. Stores are per model by default (chat's convention), so the picker lists "default" + every model name.
public partial class MemoryView : HBoxContainer, IView
{
    private readonly AppState _state;
    private readonly IApiClient _api;

    private OptionButton _storePicker;
    private LineEdit _search;
    private VBoxContainer _list;
    private Label _countLabel;
    private Label _bridgePreview, _recallPreview;
    private LineEdit _title, _keys;
    private TextEdit _body;
    private OptionButton _tier, _trust;
    private Label _saveStatus;

    public MemoryView(AppState state, IApiClient api)
    {
        _state = state;
        _api = api;
    }

    public Control Root => this;
    public void OnShown() => Reload(); // memories may have been written by chat since the view was last visible
    public void OnHidden() { }

    public override void _Ready()
    {
        SizeFlagsHorizontal = SizeFlags.ExpandFill;
        SizeFlagsVertical = SizeFlags.ExpandFill;
        AddThemeConstantOverride("separation", 0);

        // Left: header + search + catalog list.
        var left = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        left.AddThemeConstantOverride("separation", 0);
        AddChild(left);

        var header = new PanelContainer();
        Palette.StylePanel(header, Palette.AppBg, pad: 14);
        var headerRow = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        headerRow.AddThemeConstantOverride("separation", Palette.Space.Sm);
        var title = Palette.Heading("Memory", Palette.Type.H3);
        title.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        headerRow.AddChild(title);
        headerRow.AddChild(Palette.Heading("store", Palette.Type.Caption, Palette.Muted));
        _storePicker = new OptionButton { CustomMinimumSize = new Vector2(220, 0), TooltipText = "Memory store (chat uses the model's name)" };
        _storePicker.ItemSelected += _ => Reload();
        headerRow.AddChild(_storePicker);
        header.AddChild(headerRow);
        left.AddChild(header);

        var leftColumn = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill, SizeFlagsVertical = SizeFlags.ExpandFill };
        leftColumn.AddThemeConstantOverride("separation", Palette.Space.Md);
        left.AddChild(Palette.Pad(leftColumn, left: 24, right: 12, top: 12, bottom: 12, expand: true));

        var searchRow = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        searchRow.AddThemeConstantOverride("separation", Palette.Space.Sm);
        _search = new LineEdit { PlaceholderText = "Search memories… (Enter)", SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _search.TextSubmitted += _ => Reload();
        searchRow.AddChild(_search);
        _countLabel = Palette.Heading("", Palette.Type.Caption, Palette.Muted);
        searchRow.AddChild(_countLabel);
        leftColumn.AddChild(searchRow);

        var scroll = new ScrollContainer
        {
            SizeFlagsVertical = SizeFlags.ExpandFill,
            HorizontalScrollMode = ScrollContainer.ScrollMode.Disabled,
        };
        _list = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _list.AddThemeConstantOverride("separation", Palette.Space.Sm);
        scroll.AddChild(_list);
        leftColumn.AddChild(scroll);

        // Right: injection preview + inject form.
        var right = new VBoxContainer { CustomMinimumSize = new Vector2(400, 0) };
        right.AddThemeConstantOverride("separation", Palette.Space.Md);
        AddChild(Palette.Pad(right, left: 12, right: 24, top: 12, bottom: 12, expand: false));

        right.AddChild(BuildPreviewCard());
        right.AddChild(BuildInjectCard());

        _api.MemoryListReceived += OnList;
        _api.MemoryRenderReceived += OnRender;
        _api.MemorySaved += OnSaved;
        _state.HealthChanged += RefreshStores;
        RefreshStores();
    }

    private string SelectedStore() =>
        _storePicker is { Selected: >= 0 } ? _storePicker.GetItemText(_storePicker.Selected) : "default";

    // Store options: "default" + one per model (chat's per-model convention). Keeps the current pick across refreshes.
    private void RefreshStores()
    {
        string previous = SelectedStore();
        _storePicker.Clear();
        _storePicker.AddItem("default");
        int keep = previous == "default" ? 0 : -1, preferSelected = -1;
        if (_state.Health is { Ok: true } health)
            for (int i = 0; i < health.Models.Length; i++)
            {
                _storePicker.AddItem(health.Models[i]);
                if (health.Models[i] == previous) keep = i + 1;
                if (health.Models[i] == _state.SelectedModel) preferSelected = i + 1;
            }
        _storePicker.Selected = Selection.FirstValid(keep, preferSelected, 0);
        Reload();
    }

    private void Reload()
    {
        _countLabel.Text = "loading…";
        _api.MemoryList(SelectedStore(), _search.Text);
    }

    private void OnList(MemoryListResult result)
    {
        foreach (var child in _list.GetChildren()) child.QueueFree();
        if (!result.Ok)
        {
            _countLabel.Text = "";
            _list.AddChild(Palette.Heading($"Could not load memories: {result.Error}", Palette.Type.Label, Palette.Bad));
            return;
        }
        _countLabel.Text = result.Count == 1 ? "1 memory" : $"{result.Count} memories";
        if (result.Memories.Length == 0)
        {
            _list.AddChild(Palette.EmptyState("🧠", "No memories",
                string.IsNullOrEmpty(_search.Text)
                    ? "Inject one on the right, or chat with Memory on — the model's store fills as you talk."
                    : $"Nothing matches \"{_search.Text}\"."));
            return;
        }
        foreach (var card in result.Memories) _list.AddChild(BuildMemoryCard(card));
    }

    private static Control BuildMemoryCard(MemoryCardInfo info)
    {
        var body = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        body.AddThemeConstantOverride("separation", Palette.Space.Xs);

        var row = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        row.AddThemeConstantOverride("separation", Palette.Space.Sm);
        var title = Palette.Heading(info.Title, Palette.Type.Body);
        title.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        title.AutowrapMode = TextServer.AutowrapMode.WordSmart;
        row.AddChild(title);
        row.AddChild(Palette.Badge(info.Tier, info.Tier == "core" ? Palette.Tone.Accent : Palette.Tone.Neutral));
        row.AddChild(Palette.Badge(info.Trust, info.Trust switch
        {
            "curated" => Palette.Tone.Good,
            "untrusted" => Palette.Tone.Bad,
            _ => Palette.Tone.Neutral,
        }));
        body.AddChild(row);

        string meta = (info.Keys.Length > 0 ? $"[{string.Join(", ", info.Keys)}]   " : "") +
                      (string.IsNullOrEmpty(info.AsOf) ? "" : $"as of {info.AsOf}") + $"   ·   {info.Id}";
        body.AddChild(Palette.Heading(meta, Palette.Type.Caption, Palette.Muted));
        return Palette.Card(body, pad: Palette.Space.Md);
    }

    // ---- injection preview ------------------------------------------------------------------------------------

    private Control BuildPreviewCard()
    {
        var body = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        body.AddThemeConstantOverride("separation", Palette.Space.Sm);

        var go = Palette.GhostButton("👁  Preview what a message would inject", Palette.Type.Label);
        go.TooltipText = "Renders the pinned bridge plus the recall block for the search text above, with the exact budgets chat uses.";
        go.Pressed += () =>
        {
            _bridgePreview.Text = "rendering…";
            _recallPreview.Text = "";
            _api.MemoryRender(SelectedStore(), _search.Text);
        };
        body.AddChild(go);
        _bridgePreview = Preview(body);
        _recallPreview = Preview(body);
        return Palette.Card(body, "Injection preview");
    }

    private static Label Preview(VBoxContainer host)
    {
        var label = Palette.Heading("", Palette.Type.Caption, Palette.Muted);
        label.AutowrapMode = TextServer.AutowrapMode.WordSmart;
        label.CustomMinimumSize = new Vector2(360, 0);
        host.AddChild(label);
        return label;
    }

    private void OnRender(MemoryRenderResult result)
    {
        if (!result.Ok) { _bridgePreview.Text = $"Error: {result.Error}"; return; }
        _bridgePreview.Text = string.IsNullOrEmpty(result.Bridge) ? "(bridge: empty — no pinned/core memories)" : result.Bridge;
        _recallPreview.Text = string.IsNullOrEmpty(result.Recall)
            ? "(recall: nothing relevant to the search text)"
            : result.Recall;
    }

    // ---- manual inject ----------------------------------------------------------------------------------------

    private Control BuildInjectCard()
    {
        var body = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        body.AddThemeConstantOverride("separation", Palette.Space.Sm);

        _title = new LineEdit { PlaceholderText = "Title (e.g. \"Ben's GPU is an 8GB 3070\")" };
        body.AddChild(Palette.Field("Title", _title));
        _keys = new LineEdit { PlaceholderText = "comma, separated, keys" };
        body.AddChild(Palette.Field("Keys", _keys));
        _body = new TextEdit
        {
            CustomMinimumSize = new Vector2(0, 84),
            WrapMode = TextEdit.LineWrappingMode.Boundary,
            PlaceholderText = "The fact itself — one durable note.",
        };
        body.AddChild(_body);

        _tier = new OptionButton();
        foreach (var t in new[] { "long", "core", "session" }) _tier.AddItem(t);
        body.AddChild(Palette.Field("Tier", _tier));
        _trust = new OptionButton();
        foreach (var t in new[] { "curated", "chat", "untrusted" }) _trust.AddItem(t);
        body.AddChild(Palette.Field("Trust", _trust));

        var save = Palette.PrimaryButton("Inject memory");
        save.Pressed += OnInject;
        body.AddChild(save);
        _saveStatus = Palette.Heading("", Palette.Type.Caption, Palette.Muted);
        _saveStatus.AutowrapMode = TextServer.AutowrapMode.WordSmart;
        body.AddChild(_saveStatus);

        return Palette.Card(body, "Inject a memory");
    }

    private void OnInject()
    {
        if (string.IsNullOrWhiteSpace(_title.Text) && string.IsNullOrWhiteSpace(_body.Text))
        {
            _saveStatus.AddThemeColorOverride("font_color", Palette.Bad);
            _saveStatus.Text = "A memory needs a title or a body.";
            return;
        }
        var keys = _keys.Text.Split(',', System.StringSplitOptions.RemoveEmptyEntries | System.StringSplitOptions.TrimEntries);
        _saveStatus.AddThemeColorOverride("font_color", Palette.Muted);
        _saveStatus.Text = "Saving…";
        _api.MemoryPut(SelectedStore(), _title.Text.Trim(), keys, _body.Text.Trim(),
            _tier.GetItemText(_tier.Selected), _trust.GetItemText(_trust.Selected));
    }

    private void OnSaved(MemorySaveResult result)
    {
        if (!result.Ok)
        {
            _saveStatus.AddThemeColorOverride("font_color", Palette.Bad);
            _saveStatus.Text = $"Error: {result.Error}";
            return;
        }
        _saveStatus.AddThemeColorOverride("font_color", Palette.Good);
        _saveStatus.Text = $"Saved {result.Id}.";
        _title.Text = "";
        _keys.Text = "";
        _body.Text = "";
        Reload();
    }
}
