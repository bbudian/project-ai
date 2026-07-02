using Godot;

// Composition root for the ProjectAI client — wiring only. Builds the shell (NavRail | ViewHost), constructs the
// views and the shared AppState, and connects the API client's events to the store. Every behavior lives in a
// single-responsibility component: views own their logic and read shared state from AppState (never each other),
// the rail and host own navigation, and the connection panel owns the server field. Adding a destination is one
// Register + one AddDestination call here.
public partial class Main : Control
{
    private IApiClient _api;
    private AppState _state;

    public override void _Ready()
    {
        var apiNode = new ApiClient();
        AddChild(apiNode);
        _api = apiNode;

        _state = new AppState();
        _api.BaseUrl = _state.ServerUrl;

        // API results flow into the store; views subscribe to the store, never to each other.
        _api.HealthReceived += health => _state.SetHealth(health);
        _api.TrainStatusReceived += status => _state.SetTrainStatus(status);
        _state.ServerUrlChanged += () => _api.BaseUrl = _state.ServerUrl;

        BuildShell();

        _api.CheckHealth(); // attempt to connect + populate the model/backend pickers on launch
    }

    private void BuildShell()
    {
        SetAnchorsPreset(LayoutPreset.FullRect);

        var background = new PanelContainer();
        background.SetAnchorsPreset(LayoutPreset.FullRect);
        Palette.StylePanel(background, Palette.AppBg);
        AddChild(background);

        var shell = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        shell.AddThemeConstantOverride("separation", 0);
        background.AddChild(shell);

        var rail = new NavRail();
        shell.AddChild(rail);

        var connection = new ConnectionPanel(_state);
        connection.UrlEdited += url => _state.SetServerUrl(url);
        connection.CheckRequested += () => _api.CheckHealth();
        rail.AddFooter(connection);

        var host = new ViewHost();
        shell.AddChild(host);

        host.Register(ViewIds.Chat, new ChatView(_state));
        host.Register(ViewIds.Models, new ModelsView(_state, _api));
        rail.AddDestination(ViewIds.Chat, "💬   Chat");
        rail.AddDestination(ViewIds.Models, "◫   Models");

        rail.Navigated += host.Show;
        host.Shown += rail.SetActive;
        host.Show(ViewIds.Chat);
    }
}
