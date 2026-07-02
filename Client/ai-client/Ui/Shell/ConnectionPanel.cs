using System;
using Godot;

// The app-global server connection panel, pinned at the bottom of the nav rail (extracted from the old Sidebar):
// URL field, Check button, status line. It renders connection state from AppState (HealthChanged) and emits only
// intent events — Main routes UrlEdited into AppState and CheckRequested into the API client.
public partial class ConnectionPanel : PanelContainer
{
    public event Action CheckRequested;
    public event Action<string> UrlEdited;

    private readonly AppState _state;
    private LineEdit _url;
    private Label _status;

    public ConnectionPanel(AppState state) => _state = state;

    public override void _Ready()
    {
        Palette.StylePanel(this, Palette.PanelBg, radius: 10, pad: 10);

        var column = new VBoxContainer();
        column.AddThemeConstantOverride("separation", 6);
        AddChild(column);

        column.AddChild(Palette.Heading("Server", Palette.Type.Caption, Palette.Muted));

        _url = new LineEdit { Text = _state.ServerUrl, SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _url.TextChanged += text => UrlEdited?.Invoke(text);
        column.AddChild(_url);

        var check = new Button { Text = "Check connection" };
        check.Pressed += () => CheckRequested?.Invoke();
        column.AddChild(check);

        _status = new Label { Text = "Not connected", AutowrapMode = TextServer.AutowrapMode.WordSmart };
        _status.AddThemeColorOverride("font_color", Palette.Muted);
        _status.AddThemeFontSizeOverride("font_size", Palette.Type.Caption);
        column.AddChild(_status);

        _state.HealthChanged += Refresh;
        Refresh(); // render whatever state already exists (no-op before the first /health lands)
    }

    private void Refresh()
    {
        var health = _state.Health;
        if (health is null) return;
        if (!health.Ok)
        {
            SetStatus(health.Error, error: true);
            return;
        }
        int backends = 0;
        foreach (var b in health.Backends) if (b.Available) backends++;
        SetStatus(
            $"Connected ✓  —  {health.Models.Length} model{(health.Models.Length == 1 ? "" : "s")}, {backends} backend{(backends == 1 ? "" : "s")}",
            error: false);
    }

    private void SetStatus(string message, bool error)
    {
        _status.Text = message;
        _status.AddThemeColorOverride("font_color", error ? Palette.Bad : Palette.Good);
    }
}
