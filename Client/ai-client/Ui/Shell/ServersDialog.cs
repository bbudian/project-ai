using Godot;

// The "Servers…" dialog: every running ProjectAI server on this machine (self-registered + port-scanned), each
// with Connect and Stop. Connect points the whole app at that server (URL + immediate health refresh); Stop asks
// the server to shut down gracefully over PUT /shutdown. Stale registry leftovers (a crash skips deregistration)
// are shown as such instead of hidden, so the user understands what they're looking at.
public partial class ServersDialog : PopupPanel
{
    private readonly AppState _state;
    private readonly IApiClient _api;
    private readonly ServerController _controller;
    private VBoxContainer _list;
    private Label _status;

    public ServersDialog(AppState state, IApiClient api, ServerController controller)
    {
        _state = state;
        _api = api;
        _controller = controller;
    }

    public override void _Ready()
    {
        AddThemeStyleboxOverride("panel",
            Palette.Box(Palette.PanelBg, radius: Palette.Radius.Md, pad: Palette.Space.Lg, border: 1, borderColor: Palette.Border));

        var column = new VBoxContainer { CustomMinimumSize = new Vector2(520, 0) };
        column.AddThemeConstantOverride("separation", Palette.Space.Sm);
        AddChild(column);

        var headerRow = new HBoxContainer { SizeFlagsHorizontal = Control.SizeFlags.ExpandFill };
        var title = Palette.Heading("Running servers", Palette.Type.Caption, Palette.Muted);
        title.SizeFlagsHorizontal = Control.SizeFlags.ExpandFill;
        headerRow.AddChild(title);
        var refresh = Palette.GhostButton("↻  Refresh", Palette.Type.Label);
        refresh.Pressed += Refresh;
        headerRow.AddChild(refresh);
        column.AddChild(headerRow);

        _list = new VBoxContainer { SizeFlagsHorizontal = Control.SizeFlags.ExpandFill };
        _list.AddThemeConstantOverride("separation", Palette.Space.Sm);
        column.AddChild(_list);

        _status = Palette.Heading("", Palette.Type.Caption, Palette.Muted);
        column.AddChild(_status);
    }

    public void Open()
    {
        PopupCentered();
        Refresh();
    }

    private async void Refresh()
    {
        _status.Text = "Scanning the registry + ports 8080-8089…";
        foreach (var child in _list.GetChildren()) child.QueueFree();

        var servers = await _controller.DiscoverAsync();
        foreach (var child in _list.GetChildren()) child.QueueFree(); // a second Refresh may have raced this one

        if (servers.Count == 0)
        {
            _status.Text = "No running servers found. Start one below, or run `projectai serve`.";
            return;
        }
        _status.Text = $"{servers.Count} found.";

        foreach (var server in servers)
        {
            var row = new HBoxContainer { SizeFlagsHorizontal = Control.SizeFlags.ExpandFill };
            row.AddThemeConstantOverride("separation", Palette.Space.Sm);

            string describe = server.Alive
                ? $"{server.Url}   ·   {server.Models} model{(server.Models == 1 ? "" : "s")}" +
                  (server.Backend.Length > 0 ? $"   ·   {server.Backend}" : "") +
                  (server.Pid > 0 ? $"   ·   pid {server.Pid}" : "")
                : $"{server.Url}   ·   {server.Source}";
            var label = Palette.Heading(describe, Palette.Type.Label, server.Alive ? Palette.Text : Palette.Muted);
            label.SizeFlagsHorizontal = Control.SizeFlags.ExpandFill;
            label.AutowrapMode = TextServer.AutowrapMode.WordSmart;
            row.AddChild(label);

            bool isCurrent = NormalizeUrl(_state.ServerUrl) == NormalizeUrl(server.Url);
            if (server.Alive)
            {
                if (isCurrent)
                {
                    row.AddChild(Palette.Badge("connected", Palette.Tone.Accent));
                }
                else
                {
                    var connect = Palette.GhostButton("Connect", Palette.Type.Label);
                    connect.Pressed += () =>
                    {
                        _state.SetServerUrl(server.Url);
                        _api.CheckHealth();
                        Hide();
                    };
                    row.AddChild(connect);
                }
                var stop = Palette.GhostButton("Stop", Palette.Type.Label);
                stop.TooltipText = "Graceful stop via PUT /shutdown — the server deregisters and exits.";
                stop.Pressed += async () =>
                {
                    _status.Text = $"Stopping {server.Url}…";
                    bool ok = await _controller.RequestShutdownAsync(server.Url);
                    _status.Text = ok ? $"Stopped {server.Url}." : $"Could not stop {server.Url} (already gone?).";
                    if (isCurrent) _api.CheckHealth(); // reflect the disconnect in the rail status
                    Refresh();
                };
                row.AddChild(stop);
            }
            _list.AddChild(row);
        }
    }

    private static string NormalizeUrl(string url) => (url ?? "").Trim().TrimEnd('/').ToLowerInvariant();
}
