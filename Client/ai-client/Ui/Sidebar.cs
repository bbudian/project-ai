using System;
using Godot;

// Left navigation rail (Claude-desktop style): app title, New-chat, a Recents list of past prompts, and a
// bottom connection panel (server URL + Check + status) — the account-area analog. It emits intent events and
// never talks to the API itself, so swapping the transport or adding a setting touches only this file + Main.
public partial class Sidebar : PanelContainer
{
    public event Action NewChatRequested;
    public event Action<string> RecentSelected;
    public event Action CheckRequested;
    public event Action<string> ServerUrlChanged;

    private const int MaxRecents = 50; // cap so a long session doesn't accumulate buttons without bound

    private VBoxContainer _recents;
    private LineEdit _url;
    private Label _status;

    public string ServerUrl => _url.Text;

    public override void _Ready()
    {
        CustomMinimumSize = new Vector2(264, 0);
        Palette.StylePanel(this, Palette.SidebarBg, pad: 12);

        var column = new VBoxContainer();
        column.AddThemeConstantOverride("separation", 10);
        AddChild(column);

        column.AddChild(Palette.Heading("ProjectAI", 18));

        var newChat = Palette.GhostButton("＋   New chat");
        newChat.Pressed += () => NewChatRequested?.Invoke();
        column.AddChild(newChat);

        column.AddChild(Palette.Heading("Recents", 12, Palette.Muted));

        var scroll = new ScrollContainer
        {
            SizeFlagsVertical = SizeFlags.ExpandFill,
            HorizontalScrollMode = ScrollContainer.ScrollMode.Disabled,
        };
        _recents = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        scroll.AddChild(_recents);
        column.AddChild(scroll);

        column.AddChild(BuildConnectionPanel());
    }

    public void AddRecent(string prompt)
    {
        // Drop an existing identical entry so re-running a prompt moves it to the top instead of duplicating.
        foreach (var child in _recents.GetChildren())
            if (child is Button existing && existing.HasMeta("prompt") && existing.GetMeta("prompt").AsString() == prompt)
            {
                _recents.RemoveChild(existing); // immediate, so the count below is accurate (QueueFree is deferred)
                existing.QueueFree();
                break;
            }

        var item = Palette.GhostButton(Format.Ellipsize(prompt, 30), 13);
        item.SetMeta("prompt", prompt); // keep the full prompt for dedup + recall (the label is truncated)
        item.Pressed += () => RecentSelected?.Invoke(prompt);
        _recents.AddChild(item);
        _recents.MoveChild(item, 0); // newest on top

        while (_recents.GetChildCount() > MaxRecents)
        {
            var oldest = _recents.GetChild(_recents.GetChildCount() - 1);
            _recents.RemoveChild(oldest);
            oldest.QueueFree();
        }
    }

    public void SetStatus(string message, bool error)
    {
        _status.Text = message;
        _status.AddThemeColorOverride("font_color", error ? Palette.Bad : Palette.Good);
    }

    private PanelContainer BuildConnectionPanel()
    {
        var panel = new PanelContainer();
        Palette.StylePanel(panel, Palette.PanelBg, radius: 10, pad: 10);

        var column = new VBoxContainer();
        column.AddThemeConstantOverride("separation", 6);

        column.AddChild(Palette.Heading("Server", 12, Palette.Muted));

        _url = new LineEdit { Text = "http://localhost:8080", SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _url.TextChanged += text => ServerUrlChanged?.Invoke(text);
        column.AddChild(_url);

        var check = new Button { Text = "Check connection" };
        check.Pressed += () => CheckRequested?.Invoke();
        column.AddChild(check);

        _status = new Label { Text = "Not connected", AutowrapMode = TextServer.AutowrapMode.WordSmart };
        _status.AddThemeColorOverride("font_color", Palette.Muted);
        _status.AddThemeFontSizeOverride("font_size", 12);
        column.AddChild(_status);

        panel.AddChild(column);
        return panel;
    }
}
