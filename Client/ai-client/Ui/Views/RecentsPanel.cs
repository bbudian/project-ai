using System;
using Godot;

// Chat's context panel (docs/CLIENT_DESIGN.md: "Recents belong to Chat only"): New chat + the recents list,
// split out of the old app-global Sidebar. Emits intent events; ChatView decides what they mean.
public partial class RecentsPanel : PanelContainer
{
    public event Action NewChatRequested;
    public event Action<string> RecentSelected;

    private const int MaxRecents = 50; // cap so a long session doesn't accumulate buttons without bound

    private VBoxContainer _recents;

    public override void _Ready()
    {
        CustomMinimumSize = new Vector2(220, 0);
        Palette.StylePanel(this, Palette.SidebarBg, pad: Palette.Space.Md);

        var column = new VBoxContainer();
        column.AddThemeConstantOverride("separation", Palette.Space.Md);
        AddChild(column);

        var newChat = Palette.GhostButton("＋   New chat");
        newChat.Pressed += () => NewChatRequested?.Invoke();
        column.AddChild(newChat);

        column.AddChild(Palette.Heading("Recents", Palette.Type.Caption, Palette.Muted));

        var scroll = new ScrollContainer
        {
            SizeFlagsVertical = SizeFlags.ExpandFill,
            HorizontalScrollMode = ScrollContainer.ScrollMode.Disabled,
        };
        _recents = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        scroll.AddChild(_recents);
        column.AddChild(scroll);
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

        var item = Palette.GhostButton(Format.Ellipsize(prompt, 30), Palette.Type.Label);
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
}
