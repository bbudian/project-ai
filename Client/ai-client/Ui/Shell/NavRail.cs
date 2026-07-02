using System;
using System.Collections.Generic;
using Godot;

// The left navigation rail (labeled, per docs/CLIENT_DESIGN.md §1.1): brand, one button per routed destination,
// a spacer, the Settings gear (a modal in a later milestone — present but disabled until then), and an app-global
// footer slot for the connection panel. It only emits Navigated(id); the ViewHost decides what showing means.
public partial class NavRail : PanelContainer
{
    public event Action<string> Navigated;
    public event Action SettingsRequested;

    private readonly Dictionary<string, RailButton> _buttons = new();
    private VBoxContainer _column;
    private VBoxContainer _destinations;

    public override void _Ready()
    {
        CustomMinimumSize = new Vector2(236, 0);
        var box = Palette.Box(Palette.SidebarBg, pad: Palette.Space.Md);
        box.BorderWidthRight = 1;
        box.BorderColor = Palette.Border;
        AddThemeStyleboxOverride("panel", box);

        _column = new VBoxContainer();
        _column.AddThemeConstantOverride("separation", Palette.Space.Md);
        AddChild(_column);

        _column.AddChild(Palette.Heading("ProjectAI", Palette.Type.H2));

        _destinations = new VBoxContainer();
        _destinations.AddThemeConstantOverride("separation", Palette.Space.Xs);
        _column.AddChild(_destinations);

        _column.AddChild(new Control { SizeFlagsVertical = SizeFlags.ExpandFill }); // push gear + footer to the bottom

        var gear = Palette.GhostButton("⚙   Settings");
        gear.Pressed += () => SettingsRequested?.Invoke();
        _column.AddChild(gear);
    }

    public void AddDestination(string id, string label)
    {
        var button = new RailButton(label);
        button.Pressed += () => Navigated?.Invoke(id);
        _destinations.AddChild(button);
        _buttons[id] = button;
    }

    /// <summary>Highlights the active destination (driven by ViewHost.Shown, so rail state can't drift from reality).</summary>
    public void SetActive(string id)
    {
        foreach (var (buttonId, button) in _buttons) button.SetActive(buttonId == id);
    }

    /// <summary>Pins an app-global control (the connection panel) at the very bottom of the rail.</summary>
    public void AddFooter(Control footer) => _column.AddChild(footer);
}

// A rail destination button with a real active state (highlighted background), replacing the old
// "disable the active mode's button" convention that greyed out the current destination.
public partial class RailButton : Button
{
    public RailButton(string label)
    {
        Text = label;
        Flat = true;
        Alignment = HorizontalAlignment.Left;
        AddThemeStyleboxOverride("focus", Palette.Box(Palette.Transparent));
        AddThemeStyleboxOverride("hover", Palette.Box(Palette.Hover, radius: Palette.Radius.Sm, pad: 8));
        AddThemeStyleboxOverride("pressed", Palette.Box(Palette.Hover, radius: Palette.Radius.Sm, pad: 8));
        AddThemeColorOverride("font_hover_color", Palette.Text);
        SetActive(false);
    }

    public void SetActive(bool active)
    {
        // The mockup's active nav treatment: accent-tinted background + accent text, not just a hover grey.
        AddThemeStyleboxOverride("normal",
            Palette.Box(active ? new Color(Palette.Accent, 0.14f) : Palette.Transparent, radius: Palette.Radius.Sm, pad: 8));
        AddThemeColorOverride("font_color", active ? Palette.Accent : Palette.Muted);
        AddThemeColorOverride("font_hover_color", active ? Palette.Accent : Palette.Text);
    }
}
