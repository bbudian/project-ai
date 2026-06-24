using Godot;

// The single source of truth for the client's look: colors and StyleBox/control factories. Restyling the whole
// app — or retheming it — means editing only this file (DRY). Components never hardcode a color or a corner radius.
public static class Palette
{
    public static readonly Color AppBg      = new("#1e1e1e");
    public static readonly Color SidebarBg  = new("#161616");
    public static readonly Color PanelBg    = new("#252525");
    public static readonly Color InputBg    = new("#2c2c2c");
    public static readonly Color UserBubble = new("#343434");
    public static readonly Color Hover      = new("#2a2a2a");
    public static readonly Color Text       = new("#ededed");
    public static readonly Color Muted      = new("#9b9b9b");
    public static readonly Color Accent     = new("#c96442");
    public static readonly Color OnAccent   = new("#ffffff");
    public static readonly Color Good       = new("#82b682");
    public static readonly Color Bad        = new("#e0726e");
    public static readonly Color Border     = new("#323232");
    public static readonly Color Transparent = new(0, 0, 0, 0);

    /// <summary>A flat rounded box, optionally padded and bordered — the building block for panels and buttons.</summary>
    public static StyleBoxFlat Box(Color bg, int radius = 0, int pad = 0, int border = 0, Color? borderColor = null)
    {
        var box = new StyleBoxFlat { BgColor = bg };
        box.SetCornerRadiusAll(radius);
        if (pad > 0) box.SetContentMarginAll(pad);
        if (border > 0) { box.SetBorderWidthAll(border); box.BorderColor = borderColor ?? Border; }
        return box;
    }

    public static void StylePanel(Control panel, Color bg, int radius = 0, int pad = 0, int border = 0, Color? borderColor = null)
        => panel.AddThemeStyleboxOverride("panel", Box(bg, radius, pad, border, borderColor));

    public static Label Heading(string text, int size, Color? color = null)
    {
        var label = new Label { Text = text };
        label.AddThemeFontSizeOverride("font_size", size);
        label.AddThemeColorOverride("font_color", color ?? Text);
        return label;
    }

    /// <summary>A flat, borderless button that highlights on hover — sidebar nav items.</summary>
    public static Button GhostButton(string text, int fontSize = 14)
    {
        var button = new Button { Text = text, Flat = true, Alignment = HorizontalAlignment.Left };
        button.AddThemeStyleboxOverride("normal", Box(Transparent, radius: 8, pad: 8));
        button.AddThemeStyleboxOverride("hover", Box(Hover, radius: 8, pad: 8));
        button.AddThemeStyleboxOverride("pressed", Box(Hover, radius: 8, pad: 8));
        button.AddThemeStyleboxOverride("focus", Box(Transparent));
        button.AddThemeColorOverride("font_color", Text);
        button.AddThemeColorOverride("font_hover_color", Text);
        button.AddThemeFontSizeOverride("font_size", fontSize);
        return button;
    }

    /// <summary>A solid accent button — the primary action (Send).</summary>
    public static Button PrimaryButton(string text)
    {
        var button = new Button { Text = text };
        button.AddThemeStyleboxOverride("normal", Box(Accent, radius: 8, pad: 10));
        button.AddThemeStyleboxOverride("hover", Box(Accent.Lightened(0.08f), radius: 8, pad: 10));
        button.AddThemeStyleboxOverride("pressed", Box(Accent.Darkened(0.08f), radius: 8, pad: 10));
        button.AddThemeStyleboxOverride("disabled", Box(Accent.Darkened(0.35f), radius: 8, pad: 10));
        button.AddThemeColorOverride("font_color", OnAccent);
        button.AddThemeColorOverride("font_hover_color", OnAccent);
        button.AddThemeColorOverride("font_disabled_color", new Color(OnAccent, 0.6f));
        return button;
    }
}
