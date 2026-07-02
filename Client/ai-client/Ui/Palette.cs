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

    /// <summary>Type scale — every font size in the app comes from here so text hierarchy can't drift per view.</summary>
    public static class Type
    {
        public const int Caption = 12;
        public const int Label = 13;
        public const int Body = 14;
        public const int H3 = 16;
        public const int H2 = 18;
        public const int H1 = 22;
        public const int Mono = 13;
    }

    /// <summary>Spacing scale — margins/separations pick from these steps instead of per-component magic numbers.</summary>
    public static class Space
    {
        public const int Xs = 4;
        public const int Sm = 8;
        public const int Md = 12;
        public const int Lg = 16;
        public const int Xl = 24;
        public const int Xxl = 32;
    }

    /// <summary>Corner-radius scale.</summary>
    public static class Radius
    {
        public const int Sm = 8;
        public const int Md = 12;
        public const int Lg = 14;
        public const int Pill = 999;
    }

    /// <summary>Default conversation text size — shared by the transcript and the composer's "Text size" control so they can't drift.</summary>
    public const int DefaultFontSize = Type.Body;

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
    public static Button GhostButton(string text, int fontSize = Type.Body)
    {
        var button = new Button { Text = text, Flat = true, Alignment = HorizontalAlignment.Left };
        button.AddThemeStyleboxOverride("normal", Box(Transparent, radius: Radius.Sm, pad: 8));
        button.AddThemeStyleboxOverride("hover", Box(Hover, radius: Radius.Sm, pad: 8));
        button.AddThemeStyleboxOverride("pressed", Box(Hover, radius: Radius.Sm, pad: 8));
        button.AddThemeStyleboxOverride("disabled", Box(Transparent, radius: Radius.Sm, pad: 8));
        button.AddThemeStyleboxOverride("focus", Box(Transparent));
        button.AddThemeColorOverride("font_color", Text);
        button.AddThemeColorOverride("font_hover_color", Text);
        button.AddThemeColorOverride("font_disabled_color", new Color(Muted, 0.7f));
        button.AddThemeFontSizeOverride("font_size", fontSize);
        return button;
    }

    /// <summary>A pill toggle chip (the mockup's Memory/Web toggles): outlined when off, accent-tinted when on.</summary>
    public static Button Chip(string text)
    {
        var chip = new Button { Text = text, ToggleMode = true, Flat = true };
        chip.AddThemeStyleboxOverride("normal", Box(Transparent, radius: Radius.Pill, pad: 6, border: 1, borderColor: Border));
        chip.AddThemeStyleboxOverride("hover", Box(Hover, radius: Radius.Pill, pad: 6, border: 1, borderColor: Border));
        chip.AddThemeStyleboxOverride("pressed", Box(new Color(Accent, 0.18f), radius: Radius.Pill, pad: 6, border: 1, borderColor: new Color(Accent, 0.5f)));
        chip.AddThemeStyleboxOverride("focus", Box(Transparent));
        chip.AddThemeColorOverride("font_color", Muted);
        chip.AddThemeColorOverride("font_hover_color", Text);
        chip.AddThemeColorOverride("font_pressed_color", Accent);
        chip.AddThemeFontSizeOverride("font_size", Type.Label);
        return chip;
    }

    /// <summary>A solid accent button — the primary action (Send).</summary>
    public static Button PrimaryButton(string text)
    {
        var button = new Button { Text = text };
        button.AddThemeStyleboxOverride("normal", Box(Accent, radius: Radius.Sm, pad: 10));
        button.AddThemeStyleboxOverride("hover", Box(Accent.Lightened(0.08f), radius: Radius.Sm, pad: 10));
        button.AddThemeStyleboxOverride("pressed", Box(Accent.Darkened(0.08f), radius: Radius.Sm, pad: 10));
        button.AddThemeStyleboxOverride("disabled", Box(Accent.Darkened(0.35f), radius: Radius.Sm, pad: 10));
        button.AddThemeColorOverride("font_color", OnAccent);
        button.AddThemeColorOverride("font_hover_color", OnAccent);
        button.AddThemeColorOverride("font_disabled_color", new Color(OnAccent, 0.6f));
        return button;
    }

    /// <summary>Wraps a control in a MarginContainer so spacing lives at the composition site, not inside components.</summary>
    public static MarginContainer Pad(Control child, int left, int right, int top, int bottom, bool expand = false)
    {
        var margin = new MarginContainer();
        if (expand) margin.SizeFlagsVertical = Control.SizeFlags.ExpandFill;
        margin.AddThemeConstantOverride("margin_left", left);
        margin.AddThemeConstantOverride("margin_right", right);
        margin.AddThemeConstantOverride("margin_top", top);
        margin.AddThemeConstantOverride("margin_bottom", bottom);
        margin.AddChild(child);
        return margin;
    }

    /// <summary>A rounded panel card with optional caption — the building block for view content sections.</summary>
    public static PanelContainer Card(Control content, string title = null, int pad = Space.Lg)
    {
        var panel = new PanelContainer { SizeFlagsHorizontal = Control.SizeFlags.ExpandFill };
        StylePanel(panel, PanelBg, radius: Radius.Md, pad: pad);
        if (title is null) { panel.AddChild(content); return panel; }
        var column = new VBoxContainer { SizeFlagsHorizontal = Control.SizeFlags.ExpandFill };
        column.AddThemeConstantOverride("separation", Space.Sm);
        column.AddChild(Heading(title, Type.Caption, Muted));
        column.AddChild(content);
        panel.AddChild(column);
        return panel;
    }

    /// <summary>Semantic tint for badges and deltas, derived from the existing palette colors.</summary>
    public enum Tone { Neutral, Accent, Good, Bad }

    private static Color ToneColor(Tone tone) => tone switch
    {
        Tone.Accent => Accent,
        Tone.Good => Good,
        Tone.Bad => Bad,
        _ => Muted,
    };

    /// <summary>A small pill badge (e.g. "default", "running", "instruct").</summary>
    public static Control Badge(string text, Tone tone = Tone.Neutral)
    {
        var color = ToneColor(tone);
        var panel = new PanelContainer { SizeFlagsHorizontal = Control.SizeFlags.ShrinkBegin };
        StylePanel(panel, new Color(color, 0.16f), radius: Radius.Pill, pad: 0, border: 1, borderColor: new Color(color, 0.45f));
        var label = new Label { Text = text };
        label.AddThemeFontSizeOverride("font_size", Type.Caption - 1);
        label.AddThemeColorOverride("font_color", color);
        panel.AddChild(Pad(label, 8, 8, 1, 1));
        return panel;
    }

    /// <summary>A badge whose tone follows a job/availability state string (idle / running / done / error / available).</summary>
    public static Control StatusBadge(string state) => Badge(state, state switch
    {
        "running" => Tone.Accent,
        "done" or "available" => Tone.Good,
        "error" => Tone.Bad,
        _ => Tone.Neutral,
    });

    /// <summary>A labelled control row: fixed-width caption on the left, the control filling the rest.</summary>
    public static HBoxContainer Field(string label, Control control, int labelWidth = 120)
    {
        var row = new HBoxContainer { SizeFlagsHorizontal = Control.SizeFlags.ExpandFill };
        row.AddThemeConstantOverride("separation", Space.Sm);
        var caption = Heading(label, Type.Label, Muted);
        caption.CustomMinimumSize = new Vector2(labelWidth, 0);
        row.AddChild(caption);
        control.SizeFlagsHorizontal = Control.SizeFlags.ExpandFill;
        row.AddChild(control);
        return row;
    }

    /// <summary>A centered empty-state block: glyph, title, hint, and an optional call-to-action button.</summary>
    public static Control EmptyState(string glyph, string title, string hint, Button cta = null)
    {
        var column = new VBoxContainer
        {
            SizeFlagsHorizontal = Control.SizeFlags.ExpandFill,
            SizeFlagsVertical = Control.SizeFlags.ExpandFill,
            Alignment = BoxContainer.AlignmentMode.Center,
        };
        column.AddThemeConstantOverride("separation", Space.Sm);
        foreach (var (text, size, color) in new[] { (glyph, Type.H1 + 6, Muted), (title, Type.H3, Text), (hint, Type.Label, Muted) })
        {
            if (string.IsNullOrEmpty(text)) continue;
            var label = Heading(text, size, color);
            label.HorizontalAlignment = HorizontalAlignment.Center;
            label.SizeFlagsHorizontal = Control.SizeFlags.ExpandFill;
            column.AddChild(label);
        }
        if (cta != null)
        {
            cta.SizeFlagsHorizontal = Control.SizeFlags.ShrinkCenter;
            column.AddChild(cta);
        }
        return column;
    }
}
