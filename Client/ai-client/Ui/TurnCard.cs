using Godot;

// One conversation turn — the user's prompt above the model's response — rendered as two stacked cards. It is
// self-contained and reusable: the transcript instantiates one per prompt and calls Resolve/Fail when the
// response lands. The two cards are built by a single helper so their styling stays in lockstep (DRY).
public partial class TurnCard : VBoxContainer
{
    private Label _response;

    public TurnCard(string prompt)
    {
        AddThemeConstantOverride("separation", 6);
        AddChild(Card("You", prompt, Palette.UserBubble, Palette.Text, out _));
        AddChild(Card("ProjectAI", "Generating…", Palette.PanelBg, Palette.Muted, out _response));
    }

    public void Resolve(string text)
    {
        _response.Text = string.IsNullOrEmpty(text) ? "(empty response)" : text;
        _response.AddThemeColorOverride("font_color", Palette.Text);
    }

    public void Fail(string error)
    {
        _response.Text = error;
        _response.AddThemeColorOverride("font_color", Palette.Bad);
    }

    private static PanelContainer Card(string speaker, string text, Color background, Color textColor, out Label body)
    {
        var panel = new PanelContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        Palette.StylePanel(panel, background, radius: 10, pad: 12);

        var column = new VBoxContainer();
        column.AddThemeConstantOverride("separation", 4);

        var caption = Palette.Heading(speaker, 11, Palette.Muted);
        column.AddChild(caption);

        body = new Label { Text = text, AutowrapMode = TextServer.AutowrapMode.WordSmart };
        body.AddThemeColorOverride("font_color", textColor);
        column.AddChild(body);

        panel.AddChild(column);
        return panel;
    }
}
