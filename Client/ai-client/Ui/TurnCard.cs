using Godot;

// One conversation turn — the user's prompt above the model's response — rendered as two stacked cards. It is
// self-contained and reusable: the transcript instantiates one per prompt; the host streams the reply in with
// Append, ends it with Complete, or reports a failure with Fail. While waiting, the response card shows a spinner
// next to "Generating…"; the first Append/Complete/Fail clears it.
public partial class TurnCard : VBoxContainer
{
    private Label _promptBody;
    private Label _response;
    private Label _note;
    private Spinner _spinner;
    private bool _streamed;

    public TurnCard(string prompt, int fontSize)
    {
        AddThemeConstantOverride("separation", 6);
        AddChild(Card("You", prompt, Palette.UserBubble, Palette.Text, out _promptBody));
        AddChild(ResponseCard(out _response, out _note, out _spinner));
        SetFontSize(fontSize);
    }

    /// <summary>Resizes the conversation text (both bubbles) live; the captions stay fixed.</summary>
    public void SetFontSize(int size)
    {
        _promptBody.AddThemeFontSizeOverride("font_size", size);
        _response.AddThemeFontSizeOverride("font_size", size);
    }

    public void Fail(string error)
    {
        StopSpinner();
        _response.Text = error;
        _response.AddThemeColorOverride("font_color", Palette.Bad);
    }

    /// <summary>Streams a decoded text delta into the response (clears the spinner/placeholder on the first one).</summary>
    public void Append(string delta)
    {
        if (string.IsNullOrEmpty(delta)) return;
        if (!_streamed)
        {
            StopSpinner();
            _response.Text = "";
            _response.AddThemeColorOverride("font_color", Palette.Text);
            _streamed = true;
        }
        _response.Text += delta;
    }

    /// <summary>
    /// Marks a streamed turn finished, annotating why it ended (<paramref name="stop"/>). A natural finish
    /// (eos / im_end / a hit length cap) shows nothing extra; a cancel or a context-limit stop adds a small muted
    /// footer; and a turn that produced no text explains the blank instead of a bare "(empty response)".
    /// </summary>
    public void Complete(string stop)
    {
        StopSpinner();
        if (!_streamed)
        {
            _response.Text = stop switch
            {
                "context_full" => "Context window full — start a New chat to continue.",
                "canceled" => "Stopped before any output.",
                _ => "(empty response)",
            };
            _response.AddThemeColorOverride("font_color", Palette.Muted);
            return;
        }

        string note = stop switch
        {
            "canceled" => "⏹ Stopped",
            "context" => "Reached the context limit — start a New chat to continue.",
            _ => "", // eos / im_end / maxTokens → a natural end, no footer
        };
        if (note.Length > 0) { _note.Text = note; _note.Visible = true; }
    }

    private void StopSpinner()
    {
        if (_spinner is not null) { _spinner.QueueFree(); _spinner = null; }
    }

    // The model's response card: caption over a [spinner + body] row, so the spinner sits inline with "Generating…".
    private static PanelContainer ResponseCard(out Label body, out Label note, out Spinner spinner)
    {
        var panel = new PanelContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        Palette.StylePanel(panel, Palette.PanelBg, radius: 10, pad: 12);

        var column = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        column.AddThemeConstantOverride("separation", 4);
        column.AddChild(Palette.Heading("ProjectAI", 11, Palette.Muted));

        var row = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        row.AddThemeConstantOverride("separation", 8);
        spinner = new Spinner();
        row.AddChild(spinner);
        body = new Label
        {
            Text = "Generating…",
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        body.AddThemeColorOverride("font_color", Palette.Muted);
        row.AddChild(body);
        column.AddChild(row);

        // A small muted footer for the stop reason (stopped / context limit); hidden on a natural finish.
        note = Palette.Heading("", 11, Palette.Muted);
        note.Visible = false;
        column.AddChild(note);

        panel.AddChild(column);
        return panel;
    }

    private static PanelContainer Card(string speaker, string text, Color background, Color textColor, out Label body)
    {
        var panel = new PanelContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        Palette.StylePanel(panel, background, radius: 10, pad: 12);

        var column = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        column.AddThemeConstantOverride("separation", 4);

        var caption = Palette.Heading(speaker, 11, Palette.Muted);
        column.AddChild(caption);

        body = new Label
        {
            Text = text,
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
            SizeFlagsHorizontal = SizeFlags.ExpandFill,
        };
        body.AddThemeColorOverride("font_color", textColor);
        column.AddChild(body);

        panel.AddChild(column);
        return panel;
    }
}
