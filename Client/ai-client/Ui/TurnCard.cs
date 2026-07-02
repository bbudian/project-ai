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
    private VBoxContainer _sources;
    private Spinner _spinner;
    private bool _streamed;

    public TurnCard(string prompt, int fontSize)
    {
        AddThemeConstantOverride("separation", 8);
        // The user's turn is a right-aligned bubble at ~72% width (the mockup's chat convention: your messages on
        // the right, the model's replies full-width); the response card below spans the column.
        var userRow = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        userRow.AddChild(new Control { SizeFlagsHorizontal = SizeFlags.ExpandFill, SizeFlagsStretchRatio = 28 });
        var bubble = Card(null, prompt, Palette.UserBubble, Palette.Text, out _promptBody);
        bubble.SizeFlagsHorizontal = SizeFlags.ExpandFill;
        bubble.SizeFlagsStretchRatio = 72;
        userRow.AddChild(bubble);
        AddChild(userRow);
        AddChild(ResponseCard(out _response, out _note, out _sources, out _spinner));
        SetFontSize(fontSize);
    }

    /// <summary>Shows the web sources the answer was grounded in (web-research mode) as clickable citations.</summary>
    public void SetSources(SourceLink[] sources)
    {
        if (sources is null || sources.Length == 0) return;
        foreach (var child in _sources.GetChildren()) child.QueueFree();
        _sources.AddChild(Palette.Heading("Sources", 11, Palette.Muted));
        for (int i = 0; i < sources.Length; i++)
        {
            var s = sources[i];
            var link = new LinkButton
            {
                Text = $"[{i + 1}] {(s.Title.Length > 70 ? s.Title[..70] + "…" : s.Title)}",
                TooltipText = s.Url,
                SizeFlagsHorizontal = SizeFlags.ShrinkBegin,
            };
            link.AddThemeColorOverride("font_color", Palette.Accent);
            link.AddThemeColorOverride("font_hover_color", Palette.Accent);
            link.Pressed += () => OS.ShellOpen(s.Url); // open the source in the user's browser
            _sources.AddChild(link);
        }
        _sources.Visible = true;
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
    private static PanelContainer ResponseCard(out Label body, out Label note, out VBoxContainer sources, out Spinner spinner)
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

        // Web-research citations (web mode); populated by SetSources, hidden otherwise.
        sources = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill, Visible = false };
        sources.AddThemeConstantOverride("separation", 2);
        column.AddChild(sources);

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
        Palette.StylePanel(panel, background, radius: Palette.Radius.Md, pad: 12);

        var column = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        column.AddThemeConstantOverride("separation", 4);

        if (speaker is not null) column.AddChild(Palette.Heading(speaker, 11, Palette.Muted)); // user bubbles are captionless

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
