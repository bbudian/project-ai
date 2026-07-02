using Godot;

// The scrollable conversation column. It owns the list of TurnCards and nothing else: Begin starts a new turn
// (returning its card so the caller can stream into and complete it later) and Clear resets to the empty
// placeholder. The view never reaches in to mutate cards directly.
public partial class Transcript : ScrollContainer
{
    private VBoxContainer _column;
    private Control _placeholder;
    private int _fontSize = Palette.DefaultFontSize;

    public override void _Ready()
    {
        HorizontalScrollMode = ScrollMode.Disabled;
        SizeFlagsHorizontal = SizeFlags.ExpandFill;
        SizeFlagsVertical = SizeFlags.ExpandFill;

        // Vertical ExpandFill lets the empty-state center in the viewport; with real content the column just grows.
        _column = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill, SizeFlagsVertical = SizeFlags.ExpandFill };
        _column.AddThemeConstantOverride("separation", 12);
        AddChild(_column);

        ShowPlaceholder();
    }

    /// <summary>Starts a new turn for <paramref name="prompt"/> and returns its card to resolve later.</summary>
    public TurnCard Begin(string prompt)
    {
        if (_placeholder != null) { _placeholder.QueueFree(); _placeholder = null; }

        var card = new TurnCard(prompt, _fontSize) { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _column.AddChild(card);
        Callable.From(ScrollToBottom).CallDeferred(); // after layout settles
        return card;
    }

    /// <summary>Sets the conversation text size for new turns and updates any already on screen.</summary>
    public void SetFontSize(int size)
    {
        _fontSize = size;
        foreach (var child in _column.GetChildren())
            if (child is TurnCard card) card.SetFontSize(size);
    }

    public void Clear()
    {
        foreach (var child in _column.GetChildren()) child.QueueFree();
        _placeholder = null;
        ShowPlaceholder();
    }

    public void ScrollToBottom() => ScrollVertical = (int)GetVScrollBar().MaxValue;

    private void ShowPlaceholder()
    {
        _placeholder = Palette.EmptyState("💬", "Start a conversation",
            "Pick a model below and say something — replies stream in token by token.");
        _column.AddChild(_placeholder);
    }
}
