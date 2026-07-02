using Godot;

// A progress bar over a status line — the job-progress block used by the train panel (and, later, benchmark runs).
// Extracted so every long-running job in the app reports progress the same way: the host maps its job states to
// SetPercent + SetStatus and never styles the pair itself.
public partial class ProgressRow : VBoxContainer
{
    private readonly ProgressBar _bar;
    private readonly Label _status;

    public ProgressRow()
    {
        AddThemeConstantOverride("separation", Palette.Space.Sm);
        _bar = new ProgressBar
        {
            MinValue = 0,
            MaxValue = 100,
            Value = 0,
            ShowPercentage = false,
            CustomMinimumSize = new Vector2(0, 16),
        };
        AddChild(_bar);
        _status = Palette.Heading("", Palette.Type.Label, Palette.Muted);
        AddChild(_status);
    }

    public void SetPercent(double percent) => _bar.Value = percent;

    public void SetStatus(string text, Color color)
    {
        _status.Text = text;
        _status.AddThemeColorOverride("font_color", color);
    }
}
