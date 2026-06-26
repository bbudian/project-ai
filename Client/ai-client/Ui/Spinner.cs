using Godot;

// A small indeterminate loading indicator: a bright arc sweeping over a faint track, drawn in code (no assets)
// so it themes with the Palette. The host (TurnCard) frees it when the operation finishes.
public partial class Spinner : Control
{
    private const float Speed = Mathf.Tau * 1.1f; // ~1.1 turns per second
    private const float Sweep = Mathf.Tau * 0.72f; // arc length (~260°)
    private float _angle;

    public Spinner()
    {
        CustomMinimumSize = new Vector2(16, 16);
        SizeFlagsVertical = SizeFlags.ShrinkCenter;   // keep it 16px tall, centered on the text row
        SizeFlagsHorizontal = SizeFlags.ShrinkCenter;
    }

    public override void _Process(double delta)
    {
        _angle = Mathf.Wrap(_angle + (float)delta * Speed, 0f, Mathf.Tau);
        QueueRedraw();
    }

    public override void _Draw()
    {
        Vector2 center = Size / 2f;
        float radius = Mathf.Min(Size.X, Size.Y) / 2f - 2f;
        if (radius <= 0f) return;
        const float width = 2f;

        DrawArc(center, radius, 0f, Mathf.Tau, 32, new Color(Palette.Muted, 0.22f), width, antialiased: true);
        DrawArc(center, radius, _angle, _angle + Sweep, 24, Palette.Accent, width, antialiased: true);
    }
}
