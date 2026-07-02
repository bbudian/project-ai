using System;
using System.Collections.Generic;
using System.Globalization;
using Godot;

// A lightweight sortable table: header buttons sort (numeric-aware, toggling direction), rows are activatable
// (click → RowActivated with the ORIGINAL row index, stable across sorts). Built for the Benchmark Compare view —
// deliberately not a general grid; columns are text with an optional tone.
public partial class DataTable : VBoxContainer
{
    /// <summary>Raised with the original (pre-sort) index of the clicked row.</summary>
    public event Action<int> RowActivated;

    public readonly record struct Cell(string Text, Palette.Tone? Tone = null, bool Bold = false);

    private string[] _columns = [];
    private Cell[][] _rows = [];
    private int[] _order = [];
    private int _sortColumn = -1;
    private bool _sortDescending;
    private HBoxContainer _header;
    private VBoxContainer _body;

    public override void _Ready()
    {
        AddThemeConstantOverride("separation", 2);
        _header = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _header.AddThemeConstantOverride("separation", Palette.Space.Sm);
        AddChild(_header);
        _body = new VBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
        _body.AddThemeConstantOverride("separation", 2);
        AddChild(_body);
    }

    public void SetData(string[] columns, Cell[][] rows)
    {
        _columns = columns;
        _rows = rows;
        _order = new int[rows.Length];
        for (int i = 0; i < rows.Length; i++) _order[i] = i;
        if (_sortColumn >= columns.Length) _sortColumn = -1;
        ApplySort();
        Rebuild();
    }

    private void SortBy(int column)
    {
        if (_sortColumn == column) _sortDescending = !_sortDescending;
        else { _sortColumn = column; _sortDescending = false; }
        ApplySort();
        Rebuild();
    }

    private void ApplySort()
    {
        if (_sortColumn < 0) return;
        Array.Sort(_order, (a, b) =>
        {
            int cmp = CompareCells(_rows[a][_sortColumn].Text, _rows[b][_sortColumn].Text);
            return _sortDescending ? -cmp : cmp;
        });
    }

    // Numeric-aware: "45.85" sorts as a number (ignoring a trailing unit like " tok/s" or "%"), else ordinal text.
    private static int CompareCells(string a, string b)
    {
        static bool Num(string s, out double v)
        {
            int end = 0;
            while (end < s.Length && (char.IsAsciiDigit(s[end]) || s[end] is '.' or '-' or ',')) end++;
            return double.TryParse(s[..end].Replace(",", ""), NumberStyles.Float, CultureInfo.InvariantCulture, out v);
        }
        if (Num(a, out double na) && Num(b, out double nb)) return na.CompareTo(nb);
        return string.CompareOrdinal(a, b);
    }

    private void Rebuild()
    {
        foreach (var child in _header.GetChildren()) child.QueueFree();
        foreach (var child in _body.GetChildren()) child.QueueFree();

        for (int c = 0; c < _columns.Length; c++)
        {
            int column = c;
            string arrow = _sortColumn == c ? (_sortDescending ? "  ▾" : "  ▴") : "";
            var button = Palette.GhostButton(_columns[c] + arrow, Palette.Type.Caption);
            button.SizeFlagsHorizontal = SizeFlags.ExpandFill;
            button.Pressed += () => SortBy(column);
            _header.AddChild(button);
        }

        foreach (int original in _order)
        {
            var row = _rows[original];
            var panel = new PanelContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
            Palette.StylePanel(panel, Palette.PanelBg, radius: Palette.Radius.Sm, pad: 6);
            var line = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
            line.AddThemeConstantOverride("separation", Palette.Space.Sm);
            foreach (var cell in row)
            {
                var label = Palette.Heading(cell.Text, cell.Bold ? Palette.Type.Body : Palette.Type.Label,
                    cell.Tone is { } tone ? ToneColor(tone) : Palette.Text);
                label.SizeFlagsHorizontal = SizeFlags.ExpandFill;
                label.AutowrapMode = TextServer.AutowrapMode.WordSmart;
                line.AddChild(label);
            }
            panel.AddChild(line);
            int captured = original;
            panel.GuiInput += ev =>
            {
                if (ev is InputEventMouseButton { Pressed: true, ButtonIndex: MouseButton.Left })
                    RowActivated?.Invoke(captured);
            };
            _body.AddChild(panel);
        }
    }

    private static Color ToneColor(Palette.Tone tone) => tone switch
    {
        Palette.Tone.Accent => Palette.Accent,
        Palette.Tone.Good => Palette.Good,
        Palette.Tone.Bad => Palette.Bad,
        _ => Palette.Muted,
    };
}
