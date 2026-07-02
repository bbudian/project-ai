using System;
using System.Collections.Generic;
using Godot;

// The content area to the right of the nav rail: holds every registered view, shows exactly one. Views are
// created once and kept alive (hidden, not freed) so their state — transcript, form fields, an in-flight train
// job — survives navigation. Emits Shown so the rail can highlight the active destination.
public partial class ViewHost : VBoxContainer
{
    public event Action<string> Shown;

    private readonly Dictionary<string, IView> _views = new();
    private string _active;

    public ViewHost()
    {
        SizeFlagsHorizontal = SizeFlags.ExpandFill;
        SizeFlagsVertical = SizeFlags.ExpandFill;
        AddThemeConstantOverride("separation", 0);
    }

    public void Register(string id, IView view)
    {
        _views[id] = view;
        view.Root.Visible = false;
        AddChild(view.Root);
    }

    public void Show(string id)
    {
        if (id == _active || !_views.ContainsKey(id)) return;
        if (_active != null)
        {
            var previous = _views[_active];
            previous.Root.Visible = false;
            previous.OnHidden();
        }
        var next = _views[id];
        next.Root.Visible = true;
        next.OnShown();
        _active = id;
        Shown?.Invoke(id);
    }
}
