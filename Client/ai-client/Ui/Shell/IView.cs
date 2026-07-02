using Godot;

// A routed destination in the shell. The ViewHost keeps every registered view alive (state survives switching —
// the chat transcript and a running train job don't reset when the user navigates away) and toggles visibility,
// calling OnShown/OnHidden so a view can start/stop its own polling.
public interface IView
{
    Control Root { get; }
    void OnShown();
    void OnHidden();
}

// The registry of destination ids. Chat and Models are live; Bench, Memory, and Upgrade are reserved here (per
// docs/CLIENT_DESIGN.md) so registering them later is one Register(...) call in Main — they are deliberately NOT
// shown as dead "coming soon" rail entries until their views and server backends exist.
public static class ViewIds
{
    public const string Chat = "chat";
    public const string Models = "models";
    public const string Bench = "bench";
    public const string Memory = "memory";
    public const string Upgrade = "upgrade";
}
