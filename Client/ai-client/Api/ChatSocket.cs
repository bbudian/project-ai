using System;
using System.Collections.Generic;
using System.Text;
using Godot;

// WebSocket transport for streaming chat (the server's /chat endpoint). Godot's WebSocketPeer must be polled every
// frame, so this is a Node. Outgoing frames are queued until the socket opens, then flushed in order; incoming JSON
// frames are parsed and surfaced as events (Ready / Token / Done / ChatError / Closed) so the UI stays
// transport-agnostic. One persistent connection keeps the server's KV cache warm across turns (Phase 1 of live chat).
public partial class ChatSocket : Node
{
    public event Action<SessionInfo> SessionReady; // server accepted the session (model/backend/instruct/context)
    public event Action<string> Token;             // a streamed text delta
    public event Action<TurnStats> Done;           // a turn finished (stop reason + token/timing/context accounting)
    public event Action<SourceLink[]> Sources;     // web-research sources for the turn (arrives before the tokens)
    public event Action<string> ChatError;         // server-reported or transport error
    public event Action Closed;                    // the connection dropped/closed

    private readonly WebSocketPeer _peer = new();
    private readonly Queue<string> _outbox = new();
    private bool _wasOpen;
    private bool _connecting;

    public bool IsOpen => _peer.GetReadyState() == WebSocketPeer.State.Open;
    public bool IsActive => _connecting || IsOpen;

    /// <summary>Opens a connection to the server's /chat endpoint (derived from the HTTP base URL).</summary>
    public void Connect(string baseUrl)
    {
        string ws = ToWsUrl(baseUrl);
        _outbox.Clear();
        _wasOpen = false;
        _connecting = true;
        var err = _peer.ConnectToUrl(ws);
        if (err != Error.Ok) { _connecting = false; ChatError?.Invoke($"Could not open WebSocket to {ws} ({err})."); }
    }

    /// <summary>
    /// Starts (or resets) the session for a model + backend, optionally attaching the server-side memory store.
    /// Memory is session-scoped (the server honors it only on this frame, baking the bridge into the warm cache),
    /// so toggling it mid-conversation requires a session restart. The store id is omitted — the server defaults it
    /// to the model name (docs/CLIENT_DESIGN.md §4.4) — and the user is the fixed single-local-user "default".
    /// </summary>
    public void SendStart(string model, string backend, bool memory)
    {
        var d = new Godot.Collections.Dictionary
        {
            { "type", "start" }, { "model", model ?? "" }, { "backend", backend ?? "" },
        };
        if (memory) { d["memory"] = true; d["user"] = "default"; }
        Enqueue(d);
    }

    /// <summary>Sends a user message; the assistant reply streams back as Token events ending in Done.</summary>
    public void SendMessage(GenerateRequest r)
    {
        var d = new Godot.Collections.Dictionary
        {
            { "type", "message" }, { "text", r.Prompt }, { "maxTokens", r.MaxTokens }, { "sample", r.Sample }, { "research", r.Research },
        };
        if (r.Sample) { d["temperature"] = r.Temperature; d["topK"] = r.TopK; d["topP"] = r.TopP; d["seed"] = r.Seed; }
        Enqueue(d);
    }

    /// <summary>Asks the server to stop the in-flight generation; it ends the turn early with a "done" (stop=canceled).</summary>
    public void Cancel() => Enqueue(new Godot.Collections.Dictionary { { "type", "cancel" } });

    public void Disconnect()
    {
        if (_peer.GetReadyState() is WebSocketPeer.State.Open or WebSocketPeer.State.Connecting) _peer.Close();
        _connecting = false;
        _wasOpen = false;
        _outbox.Clear();
    }

    public override void _Process(double delta)
    {
        _peer.Poll();
        switch (_peer.GetReadyState())
        {
            case WebSocketPeer.State.Open:
                if (!_wasOpen) { _wasOpen = true; _connecting = false; FlushOutbox(); }
                while (_peer.GetAvailablePacketCount() > 0)
                    HandleFrame(Encoding.UTF8.GetString(_peer.GetPacket()));
                break;
            case WebSocketPeer.State.Closed:
                if (_wasOpen || _connecting) { _wasOpen = false; _connecting = false; Closed?.Invoke(); }
                break;
        }
    }

    private void Enqueue(Godot.Collections.Dictionary payload)
    {
        _outbox.Enqueue(Json.Stringify(payload));
        if (IsOpen) FlushOutbox();
    }

    private void FlushOutbox()
    {
        while (_outbox.Count > 0) _peer.SendText(_outbox.Dequeue());
    }

    private void HandleFrame(string raw)
    {
        var json = new Json();
        if (json.Parse(raw) != Error.Ok || json.Data.VariantType != Variant.Type.Dictionary) return;
        var d = json.Data.AsGodotDictionary();
        switch (d.Str("type"))
        {
            case "ready":
                SessionReady?.Invoke(new SessionInfo(
                    d.Str("model"), d.Str("backend"), d.Bool("instruct"), d.Int("contextLimit")));
                break;
            case "token": Token?.Invoke(d.Str("text")); break;
            case "sources": Sources?.Invoke(ParseSources(d)); break;
            case "done":
                // The short form (research canceled mid-search) carries only stop — the numeric fields default to 0.
                Done?.Invoke(new TurnStats(
                    d.Str("stop"), d.Int("promptTokens"), d.Int("generatedTokens"),
                    d.Float("seconds"), d.Int("position"), d.Int("contextLimit")));
                break;
            case "error": ChatError?.Invoke(d.Str("error", "chat error")); break;
        }
    }

    private static SourceLink[] ParseSources(Godot.Collections.Dictionary d)
    {
        if (!d.ContainsKey("items")) return [];
        var arr = d["items"].AsGodotArray();
        var sources = new SourceLink[arr.Count];
        for (int i = 0; i < arr.Count; i++)
        {
            var item = arr[i].AsGodotDictionary();
            sources[i] = new SourceLink(item.Str("title"), item.Str("url"));
        }
        return sources;
    }

    private static string ToWsUrl(string baseUrl)
    {
        string u = (baseUrl ?? "").Trim().TrimEnd('/');
        if (u.StartsWith("https://")) u = "wss://" + u["https://".Length..];
        else if (u.StartsWith("http://")) u = "ws://" + u["http://".Length..];
        else if (!u.StartsWith("ws")) u = "ws://" + u;
        return u + "/chat";
    }
}
