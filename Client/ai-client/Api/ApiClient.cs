using System;
using System.Collections.Generic;
using System.Text;
using Godot;

// HttpRequest-backed IApiClient. Serializes requests, and turns transport/HTTP/parse failures into Ok=false results
// so callers never have to branch on Godot error codes. Completion signals fire on the main thread, making the
// events UI-safe.
//
// Concurrency: a small pool of HttpRequest nodes (each serves one request at a time) with two policies —
// user actions queue for the next free slot (never dropped), status polls drop when a slot isn't free or the same
// poll is already in flight (the timer retries; a stale poll result is worthless anyway). This replaces the old
// single-slot client that silently dropped ANY call while busy — with several views sharing one client, an
// invisible drop of a user action is no longer acceptable.
//
// Each endpoint is defined in ONE place: a Send(...) call that supplies how to turn a success body into its result
// event and how to shape a failure for the same event. There is no central per-endpoint switch, so adding an
// endpoint as the API grows means adding one method — nothing else changes (open/closed).
public partial class ApiClient : Node, IApiClient
{
    private static readonly string[] JsonHeaders = { "Content-Type: application/json" };
    private const int PoolSize = 3;

    private sealed record Pending(string Path, Godot.HttpClient.Method Method, string Body,
        Action<Godot.Collections.Dictionary> OnSuccess, Action<string> OnFailure, bool Poll)
    {
        public string Url; // captured at dispatch, so a failure names the URL actually contacted (BaseUrl may have changed since)
    }

    private readonly List<HttpRequest> _idle = new();
    private readonly Dictionary<HttpRequest, Pending> _active = new();
    private readonly Queue<Pending> _waiting = new();       // user actions parked until a slot frees
    private readonly HashSet<string> _pollsInFlight = new(); // dedupes polls by path
    private int _healthSeq; // latest-wins guard: only the newest /health request may publish (they can race on the pool)

    public string BaseUrl { get; set; } = "http://localhost:8080";

    public event Action<HealthResult> HealthReceived;
    public event Action<TrainStartResult> TrainStarted;
    public event Action<TrainStatus> TrainStatusReceived;

    public override void _Ready()
    {
        for (int i = 0; i < PoolSize; i++)
        {
            // A finite timeout so a hung request can't hold a pool slot for the OS connect timeout (~20s); matches
            // the server's own 30s entity-body window.
            var http = new HttpRequest { Name = $"Http{i}", Timeout = 30 };
            AddChild(http);
            var slot = http; // capture the node, not the loop variable's final value
            http.RequestCompleted += (result, code, _, body) => OnCompleted(slot, result, code, body);
            _idle.Add(http);
        }
        // Anything sent before this node entered the tree was parked in _waiting; drain it now that slots exist.
        while (_waiting.Count > 0 && _idle.Count > 0) Dispatch(_waiting.Dequeue());
    }

    // Two /health requests can race on the pool (e.g. Check clicked twice around a URL edit, or the automatic
    // refresh after training). Only the NEWEST one may publish — a stale straggler must not overwrite fresh
    // connection state or repopulate the pickers from a server the app no longer points at.
    public void CheckHealth()
    {
        int seq = ++_healthSeq;
        Send(new Pending(
            "/health", Godot.HttpClient.Method.Get, null,
            data => { if (seq == _healthSeq) HealthReceived?.Invoke(ParseHealth(data)); },
            error => { if (seq == _healthSeq) HealthReceived?.Invoke(new HealthResult(false, [], "", [], "", [], error)); },
            Poll: false));
    }

    public void StartTraining(TrainRequest request)
    {
        var body = new Godot.Collections.Dictionary
        {
            { "name", request.Name },
            { "text", request.Text },
            { "size", request.Size },
            { "steps", request.Steps },
        };
        if (!string.IsNullOrEmpty(request.Backend)) body["backend"] = request.Backend;

        Send(new Pending("/train", Godot.HttpClient.Method.Post, Json.Stringify(body),
            _ => TrainStarted?.Invoke(new TrainStartResult(true, null)),
            error => TrainStarted?.Invoke(new TrainStartResult(false, error)),
            Poll: false));
    }

    // A poll: dropped when its slot isn't free — the 1.5s timer simply asks again.
    public void CheckTrainStatus() => Send(new Pending(
        "/train/status", Godot.HttpClient.Method.Get, null,
        data => TrainStatusReceived?.Invoke(ParseTrainStatus(data)),
        error => TrainStatusReceived?.Invoke(new TrainStatus("error", "", 0, 0, 0f, error)),
        Poll: true));

    private void Send(Pending pending)
    {
        if (pending.Poll)
        {
            if (_pollsInFlight.Contains(pending.Path) || _idle.Count == 0) return;
            _pollsInFlight.Add(pending.Path);
            Dispatch(pending);
            return;
        }
        if (_idle.Count == 0) { _waiting.Enqueue(pending); return; }
        Dispatch(pending);
    }

    private void Dispatch(Pending pending)
    {
        var http = _idle[^1];
        _idle.RemoveAt(_idle.Count - 1);
        _active[http] = pending;
        pending.Url = Url(pending.Path);

        Error started = pending.Body == null
            ? http.Request(pending.Url, JsonHeaders, pending.Method)
            : http.Request(pending.Url, JsonHeaders, pending.Method, pending.Body);
        if (started != Error.Ok) Finish(http, null, "Could not start request.");
    }

    private void OnCompleted(HttpRequest http, long result, long code, byte[] body)
    {
        if ((HttpRequest.Result)result != HttpRequest.Result.Success)
        {
            // Surface the specific Godot transport result (CantConnect / CantResolve / Timeout …) so a failure is
            // diagnosable instead of a generic "failed". Names the URL this request actually contacted.
            Finish(http, null, $"Can't reach {_active[http].Url} — {(HttpRequest.Result)result}. Is `projectai serve` running on that port?");
            return;
        }

        Godot.Collections.Dictionary data = null;
        var json = new Json();
        if (json.Parse(Encoding.UTF8.GetString(body)) == Error.Ok && json.Data.VariantType == Variant.Type.Dictionary)
            data = json.Data.AsGodotDictionary();

        if (data == null) { Finish(http, null, $"Unreadable response (HTTP {code})."); return; }
        // 202 (train accepted) is success too; anything outside 2xx is an error (with the server's message if present).
        if (code is < 200 or >= 300) { Finish(http, null, data.ContainsKey("error") ? data.Str("error") : $"HTTP {code}"); return; }
        Finish(http, data, null);
    }

    // Returns the slot to the pool, delivers this request's result, THEN starts the next queued action — so events
    // stay FIFO (a queued request whose start fails synchronously must not report before the request that freed it).
    private void Finish(HttpRequest http, Godot.Collections.Dictionary data, string error)
    {
        var pending = _active[http];
        _active.Remove(http);
        _idle.Add(http);
        if (pending.Poll) _pollsInFlight.Remove(pending.Path);

        if (error != null) pending.OnFailure?.Invoke(error);
        else pending.OnSuccess?.Invoke(data);

        if (_waiting.Count > 0 && _idle.Count > 0) Dispatch(_waiting.Dequeue());
    }

    private static HealthResult ParseHealth(Godot.Collections.Dictionary data)
    {
        var modelsArr = data.Arr("models");
        var names = new string[modelsArr.Count];
        for (int i = 0; i < names.Length; i++) names[i] = modelsArr[i].AsString();

        var backendsArr = data.Arr("backends");
        var backends = new BackendOption[backendsArr.Count];
        for (int i = 0; i < backends.Length; i++)
        {
            var b = backendsArr[i].AsGodotDictionary();
            backends[i] = new BackendOption(b.Str("id"), b.Str("label"), b.Bool("available"), b.Str("reason"));
        }

        var sizesArr = data.Arr("sizes");
        var sizes = new SizeOption[sizesArr.Count];
        for (int i = 0; i < sizes.Length; i++)
        {
            var s = sizesArr[i].AsGodotDictionary();
            sizes[i] = new SizeOption(s.Str("id"), s.Str("label"));
        }

        return new HealthResult(true, names, data.Str("default"), backends, data.Str("defaultBackend"), sizes, null);
    }

    private static TrainStatus ParseTrainStatus(Godot.Collections.Dictionary data)
    {
        if (!data.ContainsKey("training")) return new TrainStatus("idle", "", 0, 0, 0f, null);
        var t = data["training"].AsGodotDictionary();
        return new TrainStatus(
            t.Str("state", "idle"), t.Str("name"), t.Int("step"), t.Int("totalSteps"), t.Float("loss"), t.Str("error", null));
    }

    private string Url(string path) => BaseUrl.Trim().TrimEnd('/') + path;
}
