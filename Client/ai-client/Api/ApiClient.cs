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
    public event Action<TokenizeResult> TokenizeReceived;
    public event Action<MemoryListResult> MemoryListReceived;
    public event Action<MemoryRenderResult> MemoryRenderReceived;
    public event Action<MemorySaveResult> MemorySaved;
    public event Action<BenchStartResult> BenchStarted;
    public event Action<BenchStatusInfo> BenchStatusReceived;
    public event Action<BenchSuiteInfo[]> BenchSuitesReceived;
    public event Action<BenchRunSummary[]> BenchRunsReceived;
    public event Action<BenchRunDetail> BenchRunReceived;
    public event Action<ConfigInfo> ConfigReceived;
    public event Action<SecretStatus> SecretUpdated;

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
            error => { if (seq == _healthSeq) HealthReceived?.Invoke(new HealthResult(false, [], [], "", [], "", [], error)); },
            Poll: false));
    }

    public void Tokenize(string model, string text)
    {
        var body = new Godot.Collections.Dictionary { { "text", text } };
        if (!string.IsNullOrEmpty(model)) body["model"] = model;
        Send(new Pending("/tokenize", Godot.HttpClient.Method.Post, Json.Stringify(body),
            data => TokenizeReceived?.Invoke(ParseTokenize(data)),
            error => TokenizeReceived?.Invoke(new TokenizeResult(false, model, 0, [], error)),
            Poll: false));
    }

    // Memory endpoints. The user is always the fixed single-local-user "default" (no user picker by design —
    // the server's per-user partition is namespacing, not auth; see docs/CLIENT_DESIGN.md security notes).
    private static string MemoryQuery(string store, string query) =>
        $"?user=default&store={Uri.EscapeDataString(store ?? "default")}" +
        (string.IsNullOrEmpty(query) ? "" : $"&q={Uri.EscapeDataString(query)}");

    public void MemoryList(string store, string query) => Send(new Pending(
        "/memory" + MemoryQuery(store, query), Godot.HttpClient.Method.Get, null,
        data => MemoryListReceived?.Invoke(ParseMemoryList(data)),
        error => MemoryListReceived?.Invoke(new MemoryListResult(false, store, 0, [], error)),
        Poll: false));

    public void MemoryRender(string store, string query) => Send(new Pending(
        "/memory/render" + MemoryQuery(store, query), Godot.HttpClient.Method.Get, null,
        data => MemoryRenderReceived?.Invoke(new MemoryRenderResult(true, data.Str("bridge"), data.Str("recall"), null)),
        error => MemoryRenderReceived?.Invoke(new MemoryRenderResult(false, "", "", error)),
        Poll: false));

    public void MemoryPut(string store, string title, string[] keys, string body, string tier, string trust)
    {
        var keysArr = new Godot.Collections.Array();
        foreach (var k in keys) keysArr.Add(k);
        var payload = new Godot.Collections.Dictionary
        {
            { "title", title }, { "keys", keysArr }, { "body", body },
            { "tier", tier }, { "trust", trust }, { "user", "default" }, { "store", store ?? "default" },
        };
        Send(new Pending("/memory", Godot.HttpClient.Method.Put, Json.Stringify(payload),
            data => MemorySaved?.Invoke(new MemorySaveResult(true, data.Str("id"), null)),
            error => MemorySaved?.Invoke(new MemorySaveResult(false, "", error)),
            Poll: false));
    }

    // ---- benchmark endpoints ---------------------------------------------------------------------------------

    public void StartBenchmark(BenchStartRequest request)
    {
        var models = new Godot.Collections.Array();
        foreach (var m in request.Models) models.Add(m);
        var body = new Godot.Collections.Dictionary
        {
            { "suite", request.Suite }, { "models", models }, { "repeats", request.Repeats },
        };
        if (!string.IsNullOrEmpty(request.Backend)) body["backend"] = request.Backend;
        Send(new Pending("/benchmark", Godot.HttpClient.Method.Post, Json.Stringify(body),
            data => BenchStarted?.Invoke(new BenchStartResult(true, data.Str("runId"), data.Int("total"), null)),
            error => BenchStarted?.Invoke(new BenchStartResult(false, "", 0, error)),
            Poll: false));
    }

    // A poll (like /train/status): dropped when a slot isn't free — the timer just asks again.
    public void CheckBenchStatus() => Send(new Pending(
        "/benchmark/status", Godot.HttpClient.Method.Get, null,
        data => BenchStatusReceived?.Invoke(ParseBenchStatus(data)),
        error => BenchStatusReceived?.Invoke(new BenchStatusInfo("error", "", "", 0, 0, "", "", error)),
        Poll: true));

    public void CancelBenchmark() => Send(new Pending(
        "/benchmark/cancel", Godot.HttpClient.Method.Post, "{}",
        _ => { }, _ => { }, Poll: false));

    public void FetchBenchSuites() => Send(new Pending(
        "/benchmark/suites", Godot.HttpClient.Method.Get, null,
        data =>
        {
            var arr = data.Arr("suites");
            var suites = new BenchSuiteInfo[arr.Count];
            for (int i = 0; i < suites.Length; i++)
            {
                var s = arr[i].AsGodotDictionary();
                suites[i] = new BenchSuiteInfo(s.Str("id"), s.Str("label"), s.Int("caseCount"), s.Bool("hasCorpus"));
            }
            BenchSuitesReceived?.Invoke(suites);
        },
        _ => BenchSuitesReceived?.Invoke([]),
        Poll: false));

    public void FetchBenchRuns() => Send(new Pending(
        "/benchmark/runs", Godot.HttpClient.Method.Get, null,
        data =>
        {
            var arr = data.Arr("runs");
            var runs = new BenchRunSummary[arr.Count];
            for (int i = 0; i < runs.Length; i++)
            {
                var r = arr[i].AsGodotDictionary();
                var modelsArr = r.Arr("models");
                var models = new string[modelsArr.Count];
                for (int m = 0; m < models.Length; m++) models[m] = modelsArr[m].AsString();
                runs[i] = new BenchRunSummary(r.Str("id"), r.Str("suiteId"), models, r.Str("backend"),
                    r.Str("startedUtc"), r.Str("state"), r.Int("cases"));
            }
            BenchRunsReceived?.Invoke(runs);
        },
        _ => BenchRunsReceived?.Invoke([]),
        Poll: false));

    public void FetchBenchRun(string id) => Send(new Pending(
        "/benchmark/run/" + Uri.EscapeDataString(id), Godot.HttpClient.Method.Get, null,
        data => BenchRunReceived?.Invoke(ParseBenchRun(data)),
        error => BenchRunReceived?.Invoke(new BenchRunDetail(false, id, "", "", [], [], error)),
        Poll: false));

    private static BenchStatusInfo ParseBenchStatus(Godot.Collections.Dictionary data)
    {
        var b = data.ContainsKey("bench") ? data["bench"].AsGodotDictionary() : data;
        return new BenchStatusInfo(
            b.Str("state", "idle"), b.Str("runId"), b.Str("suite"), b.Int("done"), b.Int("total"),
            b.Str("currentModel"), b.Str("currentCase"), b.Str("error", null));
    }

    // Hand-written walker over the camelCased BenchRun graph — the client never references server types (net8/net10).
    private static BenchRunDetail ParseBenchRun(Godot.Collections.Dictionary data)
    {
        var aggArr = data.Arr("aggregates");
        var aggregates = new BenchAggregateInfo[aggArr.Count];
        for (int i = 0; i < aggregates.Length; i++)
        {
            var a = aggArr[i].AsGodotDictionary();
            // meanBpb is null when the suite had no corpus — surface as NaN so the UI prints "—".
            double bpb = a.ContainsKey("meanBpb") && a["meanBpb"].VariantType is Variant.Type.Float or Variant.Type.Int
                ? a["meanBpb"].AsDouble() : double.NaN;
            aggregates[i] = new BenchAggregateInfo(
                a.Str("model"), a.Int("n"), bpb, a.Float("medianTokPerSec"), a.Float("checkPassRate"));
        }

        var cellArr = data.Arr("cells");
        var cells = new List<BenchCellInfo>(cellArr.Count);
        for (int i = 0; i < cellArr.Count; i++)
        {
            var c = cellArr[i].AsGodotDictionary();
            if (c.Str("caseId") == "__bpb__") continue; // bookkeeping cell, not a case
            cells.Add(new BenchCellInfo(
                c.Str("model"), c.Str("caseId"), c.Str("output"), c.Int("generatedTokens"), c.Str("stop"),
                c.Float("medianTokPerSec"), c.Float("checkPassRate"), c.Str("error", null)));
        }

        return new BenchRunDetail(true, data.Str("id"), data.Str("suiteId"), data.Str("state"), aggregates, [.. cells], null);
    }

    // ---- config + secrets ------------------------------------------------------------------------------------

    public void FetchConfig() => Send(new Pending(
        "/config", Godot.HttpClient.Method.Get, null,
        data => ConfigReceived?.Invoke(ParseConfig(data)),
        error => ConfigReceived?.Invoke(new ConfigInfo(false, 0, 0, 0, 0, [], error)),
        Poll: false));

    public void SaveMemoryBudgets(int bridgeCards, int bridgeBudget, int recallHits, int recallBudget)
    {
        var body = new Godot.Collections.Dictionary
        {
            { "memory", new Godot.Collections.Dictionary
            {
                { "bridgeCards", bridgeCards }, { "bridgeBudget", bridgeBudget },
                { "recallHits", recallHits }, { "recallBudget", recallBudget },
            } },
        };
        Send(new Pending("/config", Godot.HttpClient.Method.Put, Json.Stringify(body),
            data => ConfigReceived?.Invoke(ParseConfig(data)),
            error => ConfigReceived?.Invoke(new ConfigInfo(false, 0, 0, 0, 0, [], error)),
            Poll: false));
    }

    public void SaveSecret(string key, string value) => Send(new Pending(
        "/config/secrets/" + Uri.EscapeDataString(key), Godot.HttpClient.Method.Put,
        Json.Stringify(new Godot.Collections.Dictionary { { "value", value } }),
        data => SecretUpdated?.Invoke(ParseSecret(data)),
        error => SecretUpdated?.Invoke(new SecretStatus(false, key, false, "", "", error)),
        Poll: false));

    public void ClearSecret(string key) => Send(new Pending(
        "/config/secrets/" + Uri.EscapeDataString(key), Godot.HttpClient.Method.Delete, null,
        data => SecretUpdated?.Invoke(ParseSecret(data)),
        error => SecretUpdated?.Invoke(new SecretStatus(false, key, false, "", "", error)),
        Poll: false));

    private static SecretStatus ParseSecret(Godot.Collections.Dictionary d) =>
        new(true, d.Str("key"), d.Bool("set"), d.Str("hint"), d.Str("source"), null);

    private static ConfigInfo ParseConfig(Godot.Collections.Dictionary data)
    {
        var m = data.ContainsKey("memory") ? data["memory"].AsGodotDictionary() : new Godot.Collections.Dictionary();
        var secretsArr = data.Arr("secrets");
        var secrets = new SecretStatus[secretsArr.Count];
        for (int i = 0; i < secrets.Length; i++)
            secrets[i] = ParseSecret(secretsArr[i].AsGodotDictionary());
        return new ConfigInfo(true,
            m.Int("bridgeCards"), m.Int("bridgeBudget"), m.Int("recallHits"), m.Int("recallBudget"), secrets, null);
    }

    private static MemoryListResult ParseMemoryList(Godot.Collections.Dictionary data)
    {
        var arr = data.Arr("memories");
        var cards = new MemoryCardInfo[arr.Count];
        for (int i = 0; i < cards.Length; i++)
        {
            var m = arr[i].AsGodotDictionary();
            var keysArr = m.Arr("keys");
            var keys = new string[keysArr.Count];
            for (int k = 0; k < keys.Length; k++) keys[k] = keysArr[k].AsString();
            cards[i] = new MemoryCardInfo(m.Str("id"), m.Str("title"), keys, m.Str("tier"), m.Str("trust"), m.Str("asof"));
        }
        return new MemoryListResult(true, data.Str("store"), data.Int("count"), cards, null);
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

        // Enriched catalog when the server provides it; otherwise synthesize name-only entries (older server).
        ModelInfo[] infos;
        if (data.ContainsKey("modelInfos"))
        {
            var infosArr = data.Arr("modelInfos");
            infos = new ModelInfo[infosArr.Count];
            for (int i = 0; i < infos.Length; i++)
            {
                var m = infosArr[i].AsGodotDictionary();
                infos[i] = new ModelInfo(
                    m.Str("name"), m.Long("params"), m.Int("layers"), m.Int("ctx"), m.Int("vocab"),
                    m.Str("tokenizer"), m.Str("dtype"), m.Int("step"), m.Bool("instruct"),
                    m.Long("fileBytes"), m.Str("error", null));
            }
        }
        else
        {
            infos = new ModelInfo[names.Length];
            for (int i = 0; i < names.Length; i++)
                infos[i] = new ModelInfo(names[i], 0, 0, 0, 0, "", "", 0, false, 0, null);
        }

        return new HealthResult(true, names, infos, data.Str("default"), backends, data.Str("defaultBackend"), sizes, null);
    }

    private static TokenizeResult ParseTokenize(Godot.Collections.Dictionary data)
    {
        var tokensArr = data.Arr("tokens");
        var pieces = new string[tokensArr.Count];
        for (int i = 0; i < pieces.Length; i++)
            pieces[i] = tokensArr[i].AsGodotDictionary().Str("text");
        return new TokenizeResult(true, data.Str("model"), data.Int("count"), pieces, null);
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
