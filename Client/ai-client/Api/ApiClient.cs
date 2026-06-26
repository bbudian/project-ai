using System;
using System.Text;
using Godot;

// HttpRequest-backed IApiClient. Serializes requests, and turns transport/HTTP/parse failures into Ok=false results
// so callers never have to branch on Godot error codes. HttpRequest serves one request at a time, so a single
// in-flight continuation is enough; its completion signal fires on the main thread, making the events UI-safe.
//
// Each endpoint is defined in ONE place: a Send(...) call that supplies how to turn a success body into its result
// event and how to shape a failure for the same event. There is no central per-endpoint switch, so adding an
// endpoint as the API grows means adding one method — nothing else changes (open/closed).
public partial class ApiClient : Node, IApiClient
{
    private static readonly string[] JsonHeaders = { "Content-Type: application/json" };

    private HttpRequest _http;
    private Action<Godot.Collections.Dictionary> _onSuccess;
    private Action<string> _onFailure;

    public string BaseUrl { get; set; } = "http://localhost:8080";
    public bool Busy { get; private set; }

    public event Action<HealthResult> HealthReceived;
    public event Action<TrainStartResult> TrainStarted;
    public event Action<TrainStatus> TrainStatusReceived;

    public override void _Ready()
    {
        _http = new HttpRequest { Name = "Http" };
        AddChild(_http);
        _http.RequestCompleted += OnCompleted;
    }

    public void CheckHealth() => Send(
        "/health", Godot.HttpClient.Method.Get, null,
        data => HealthReceived?.Invoke(ParseHealth(data)),
        error => HealthReceived?.Invoke(new HealthResult(false, [], "", [], "", [], error)));

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

        Send("/train", Godot.HttpClient.Method.Post, Json.Stringify(body),
            _ => TrainStarted?.Invoke(new TrainStartResult(true, null)),
            error => TrainStarted?.Invoke(new TrainStartResult(false, error)));
    }

    public void CheckTrainStatus() => Send(
        "/train/status", Godot.HttpClient.Method.Get, null,
        data => TrainStatusReceived?.Invoke(ParseTrainStatus(data)),
        error => TrainStatusReceived?.Invoke(new TrainStatus("error", "", 0, 0, 0f, error)));

    // Starts one request, remembering how to finish it. Ignored if a request is already in flight (HttpRequest is
    // single-shot). body == null sends no payload (GET); otherwise it's the JSON string for a POST.
    private void Send(string path, Godot.HttpClient.Method method, string body,
                      Action<Godot.Collections.Dictionary> onSuccess, Action<string> onFailure)
    {
        if (Busy) return;
        _onSuccess = onSuccess;
        _onFailure = onFailure;
        Busy = true;

        Error started = body == null
            ? _http.Request(Url(path), JsonHeaders, method)
            : _http.Request(Url(path), JsonHeaders, method, body);
        if (started != Error.Ok) Fail("Could not start request.");
    }

    private void OnCompleted(long result, long code, string[] headers, byte[] body)
    {
        if ((HttpRequest.Result)result != HttpRequest.Result.Success)
        {
            // Surface the specific Godot transport result (CantConnect / CantResolve / Timeout …) so a failure is
            // diagnosable instead of a generic "failed".
            Fail($"Can't reach {BaseUrl} — {(HttpRequest.Result)result}. Is `projectai serve` running on that port?");
            return;
        }

        Godot.Collections.Dictionary data = null;
        var json = new Json();
        if (json.Parse(Encoding.UTF8.GetString(body)) == Error.Ok && json.Data.VariantType == Variant.Type.Dictionary)
            data = json.Data.AsGodotDictionary();

        if (data == null) { Fail($"Unreadable response (HTTP {code})."); return; }
        // 202 (train accepted) is success too; anything outside 2xx is an error (with the server's message if present).
        if (code is < 200 or >= 300) { Fail(data.ContainsKey("error") ? data.Str("error") : $"HTTP {code}"); return; }
        Succeed(data);
    }

    // Go idle, then invoke the stored success/failure continuation. Capturing it BEFORE clearing lets a handler that
    // immediately fires another request (which sets new continuations) do so safely.
    private void Succeed(Godot.Collections.Dictionary data)
    {
        var cont = _onSuccess;
        GoIdle();
        cont?.Invoke(data);
    }

    private void Fail(string error)
    {
        var cont = _onFailure;
        GoIdle();
        cont?.Invoke(error);
    }

    private void GoIdle()
    {
        _onSuccess = null;
        _onFailure = null;
        Busy = false;
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
