using System;
using System.Text;
using Godot;

// HttpRequest-backed IApiClient. Serializes requests, and turns transport/HTTP/parse failures into Ok=false
// results so callers never have to branch on Godot error codes. HttpRequest serves one request at a time, so a
// single Pending flag is enough; its completion signal fires on the main thread, making the events UI-safe.
public partial class ApiClient : Node, IApiClient
{
    private enum Pending { None, Health, Generate }

    private static readonly string[] JsonHeaders = { "Content-Type: application/json" };

    private HttpRequest _http;
    private Pending _pending = Pending.None;

    public string BaseUrl { get; set; } = "http://localhost:8080";
    public bool Busy => _pending != Pending.None;

    public event Action<HealthResult> HealthReceived;
    public event Action<GenerateResult> GenerationReceived;

    public override void _Ready()
    {
        _http = new HttpRequest { Name = "Http" };
        AddChild(_http);
        _http.RequestCompleted += OnCompleted;
    }

    public void CheckHealth()
    {
        if (Busy) return;
        _pending = Pending.Health;
        if (_http.Request(Url("/health"), JsonHeaders, Godot.HttpClient.Method.Get) != Error.Ok)
            FailPending("Could not start request.");
    }

    public void Generate(GenerateRequest request)
    {
        if (Busy) return;
        _pending = Pending.Generate;

        var body = new Godot.Collections.Dictionary
        {
            { "prompt", request.Prompt },
            { "maxTokens", request.MaxTokens },
        };
        if (!string.IsNullOrEmpty(request.Model)) body["model"] = request.Model;
        if (request.Sample)
        {
            body["temperature"] = request.Temperature;
            body["topK"] = request.TopK;
            body["topP"] = request.TopP;
            body["seed"] = 0;
        }

        if (_http.Request(Url("/generate"), JsonHeaders, Godot.HttpClient.Method.Post, Json.Stringify(body)) != Error.Ok)
            FailPending("Could not start request.");
    }

    private void OnCompleted(long result, long code, string[] headers, byte[] body)
    {
        var pending = _pending;
        _pending = Pending.None;

        if ((HttpRequest.Result)result != HttpRequest.Result.Success)
        {
            // Surface the specific Godot transport result (CantConnect / CantResolve / Timeout …) so a failure
            // is diagnosable instead of a generic "failed".
            Emit(pending, $"Can't reach {BaseUrl} — {(HttpRequest.Result)result}. Is `projectai serve` running on that port?");
            return;
        }

        Godot.Collections.Dictionary data = null;
        var json = new Json();
        if (json.Parse(Encoding.UTF8.GetString(body)) == Error.Ok && json.Data.VariantType == Variant.Type.Dictionary)
            data = json.Data.AsGodotDictionary();

        if (data == null) { Emit(pending, $"Unreadable response (HTTP {code})."); return; }
        if (code != 200) { Emit(pending, data.ContainsKey("error") ? data["error"].AsString() : $"HTTP {code}"); return; }

        if (pending == Pending.Health)
        {
            var array = data.ContainsKey("models") ? data["models"].AsGodotArray() : [];
            var names = new string[array.Count];
            for (int i = 0; i < names.Length; i++) names[i] = array[i].AsString();
            HealthReceived?.Invoke(new HealthResult(true, names, data.ContainsKey("default") ? data["default"].AsString() : "", null));
        }
        else if (pending == Pending.Generate)
            GenerationReceived?.Invoke(new GenerateResult(true, data.ContainsKey("text") ? data["text"].AsString() : "", null));
    }

    private void FailPending(string error)
    {
        var pending = _pending;
        _pending = Pending.None;
        Emit(pending, error);
    }

    private void Emit(Pending pending, string error)
    {
        if (pending == Pending.Health) HealthReceived?.Invoke(new HealthResult(false, [], "", error));
        else if (pending == Pending.Generate) GenerationReceived?.Invoke(new GenerateResult(false, null, error));
    }

    private string Url(string path) => BaseUrl.Trim().TrimEnd('/') + path;
}
