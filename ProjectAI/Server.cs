using System.Net;
using System.Text.Json;
using ProjectAI.Core;
using ProjectAI.Models;

// A minimal local HTTP API for the UI client. Serves generation from a directory of checkpoint models (the UI's
// model picker), loading each on first use and caching it. Built on HttpListener — no web framework dependency.
//   GET  /health   → { status, models:[...], default }
//   GET  /models   → { models:[...], default }
//   POST /generate → { prompt, model?, maxTokens?, temperature?, topK?, topP?, seed? } → { text, prompt, model }
// One request is served at a time (tiny CPU models; the client disables its button while generating).
internal static class Server
{
    private sealed record GenerateRequest(
        string? Prompt, string? Model, int? MaxTokens, float? Temperature, int? TopK, float? TopP, ulong? Seed);

    private static readonly JsonSerializerOptions JsonOpts = new() { PropertyNameCaseInsensitive = true };

    public static void Run(IComputeBackend be, string modelsDirectory, string defaultModel, int port)
    {
        var registry = new ModelRegistry(be, modelsDirectory);
        var models = registry.List();
        if (models.Count == 0)
        {
            Console.Error.WriteLine($"error: no .ckpt models found in '{modelsDirectory}' (run `train` first, or pass --models <dir>).");
            Environment.Exit(2);
        }
        if (!models.Contains(defaultModel)) defaultModel = models[0];

        try
        {
            registry.Get(defaultModel); // fail fast if the default checkpoint is corrupt
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"error: could not load default model '{defaultModel}': {ex.Message}");
            Environment.Exit(2);
        }

        using var listener = new HttpListener();
        string prefix = $"http://localhost:{port}/";
        listener.Prefixes.Add(prefix);
        try
        {
            listener.Start();
        }
        catch (HttpListenerException ex)
        {
            Console.Error.WriteLine($"error: could not listen on {prefix}: {ex.Message}");
            Console.Error.WriteLine("(another process may hold the port; try --port, or run an elevated shell once.)");
            Environment.Exit(2);
        }
        if (OperatingSystem.IsWindows())
            try { listener.TimeoutManager.EntityBody = TimeSpan.FromSeconds(30); } catch { /* unsupported config */ }

        Console.WriteLine($"Models in '{modelsDirectory}': {string.Join(", ", models)}   (default: {defaultModel})");
        Console.WriteLine($"Serving on {prefix}  —  POST /generate, GET /health, GET /models  (Ctrl+C to stop)");
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; listener.Stop(); };

        while (listener.IsListening)
        {
            HttpListenerContext ctx;
            try { ctx = listener.GetContext(); }
            catch { break; } // listener stopped
            Handle(ctx, be, registry, defaultModel);
        }
        Console.WriteLine("Server stopped.");
    }

    private static void Handle(HttpListenerContext ctx, IComputeBackend be, ModelRegistry registry, string defaultModel)
    {
        var req = ctx.Request;
        var res = ctx.Response;
        Console.WriteLine($"  ← {req.HttpMethod,-4} {req.Url?.AbsolutePath}  from {req.RemoteEndPoint}");
        // Permissive CORS so a browser/web export can call the API too; native Godot ignores these.
        res.AddHeader("Access-Control-Allow-Origin", "*");
        res.AddHeader("Access-Control-Allow-Headers", "Content-Type");
        res.AddHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");

        try
        {
            string path = req.Url?.AbsolutePath ?? "/";
            if (req.HttpMethod == "OPTIONS") { res.StatusCode = 204; res.Close(); return; }

            if (req.HttpMethod == "GET" && (path == "/health" || path == "/models"))
            {
                WriteJson(res, 200, new { status = "ok", models = registry.List(), @default = defaultModel });
                return;
            }

            if (req.HttpMethod == "POST" && path == "/generate")
            {
                HandleGenerate(req, res, be, registry, defaultModel);
                return;
            }

            WriteJson(res, 404, new { error = "not found; use GET /health, GET /models, or POST /generate" });
        }
        catch (Exception ex)
        {
            try { WriteJson(res, 500, new { error = ex.Message }); } catch { /* client gone */ }
        }
    }

    private static void HandleGenerate(
        HttpListenerRequest req, HttpListenerResponse res, IComputeBackend be, ModelRegistry registry, string defaultModel)
    {
        // Cap the request body so an oversized or slow POST can't exhaust memory / wedge the single-threaded loop.
        const int maxBody = 1 << 20; // 1 MiB — far more than any prompt needs
        if (req.ContentLength64 > maxBody) { WriteJson(res, 413, new { error = "request body too large" }); return; }
        string body;
        using (var reader = new StreamReader(req.InputStream, req.ContentEncoding))
        {
            var buffer = new char[maxBody + 1];
            int read = reader.ReadBlock(buffer, 0, buffer.Length);
            if (read > maxBody) { WriteJson(res, 413, new { error = "request body too large" }); return; }
            body = new string(buffer, 0, read);
        }

        GenerateRequest? gr;
        try { gr = JsonSerializer.Deserialize<GenerateRequest>(body, JsonOpts); }
        catch (JsonException) { WriteJson(res, 400, new { error = "invalid JSON body" }); return; }

        if (gr is null || string.IsNullOrEmpty(gr.Prompt)) { WriteJson(res, 400, new { error = "missing 'prompt'" }); return; }
        if (gr.Temperature is { } temp && (temp < 0f || !float.IsFinite(temp))) { WriteJson(res, 400, new { error = "temperature must be a finite number >= 0" }); return; }
        if (gr.TopK is < 0) { WriteJson(res, 400, new { error = "topK must be >= 0" }); return; }
        if (gr.TopP is { } topp && (!float.IsFinite(topp) || topp <= 0f || topp > 1f)) { WriteJson(res, 400, new { error = "topP must be a finite number in (0, 1]" }); return; }

        string modelName = string.IsNullOrEmpty(gr.Model) ? defaultModel : gr.Model;
        LoadedModel? loaded;
        try { loaded = registry.Get(modelName); }
        catch (Exception ex) { WriteJson(res, 500, new { error = $"failed to load model '{modelName}': {ex.Message}" }); return; }
        if (loaded is null) { WriteJson(res, 400, new { error = $"unknown model '{modelName}'" }); return; }

        int maxTokens = gr.MaxTokens is > 0 and <= 512 ? gr.MaxTokens.Value : 64;
        bool sample = gr.Temperature.HasValue || gr.TopK.HasValue || gr.TopP.HasValue;
        ISampler sampler = sample
            ? new TopKTopPSampler(new PcgRng(gr.Seed ?? 0), gr.Temperature ?? 1f, gr.TopK ?? 0, gr.TopP ?? 1f)
            : new GreedySampler();

        string text = Inference.GenerateText(be, loaded.Model, loaded.Tokenizer, loaded.Config, gr.Prompt, sampler, maxTokens);
        WriteJson(res, 200, new { text, prompt = gr.Prompt, model = loaded.Name });
    }

    private static void WriteJson(HttpListenerResponse res, int status, object payload)
    {
        res.StatusCode = status;
        res.ContentType = "application/json";
        byte[] buf = JsonSerializer.SerializeToUtf8Bytes(payload);
        res.ContentLength64 = buf.Length;
        res.OutputStream.Write(buf, 0, buf.Length);
        res.Close();
    }
}
