using System.Net;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using ProjectAI.Core;
using ProjectAI.Memory;
using ProjectAI.Models;
using ProjectAI.Research;
using ProjectAI.Training;

// A minimal local HTTP API for the UI client. Serves generation from a directory of checkpoint models (the UI's
// model picker), and — via the ComputeRegistry — from a choice of compute backends (the UI's CPU/GPU picker).
// Each (backend, model) is loaded on first use and cached. Built on HttpListener — no web framework dependency.
//   GET  /health   → { status, models:[...], default, backends:[{id,label,available,reason}], defaultBackend, sizes:[{id,label}], training }
//   GET  /models   → same shape as /health
//   POST /generate → { prompt, model?, backend?, maxTokens?, temperature?, topK?, topP?, seed? } → { text, prompt, model, backend }
// One request is served at a time (tiny CPU models; the client disables its button while generating).
internal static class Server
{
    private sealed record GenerateRequest(
        string? Prompt, string? Model, string? Backend, int? MaxTokens, float? Temperature, int? TopK, float? TopP, ulong? Seed,
        bool? Research, int? ResearchResults, bool? Memory, string? User, string? Store);

    private static readonly JsonSerializerOptions JsonOpts = new() { PropertyNameCaseInsensitive = true };

    // Web research (RAG): fetch current web results and ground the prompt in them. Tavily behind the ISearchProvider
    // seam (reads TAVILY_API_KEY); swap the provider here to change search backends.
    private static readonly WebResearcher Researcher = new(new TavilySearchProvider());

    // Per-user long-term memory (multi-client): stores are resolved lazily per (user, store) under the memory root.
    private static MemoryStoreRegistry? _memory;

    public static void Run(IComputeBackend defaultBackend, string defaultBackendId, string modelsDirectory, string defaultModel, int port, string memoryRoot)
    {
        using var compute = new ComputeRegistry(modelsDirectory, defaultBackend, defaultBackendId);
        var training = new TrainingService();
        _memory = new MemoryStoreRegistry(memoryRoot);
        var models = compute.ListModels();
        if (models.Count == 0)
        {
            Console.Error.WriteLine($"error: no .ckpt models found in '{modelsDirectory}' (run `train` first, or pass --models <dir>).");
            Environment.Exit(2);
        }
        if (!models.Contains(defaultModel)) defaultModel = models[0];

        try
        {
            compute.Resolve(defaultBackendId).Models.Get(defaultModel); // fail fast if the default checkpoint is corrupt
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

        string available = string.Join(", ", compute.AvailableBackends.Where(b => b.Available).Select(b => b.Id));
        Console.WriteLine($"Models in '{modelsDirectory}': {string.Join(", ", models)}   (default: {defaultModel})");
        Console.WriteLine($"Backends available: {available}   (default: {defaultBackendId})");
        Console.WriteLine($"Serving on {prefix}  —  POST /generate, POST /tokenize, POST /train, GET /health, GET /models, WS /chat  (Ctrl+C to stop)");
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; listener.Stop(); };

        while (listener.IsListening)
        {
            HttpListenerContext ctx;
            try { ctx = listener.GetContext(); }
            catch { break; } // listener stopped
            // Handle each request on a thread-pool thread so the accept loop keeps running: a slow /generate (or a
            // client that disconnects mid-request, leaving the generation running with no one to send it to) must
            // not block /health and the rest of the API. Inference itself is still serialized (InferenceLock) since
            // the model/backend isn't thread-safe; the cheap endpoints (/health, /models, /train/status) don't take
            // that lock, so they always answer — which is what lets a reopened client reconnect immediately.
            ThreadPool.QueueUserWorkItem(_ =>
            {
                try { Handle(ctx, compute, training, defaultModel); }
                catch { /* never let one request take down the server process */ }
            });
        }
        Console.WriteLine("Server stopped.");
    }

    private static void Handle(HttpListenerContext ctx, ComputeRegistry compute, TrainingService training, string defaultModel)
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

            if (req.IsWebSocketRequest && path == "/chat") { RunChat(ctx, compute, defaultModel); return; }

            if (req.HttpMethod == "GET" && (path == "/health" || path == "/models"))
            {
                WriteJson(res, 200, new
                {
                    status = "ok",
                    models = compute.ListModels(),
                    @default = defaultModel,
                    backends = compute.AvailableBackends.Select(b => new { id = b.Id, label = b.Label, available = b.Available, reason = b.Reason }),
                    defaultBackend = compute.DefaultId,
                    sizes = ModelPresets.Names.Select(s => new { id = s, label = ModelPresets.Describe(s) }),
                    training = TrainStatus(training),
                });
                return;
            }

            if (req.HttpMethod == "GET" && path == "/train/status")
            {
                WriteJson(res, 200, new { training = TrainStatus(training) });
                return;
            }

            if (req.HttpMethod == "POST" && path == "/train")
            {
                HandleTrain(req, res, compute, training, defaultModel: compute.DefaultId);
                return;
            }

            if (req.HttpMethod == "POST" && path == "/generate")
            {
                if (training.IsTraining) { WriteJson(res, 409, new { error = "a training job is in progress; try again when it finishes" }); return; }
                HandleGenerate(req, res, compute, defaultModel);
                return;
            }

            if (req.HttpMethod == "POST" && path == "/tokenize")
            {
                HandleTokenize(req, res, compute, defaultModel);
                return;
            }

            WriteJson(res, 404, new { error = "not found; use GET /health, GET /models, GET /train/status, POST /generate, or POST /train" });
        }
        catch (Exception ex)
        {
            try { WriteJson(res, 500, new { error = ex.Message }); } catch { /* client gone */ }
        }
    }

    // Serializes the actual model forward pass: the model/backend (KV cache, libtorch DisposeScopes) is not
    // thread-safe, so only one generation runs at a time even though requests are now handled on multiple threads.
    private static readonly object InferenceLock = new();

    private static void HandleGenerate(
        HttpListenerRequest req, HttpListenerResponse res, ComputeRegistry compute, string defaultModel)
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

        // Resolve the compute backend (the UI's CPU/GPU picker). Unknown id → 400. A known-but-unavailable backend
        // (e.g. GPU with no libtorch bundle) is rejected cheaply from the cached startup probe — so we neither
        // re-initialize the device on every request nor echo the raw libtorch error back to the caller.
        string backendId = string.IsNullOrEmpty(gr.Backend) ? compute.DefaultId : gr.Backend;
        if (!Backends.IsKnown(backendId)) { WriteJson(res, 400, new { error = $"unknown backend '{backendId}'" }); return; }
        if (compute.AvailableBackends.FirstOrDefault(b => b.Id == backendId) is { Available: false })
        {
            WriteJson(res, 400, new { error = $"backend '{backendId}' is not available on this machine (see GET /health)" });
            return;
        }

        // Optional web research (RAG): fetch current results and ground the prompt in them BEFORE taking the
        // inference lock, so the (network) search doesn't block other requests. Sources are returned for citation.
        string prompt = gr.Prompt;
        object? sources = null;
        if (gr.Research == true)
        {
            if (!Researcher.Provider.IsConfigured) { WriteJson(res, 400, new { error = $"web research unavailable: {Researcher.Provider.Unavailable}" }); return; }
            try
            {
                var rsw = System.Diagnostics.Stopwatch.StartNew();
                var rr = Researcher.ResearchAsync(gr.Prompt, gr.ResearchResults ?? 5).GetAwaiter().GetResult();
                rsw.Stop();
                prompt = rr.AugmentedPrompt;
                sources = rr.Sources.Select(s => new { title = s.Title, url = s.Url });
                Console.WriteLine($"  research: \"{Trunc(gr.Prompt)}\" → {rr.Sources.Count} sources in {rsw.Elapsed.TotalSeconds:0.00}s ({Researcher.Provider.Name})");
            }
            catch (Exception ex) { WriteJson(res, 502, new { error = $"web research failed: {ex.Message}" }); return; }
        }

        // Optional long-term memory (Stage-0 preemptive recall): prepend the always-pinned bridge + the top trusted
        // memories matched to this prompt, BEFORE the lock (pure file/index work, like web research). Per-user store.
        if (gr.Memory == true && _memory is not null)
        {
            string modelForStore = string.IsNullOrEmpty(gr.Model) ? defaultModel : gr.Model;
            var store = _memory.Resolve(gr.User, gr.Store ?? modelForStore);
            if (store.IsConfigured)
            {
                string bridge = store.RenderBridge(MemoryPolicy.BridgeCards, MemoryPolicy.BridgeBudget);
                string recall = store.RenderRecall(gr.Prompt, MemoryPolicy.RecallHits, MemoryPolicy.RecallBudget);
                if (bridge.Length + recall.Length > 0)
                {
                    prompt = bridge + recall + prompt;
                    Console.WriteLine($"  memory: store={store.StoreId} bridge={bridge.Length}c recall={recall.Length}c");
                }
            }
        }

        // Everything below touches the model cache and runs the forward pass, so it is serialized: a second
        // /generate (e.g. from a reopened client while an abandoned one is still finishing) waits here instead of
        // corrupting shared state — while /health etc. stay responsive because they never take this lock.
        lock (InferenceLock)
        {
            IComputeBackend backend;
            ModelRegistry registry;
            try { (backend, registry) = compute.Resolve(backendId); }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"  backend '{backendId}' failed to start: {ex.Message}"); // operator log only; not leaked to the client
                WriteJson(res, 400, new { error = $"backend '{backendId}' could not start (see GET /health)" });
                return;
            }

            string modelName = string.IsNullOrEmpty(gr.Model) ? defaultModel : gr.Model;
            LoadedModel? loaded;
            try { loaded = registry.Get(modelName); }
            catch (Exception ex) { WriteJson(res, 500, new { error = $"failed to load model '{modelName}': {ex.Message}" }); return; }
            if (loaded is null) { WriteJson(res, 400, new { error = $"unknown model '{modelName}'" }); return; }

            // 0/absent → generate until EOS or the model's context fills (dynamic length); a positive value is an
            // explicit cap, clamped to the context window (GenerateText stops at the context limit regardless).
            int maxTokens = gr.MaxTokens is > 0 ? Math.Min(gr.MaxTokens.Value, loaded.Config.MaxSequenceLength) : loaded.Config.MaxSequenceLength;
            bool sample = gr.Temperature.HasValue || gr.TopK.HasValue || gr.TopP.HasValue;
            ISampler sampler = sample
                ? new TopKTopPSampler(new PcgRng(gr.Seed ?? 0), gr.Temperature ?? 1f, gr.TopK ?? 0, gr.TopP ?? 1f)
                : new GreedySampler();

            string decoding = sample ? $"sample T={gr.Temperature ?? 1f:0.##} topK={gr.TopK ?? 0} topP={gr.TopP ?? 1f:0.##}" : "greedy";
            Console.WriteLine($"  generate: model={loaded.Name} backend={backendId} {decoding} maxTokens={maxTokens} prompt=\"{Trunc(gr.Prompt)}\"");

            var sw = System.Diagnostics.Stopwatch.StartNew();
            var gen = Inference.GenerateText(backend, loaded.Model, loaded.Tokenizer, loaded.Config, prompt, sampler, maxTokens);
            sw.Stop();
            double secs = sw.Elapsed.TotalSeconds;
            double tps = gen.GeneratedTokens / Math.Max(0.001, secs);
            Console.WriteLine($"  → {gen.PromptTokens} prompt + {gen.GeneratedTokens} gen tok in {secs:0.00}s ({tps:0.0} tok/s), stop={gen.StopReason}");
            if (gen.GeneratedTokens == 0)
                Console.WriteLine($"    note: model produced 0 tokens (immediate {gen.StopReason}) — the reply will be empty.");

            // Return ONLY the continuation: the client already shows the user's prompt, so echoing it back reads as
            // "the model repeated my prompt". FullText (prompt+continuation) is what the CLI prints instead.
            WriteJson(res, 200, new
            {
                text = gen.Continuation,
                prompt = gr.Prompt,
                model = loaded.Name,
                backend = backendId,
                promptTokens = gen.PromptTokens,
                generatedTokens = gen.GeneratedTokens,
                stop = gen.StopReason,
                seconds = Math.Round(secs, 3),
                sources,
            });
        }
    }

    private sealed record TokenizeRequest(string? Text, string? Model);

    // Diagnostic endpoint: shows how a string splits into tokens for a model's tokenizer (no weights loaded, no
    // generation). Returns each token's id + decoded piece, the count, and the round-trip decode.
    private static void HandleTokenize(HttpListenerRequest req, HttpListenerResponse res, ComputeRegistry compute, string defaultModel)
    {
        const int maxBody = 1 << 20;
        if (req.ContentLength64 > maxBody) { WriteJson(res, 413, new { error = "request body too large" }); return; }
        string body;
        using (var reader = new StreamReader(req.InputStream, req.ContentEncoding))
        {
            var buffer = new char[maxBody + 1];
            int read = reader.ReadBlock(buffer, 0, buffer.Length);
            if (read > maxBody) { WriteJson(res, 413, new { error = "request body too large" }); return; }
            body = new string(buffer, 0, read);
        }

        TokenizeRequest? tr;
        try { tr = JsonSerializer.Deserialize<TokenizeRequest>(body, JsonOpts); }
        catch (JsonException) { WriteJson(res, 400, new { error = "invalid JSON body" }); return; }
        if (tr?.Text is null) { WriteJson(res, 400, new { error = "missing 'text'" }); return; }

        string modelName = string.IsNullOrEmpty(tr.Model) ? defaultModel : tr.Model;
        // The tokenizer is backend-agnostic, so resolve via the default backend's registry (no GPU/weights needed).
        var registry = compute.Resolve(compute.DefaultId).Models;
        ProjectAI.Tokenizers.ITokenizer? tok;
        try { tok = registry.GetTokenizer(modelName); }
        catch (Exception ex) { WriteJson(res, 500, new { error = $"failed to load tokenizer for '{modelName}': {ex.Message}" }); return; }
        if (tok is null) { WriteJson(res, 400, new { error = $"unknown model '{modelName}'" }); return; }

        var ids = tok.Encode(tr.Text);
        var tokens = new object[ids.Count];
        for (int i = 0; i < ids.Count; i++) tokens[i] = new { id = ids[i], text = tok.Decode([ids[i]]) };
        WriteJson(res, 200, new { model = modelName, vocab = tok.VocabSize, count = ids.Count, tokens, decoded = tok.Decode(ids) });
    }

    private sealed record TrainRequest(
        string? Name, string? Text, string? Size, int? Steps, int? Batch, int? SeqLen, float? LearningRate, string? Backend);

    private static void HandleTrain(HttpListenerRequest req, HttpListenerResponse res, ComputeRegistry compute, TrainingService training, string defaultModel)
    {
        if (training.IsTraining) { WriteJson(res, 409, new { error = "a training job is already running" }); return; }

        // Training corpora are larger than prompts, so allow a bigger body here than /generate.
        const int maxBody = 64 << 20; // 64 MiB
        if (req.ContentLength64 > maxBody) { WriteJson(res, 413, new { error = "request body too large" }); return; }
        string body;
        using (var reader = new StreamReader(req.InputStream, req.ContentEncoding))
        {
            var buffer = new char[Math.Min(maxBody, (int)Math.Max(1024, req.ContentLength64)) + 1];
            int read = reader.ReadBlock(buffer, 0, buffer.Length);
            if (read > maxBody) { WriteJson(res, 413, new { error = "request body too large" }); return; }
            body = new string(buffer, 0, read);
        }

        TrainRequest? tr;
        try { tr = JsonSerializer.Deserialize<TrainRequest>(body, JsonOpts); }
        catch (JsonException) { WriteJson(res, 400, new { error = "invalid JSON body" }); return; }

        if (tr is null || string.IsNullOrWhiteSpace(tr.Name)) { WriteJson(res, 400, new { error = "missing 'name'" }); return; }
        if (string.IsNullOrEmpty(tr.Text)) { WriteJson(res, 400, new { error = "missing 'text' to train on" }); return; }
        // The name becomes a checkpoint filename in the models directory; reject anything that isn't a plain name.
        if (tr.Name.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0) { WriteJson(res, 400, new { error = "invalid model name" }); return; }

        string size = string.IsNullOrWhiteSpace(tr.Size) ? "small" : tr.Size!;
        var (defaultBatch, defaultSeqLen) = ModelPresets.DefaultTraining(size);
        int steps = tr.Steps is > 0 and <= 100_000 ? tr.Steps.Value : 300;
        int batch = tr.Batch is > 0 ? tr.Batch.Value : defaultBatch;
        int seqLen = tr.SeqLen is > 0 ? tr.SeqLen.Value : defaultSeqLen;
        float lr = tr.LearningRate is { } l && l is > 0 and <= 1 ? l : 3e-4f;
        string backendId = string.IsNullOrEmpty(tr.Backend) ? compute.DefaultId : tr.Backend!;

        if (!Backends.IsKnown(backendId)) { WriteJson(res, 400, new { error = $"unknown backend '{backendId}'" }); return; }
        if (compute.AvailableBackends.FirstOrDefault(b => b.Id == backendId) is { Available: false })
        { WriteJson(res, 400, new { error = $"backend '{backendId}' is not available on this machine (see GET /health)" }); return; }

        var start = new TrainStartRequest(tr.Name!, tr.Text!, size, steps, batch, seqLen, lr, backendId);
        var (ok, message) = training.Start(compute, compute.ModelsDirectory, start);
        if (!ok) { WriteJson(res, 409, new { error = message }); return; }
        WriteJson(res, 202, new { status = "started", name = tr.Name, size, steps, batch, backend = backendId });
    }

    private static object TrainStatus(TrainingService training)
    {
        var job = training.Current;
        if (job is null) return new { state = "idle" };
        return new
        {
            state = job.Status, // running | done | error
            name = job.Name,
            size = job.Size,
            backend = job.Backend,
            step = job.Step,
            totalSteps = job.TotalSteps,
            loss = job.Loss,
            error = job.Error,
        };
    }

    private sealed record ChatIn(
        string? Type, string? Model, string? Backend, string? Text,
        bool? Sample, float? Temperature, int? TopK, float? TopP, int? MaxTokens, ulong? Seed,
        bool? Research, int? ResearchResults, bool? Memory, string? User, string? Store);

    // Accepts a /chat WebSocket and runs a stateful streaming session (Phase 1 of live chat): a persistent warm KV
    // cache, each new user message ingested incrementally, and the assistant reply streamed token-by-token. The
    // protocol is JSON text frames: client sends {type:"start",model,backend} then {type:"message",text,...}, and
    // may send {type:"cancel"} mid-generation; the server replies {type:"ready",...}, a stream of {type:"token",text},
    // then {type:"done",...} (or {type:"error"}).
    //
    // Each turn generates on its OWN task so this loop stays free to read the next frame while a reply streams —
    // that's what lets a "cancel" land mid-generation (the decode loop checks the token each step) and lets a client
    // disconnect stop the work on the host instead of running the GPU to completion with no one listening.
    private static void RunChat(HttpListenerContext ctx, ComputeRegistry compute, string defaultModel)
    {
        WebSocket socket;
        try { socket = ctx.AcceptWebSocketAsync(null).GetAwaiter().GetResult().WebSocket; }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"  chat: handshake failed: {ex.Message}");
            try { ctx.Response.StatusCode = 400; ctx.Response.Close(); } catch { /* client gone */ }
            return;
        }

        Console.WriteLine("  ⇄ chat connected");
        ChatSession? session = null;
        var sendGate = new object();                       // .NET forbids concurrent WebSocket sends — serialize them.
        void SendSafe(object payload) { try { lock (sendGate) Send(socket, payload); } catch { /* peer gone */ } }

        CancellationTokenSource? turnCts = null;
        Task? turnTask = null;
        bool TurnRunning() => turnTask is { IsCompleted: false };

        try
        {
            while (socket.State == WebSocketState.Open)
            {
                string? raw = Receive(socket);
                if (raw is null) break; // closed or errored

                ChatIn? m;
                try { m = JsonSerializer.Deserialize<ChatIn>(raw, JsonOpts); }
                catch (JsonException) { SendSafe(new { type = "error", error = "invalid JSON" }); continue; }
                if (m is null) continue;

                switch (m.Type)
                {
                    case "cancel" or "stop":
                        turnCts?.Cancel(); // the running turn stops at its next token and sends its own "done"
                        break;

                    case "start":
                        if (TurnRunning()) { SendSafe(new { type = "error", error = "a response is still generating" }); break; }
                        try
                        {
                            lock (InferenceLock) { session = OpenSession(compute, defaultModel, m); }
                            SendSafe(new { type = "ready", model = session.ModelName, backend = session.BackendId, instruct = session.Instruct, contextLimit = session.ContextLimit });
                            Console.WriteLine($"  ⇄ chat ready: model={session.ModelName} backend={session.BackendId} instruct={session.Instruct}");
                        }
                        catch (Exception ex) { session = null; SendSafe(new { type = "error", error = ex.Message }); }
                        break;

                    case "message":
                        if (session is null) { SendSafe(new { type = "error", error = "send a 'start' message first" }); break; }
                        if (TurnRunning()) { SendSafe(new { type = "error", error = "a response is still generating" }); break; }
                        turnCts?.Dispose();
                        turnCts = new CancellationTokenSource();
                        ChatSession active = session;       // fresh locals so the task captures this turn's values
                        ChatIn request = m;
                        CancellationToken cancel = turnCts.Token;
                        turnTask = Task.Run(() => RunTurn(active, request, cancel, SendSafe));
                        break;
                }
            }
        }
        catch (Exception ex) { Console.Error.WriteLine($"  chat: {ex.Message}"); }
        finally
        {
            turnCts?.Cancel();                              // a disconnect mid-turn stops the generation on the host
            try { turnTask?.Wait(TimeSpan.FromSeconds(10)); } catch { /* faulted/observed in RunTurn */ }
            turnCts?.Dispose();
            Console.WriteLine("  ⇄ chat disconnected");
            try { if (socket.State == WebSocketState.Open) socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "bye", CancellationToken.None).GetAwaiter().GetResult(); } catch { /* already gone */ }
            socket.Dispose();
        }
    }

    // Runs one assistant turn under InferenceLock (the model/backend isn't shared concurrently), streaming tokens and
    // a final "done" via the serialized <paramref name="send"/>. Cancellation halts the decode loop at its next token
    // (stop="canceled"); the partial reply already streamed stays, and the turn is still closed cleanly in the cache.
    private static void RunTurn(ChatSession session, ChatIn m, CancellationToken cancel, Action<object> send)
    {
        bool sample = m.Sample ?? false;
        ISampler sampler = sample
            ? new TopKTopPSampler(new PcgRng(m.Seed ?? 0), m.Temperature ?? 1f, m.TopK ?? 0, m.TopP ?? 1f)
            : new GreedySampler();
        // 0/absent → generate until EOS / <|im_end|> or the model's context fills (dynamic length); a positive value
        // is an explicit cap, clamped to the context window (Turn stops there anyway).
        int maxTokens = m.MaxTokens is > 0 ? Math.Min(m.MaxTokens.Value, session.ContextLimit) : session.ContextLimit;

        // Optional web research (RAG), done before the inference lock (it's network I/O): fetch current results, ground
        // the user's message in them, and stream the sources to the client for citation.
        string text = m.Text ?? "";
        if (m.Research == true)
        {
            if (!Researcher.Provider.IsConfigured) { send(new { type = "error", error = $"web research unavailable: {Researcher.Provider.Unavailable}" }); return; }
            try
            {
                var rsw = System.Diagnostics.Stopwatch.StartNew();
                var rr = Researcher.ResearchAsync(m.Text ?? "", m.ResearchResults ?? 5, cancel).GetAwaiter().GetResult();
                rsw.Stop();
                text = rr.AugmentedPrompt;
                send(new { type = "sources", items = rr.Sources.Select(s => new { title = s.Title, url = s.Url }) });
                Console.WriteLine($"  ⇄ chat research: {rr.Sources.Count} sources in {rsw.Elapsed.TotalSeconds:0.00}s ({Researcher.Provider.Name})");
            }
            catch (OperationCanceledException) { send(new { type = "done", stop = "canceled", generatedTokens = 0 }); return; }
            catch (Exception ex) { send(new { type = "error", error = $"web research failed: {ex.Message}" }); return; }
        }

        try
        {
            lock (InferenceLock)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var r = session.Turn(text, sampler, maxTokens, cancel, delta => send(new { type = "token", text = delta }));
                sw.Stop();
                send(new
                {
                    type = "done",
                    promptTokens = r.PromptTokens,
                    generatedTokens = r.GeneratedTokens,
                    stop = r.StopReason,
                    seconds = Math.Round(sw.Elapsed.TotalSeconds, 3),
                    position = session.Position,
                    contextLimit = session.ContextLimit,
                });
                Console.WriteLine($"  ⇄ chat turn: {r.PromptTokens}→{r.GeneratedTokens} tok, stop={r.StopReason}, {sw.Elapsed.TotalSeconds:0.00}s, ctx {session.Position}/{session.ContextLimit}");
            }
        }
        catch (Exception ex) { send(new { type = "error", error = ex.Message }); }
    }

    // Resolves the backend + model for a chat session (mirrors /generate's checks). Caller holds InferenceLock.
    private static ChatSession OpenSession(ComputeRegistry compute, string defaultModel, ChatIn m)
    {
        string backendId = string.IsNullOrEmpty(m.Backend) ? compute.DefaultId : m.Backend;
        if (!Backends.IsKnown(backendId)) throw new InvalidOperationException($"unknown backend '{backendId}'");
        if (compute.AvailableBackends.FirstOrDefault(b => b.Id == backendId) is { Available: false })
            throw new InvalidOperationException($"backend '{backendId}' is not available on this machine (see GET /health)");

        var (backend, registry) = compute.Resolve(backendId);
        string modelName = string.IsNullOrEmpty(m.Model) ? defaultModel : m.Model;
        var loaded = registry.Get(modelName) ?? throw new InvalidOperationException($"unknown model '{modelName}'");
        IMemoryStore memory = m.Memory == true && _memory is not null
            ? _memory.Resolve(m.User, m.Store ?? modelName)
            : NullMemoryStore.Instance;
        return new ChatSession(backend, loaded, backendId, memory);
    }

    private static void Send(WebSocket socket, object payload)
    {
        byte[] buf = JsonSerializer.SerializeToUtf8Bytes(payload);
        socket.SendAsync(new ArraySegment<byte>(buf), WebSocketMessageType.Text, endOfMessage: true, CancellationToken.None).GetAwaiter().GetResult();
    }

    // Reads one full text message (joining continuation frames); null on close/error.
    private static string? Receive(WebSocket socket)
    {
        var buffer = new byte[8192];
        using var ms = new MemoryStream();
        while (true)
        {
            WebSocketReceiveResult res;
            try { res = socket.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None).GetAwaiter().GetResult(); }
            catch { return null; }
            if (res.MessageType == WebSocketMessageType.Close) return null;
            ms.Write(buffer, 0, res.Count);
            if (res.EndOfMessage) break;
        }
        return Encoding.UTF8.GetString(ms.ToArray());
    }

    // One-line, single-line-safe rendering of a prompt for the server log (diagnostics).
    private static string Trunc(string s)
    {
        s = (s ?? "").Replace("\r", " ").Replace("\n", " ");
        return s.Length > 60 ? s[..60] + "…" : s;
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
