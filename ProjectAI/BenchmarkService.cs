using ProjectAI.Bench;
using ProjectAI.Core;

// Runs ONE benchmark at a time on a background thread — the structural twin of TrainingService: POST /benchmark
// starts a run and returns 202 immediately, GET /benchmark/status polls progress, POST /benchmark/cancel stops it
// between generations. The runner takes the server's InferenceLock PER CASE (never for the whole run), so /health
// stays responsive and nothing else can touch the backend mid-generation; /generate and /chat are additionally
// 409-gated while a run is live (the GPU is not big enough for two workloads).
internal sealed class BenchmarkService
{
    private readonly object _gate = new();
    private BenchJob? _job;
    private CancellationTokenSource? _cts;

    public BenchJob? Current { get { lock (_gate) return _job; } }

    public bool IsBenchmarking { get { lock (_gate) return _job is { IsDone: false }; } }

    public (bool Ok, string Message, string RunId, int Total) Start(
        ComputeRegistry compute, string modelsDirectory, BenchStartRequest req, object inferenceLock)
    {
        lock (_gate)
        {
            if (_job is { IsDone: false }) return (false, "a benchmark run is already in progress", "", 0);

            BenchSuite suite;
            try { suite = SuiteLoader.Load(req.Suite, modelsDirectory); }
            catch (Exception ex) { return (false, ex.Message, "", 0); }

            if (req.Models.Count == 0) return (false, "benchmark needs at least one model", "", 0);
            foreach (string m in req.Models)
                if (!File.Exists(Path.Combine(modelsDirectory, m + ".ckpt")))
                    return (false, $"unknown model '{m}'", "", 0);

            IComputeBackend backend;
            try { (backend, _) = compute.Resolve(req.Backend); }
            catch (Exception ex) { return (false, $"backend '{req.Backend}' is unavailable: {ex.Message}", "", 0); }

            string runId = BenchRunner.NewRunId();
            int total = suite.Cases.Count * req.Models.Count;
            var job = new BenchJob(runId, suite.Id, req.Models, req.Backend, total);
            var cts = new CancellationTokenSource();
            _job = job;
            _cts = cts;

            var config = new BenchRunConfig(suite.Id, req.Models, req.Backend, req.Seed, req.Sample, req.Repeats);
            Task.Run(() => Run(backend, modelsDirectory, suite, config, job, inferenceLock, cts.Token));
            return (true, "started", runId, total);
        }
    }

    public void Cancel() { lock (_gate) _cts?.Cancel(); }

    private static void Run(
        IComputeBackend backend, string modelsDirectory, BenchSuite suite, BenchRunConfig config,
        BenchJob job, object inferenceLock, CancellationToken cancel)
    {
        try
        {
            var run = BenchRunner.Run(
                backend, config.Backend, modelsDirectory, suite, config,
                onProgress: (model, caseId, done, total) => { job.CurrentModel = model; job.CurrentCase = caseId; job.Done = done; },
                cancel: cancel,
                exclusive: work => { lock (inferenceLock) work(); }, // per-unit lock: the run never wedges the API
                runId: job.RunId);
            BenchRunner.SaveRun(modelsDirectory, run);
            job.Done = job.Total;
            job.Status = run.State; // done | canceled
        }
        catch (Exception ex)
        {
            job.Error = ex.Message;
            job.Status = "error";
        }
    }
}

// Progress for a benchmark run: the background thread writes, the status endpoint reads (volatile/atomic fields).
internal sealed class BenchJob(string runId, string suiteId, IReadOnlyList<string> models, string backend, int total)
{
    public string RunId { get; } = runId;
    public string SuiteId { get; } = suiteId;
    public IReadOnlyList<string> Models { get; } = models;
    public string Backend { get; } = backend;
    public int Total { get; } = total;
    public volatile int Done;
    public volatile string CurrentModel = "";
    public volatile string CurrentCase = "";
    public volatile string Status = "running"; // running | done | canceled | error
    public volatile string? Error;
    public bool IsDone => Status != "running";
}

internal sealed record BenchStartRequest(
    string Suite, IReadOnlyList<string> Models, string Backend, ulong Seed, bool Sample, int Repeats);
