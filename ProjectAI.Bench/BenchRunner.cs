using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using ProjectAI.Core;
using ProjectAI.Models;
using ProjectAI.Training;

namespace ProjectAI.Bench;

/// <summary>
/// Executes a benchmark run: for each model, score bpb once over the suite's eval corpus (if any), then run every
/// case Repeats+1 times (the first generation is a warmup whose timing is discarded — caches/JIT/device init make
/// it unrepresentative) and record the median timing plus the deterministic checks. One front-end is the
/// <c>projectai bench</c> CLI; the serve endpoint drives the same code path.
/// </summary>
public static class BenchRunner
{
    /// <summary>Per-case progress: (model, caseId, done, total). Cancellation is checked between generations.</summary>
    public static BenchRun Run(
        IComputeBackend be, string backendId, string modelsDir, BenchSuite suite, BenchRunConfig config,
        Action<string, string, int, int>? onProgress = null, CancellationToken cancel = default)
    {
        string runId = $"bench-{DateTime.UtcNow:yyyyMMdd-HHmmss}-{Guid.NewGuid().ToString("N")[..6]}";
        string started = DateTime.UtcNow.ToString("O");
        var cells = new List<CellResult>();
        var stamps = new List<ModelStamp>();
        var bpbByModel = new Dictionary<string, double>();
        int total = suite.Cases.Count * config.Models.Count;
        int done = 0;
        string state = "done";

        foreach (string modelName in config.Models)
        {
            if (cancel.IsCancellationRequested) { state = "canceled"; break; }

            string path = Path.Combine(modelsDir, modelName + ".ckpt");
            LlamaModel model;
            ModelConfig modelConfig;
            Tokenizers.ITokenizer tokenizer;
            int step;
            try
            {
                (model, modelConfig, tokenizer, step) = Checkpointing.LoadModel(path, be);
                stamps.Add(new ModelStamp(modelName, step, CountParams(modelConfig),
                    $"d{modelConfig.EmbeddingDim}·L{modelConfig.LayerCount}·h{modelConfig.HeadCount}/{modelConfig.KvHeadCount}·ctx{modelConfig.MaxSequenceLength}",
                    Sha256Of(path)));
            }
            catch (Exception e)
            {
                foreach (var c in suite.Cases)
                    cells.Add(new CellResult(modelName, c.Id, "", 0, 0, "error", 0, 0, [], 0, Error: $"load failed: {e.Message}"));
                done += suite.Cases.Count;
                continue;
            }

            // bpb once per model (not per case): the corpus is the quality probe, the cases are behavior probes.
            if (!string.IsNullOrEmpty(suite.EvalCorpus))
            {
                try { bpbByModel[modelName] = BpbScorer.Score(be, model, tokenizer, modelConfig, suite.EvalCorpus); }
                catch (Exception e) { cells.Add(new CellResult(modelName, "__bpb__", "", 0, 0, "error", 0, 0, [], 0, Error: $"bpb failed: {e.Message}")); }
            }

            foreach (var benchCase in suite.Cases)
            {
                if (cancel.IsCancellationRequested) { state = "canceled"; break; }
                onProgress?.Invoke(modelName, benchCase.Id, done, total);
                cells.Add(RunCase(be, model, tokenizer, modelConfig, modelName, benchCase, config, cancel));
                done++;
            }
        }

        var aggregates = Aggregate(config.Models, cells, bpbByModel);
        var meta = new RunMeta(
            typeof(BenchRunner).Assembly.GetName().Version?.ToString() ?? "dev",
            Environment.MachineName, Environment.OSVersion.ToString(), backendId, stamps);
        return new BenchRun(runId, suite.Id, backendId, started, DateTime.UtcNow.ToString("O"),
            config, meta, cells, aggregates, state);
    }

    private static CellResult RunCase(
        IComputeBackend be, LlamaModel model, Tokenizers.ITokenizer tokenizer, ModelConfig modelConfig,
        string modelName, BenchCase benchCase, BenchRunConfig config, CancellationToken cancel)
    {
        try
        {
            // Decoding held constant across all models in the run: greedy by default; sampling only if the run
            // explicitly asks, and then with the run's fixed seed so it still reproduces.
            ISampler sampler = config.Sample
                ? new TopKTopPSampler(new PcgRng(config.Seed), temperature: 0.8f, topK: 40, topP: 0.9f)
                : new GreedySampler();

            Inference.GenerationResult? result = null;
            var seconds = new List<double>();
            for (int r = 0; r <= config.Repeats; r++) // r==0 is the warmup: timing discarded, output ignored
            {
                if (cancel.IsCancellationRequested) break;
                var sw = Stopwatch.StartNew();
                var attempt = Inference.GenerateText(be, model, tokenizer, modelConfig, benchCase.Prompt, sampler, benchCase.MaxTokens);
                sw.Stop();
                if (r == 0) { result = attempt; continue; }
                seconds.Add(sw.Elapsed.TotalSeconds);
                result = attempt; // greedy → identical output every repeat; keep the last, timed one
            }
            if (result is null || seconds.Count == 0)
                return new CellResult(modelName, benchCase.Id, "", 0, 0, "canceled", 0, 0, [], 0, Error: "canceled");

            double medianSeconds = Median(seconds);
            double tokPerSec = medianSeconds > 0 ? result.GeneratedTokens / medianSeconds : 0;
            var checks = CheckEvaluator.Evaluate(benchCase, result.Continuation, result.StopReason);
            return new CellResult(modelName, benchCase.Id, result.Continuation, result.PromptTokens,
                result.GeneratedTokens, result.StopReason, Math.Round(medianSeconds, 3), Math.Round(tokPerSec, 2),
                checks, CheckEvaluator.PassRate(checks));
        }
        catch (Exception e)
        {
            return new CellResult(modelName, benchCase.Id, "", 0, 0, "error", 0, 0, [], 0, Error: e.Message);
        }
    }

    /// <summary>Standard median (mean of the middle pair for even counts); 0 for an empty list.</summary>
    public static double Median(IReadOnlyList<double> values)
    {
        if (values.Count == 0) return 0;
        var sorted = values.OrderBy(v => v).ToList();
        int mid = sorted.Count / 2;
        return sorted.Count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2.0;
    }

    private static IReadOnlyList<ArmAggregate> Aggregate(
        IReadOnlyList<string> models, IReadOnlyList<CellResult> cells, IReadOnlyDictionary<string, double> bpbByModel)
    {
        var aggregates = new List<ArmAggregate>();
        foreach (string model in models)
        {
            var mine = cells.Where(c => c.Model == model && c.CaseId != "__bpb__" && c.Error is null).ToList();
            var stops = mine.GroupBy(c => c.Stop).ToDictionary(g => g.Key, g => g.Count());
            aggregates.Add(new ArmAggregate(
                model, mine.Count, mine.Count,
                bpbByModel.TryGetValue(model, out double bpb) ? Math.Round(bpb, 4) : null,
                Math.Round(Median(mine.Select(c => c.MedianTokPerSec).ToList()), 2),
                mine.Count == 0 ? 0 : Math.Round(mine.Average(c => c.CheckPassRate), 3),
                stops));
        }
        return aggregates;
    }

    private static long CountParams(ModelConfig c)
    {
        long d = c.EmbeddingDim, ffn = c.FeedForwardHiddenDim, kvDim = (long)c.KvHeadCount * c.HeadDim;
        long perLayer = 2L * d * d + 2L * d * kvDim + 3L * d * ffn + 2L * d;
        return (long)c.VocabSize * d + c.LayerCount * perLayer + d;
    }

    private static string Sha256Of(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    // ---- persistence -------------------------------------------------------------------------------------------

    private static readonly JsonSerializerOptions JsonOpts = new() { WriteIndented = true };

    /// <summary>Writes the run JSON under &lt;modelsDir&gt;/benchmarks/runs/&lt;id&gt;.json and returns the path.</summary>
    public static string SaveRun(string modelsDir, BenchRun run)
    {
        string dir = Path.Combine(modelsDir, "benchmarks", "runs");
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, run.Id + ".json");
        string tmp = path + ".tmp";
        File.WriteAllText(tmp, JsonSerializer.Serialize(run, JsonOpts));
        File.Move(tmp, path, overwrite: true);
        return path;
    }

    public static BenchRun? LoadRun(string modelsDir, string id)
    {
        string path = Path.Combine(modelsDir, "benchmarks", "runs", id + ".json");
        if (!File.Exists(path)) return null;
        return JsonSerializer.Deserialize<BenchRun>(File.ReadAllText(path));
    }

    public static IReadOnlyList<BenchRun> ListRuns(string modelsDir)
    {
        string dir = Path.Combine(modelsDir, "benchmarks", "runs");
        if (!Directory.Exists(dir)) return [];
        var runs = new List<BenchRun>();
        foreach (string file in Directory.GetFiles(dir, "bench-*.json").OrderByDescending(f => f, StringComparer.Ordinal))
            try
            {
                var run = JsonSerializer.Deserialize<BenchRun>(File.ReadAllText(file));
                if (run is not null) runs.Add(run);
            }
            catch (JsonException) { /* skip corrupt run files; never break the listing */ }
        return runs;
    }
}
