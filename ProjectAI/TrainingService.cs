using ProjectAI.Core;
using ProjectAI.Models;
using ProjectAI.Tokenizers;
using ProjectAI.Training;

// Runs ONE training job at a time on a background thread so the (single-threaded) HTTP loop stays responsive:
// POST /train starts a job and returns immediately; GET /train/status polls its progress; the finished model is
// saved into the models directory, where it shows up in the picker. The server gates /generate while a job runs
// so the GPU isn't used by two operations at once.
internal sealed class TrainingService
{
    private readonly object _gate = new();
    private TrainJob? _job;

    /// <summary>The current/last job, or null if none has been started.</summary>
    public TrainJob? Current { get { lock (_gate) return _job; } }

    /// <summary>True while a job is actively training.</summary>
    public bool IsTraining { get { lock (_gate) return _job is { Done: false }; } }

    /// <summary>Starts a job (rejected if one is already running, the size is unknown, or the backend can't start).</summary>
    public (bool Ok, string Message) Start(ComputeRegistry compute, string modelsDirectory, TrainStartRequest req)
    {
        lock (_gate)
        {
            if (_job is { Done: false }) return (false, "a training job is already running");

            ModelConfig config;
            try
            {
                var preset = ModelPresets.Get(req.Size);
                config = preset with { MaxSequenceLength = Math.Max(preset.MaxSequenceLength, req.SeqLen) };
            }
            catch (ArgumentException ex) { return (false, ex.Message); }

            IComputeBackend backend;
            try { (backend, _) = compute.Resolve(req.Backend); }
            catch (Exception ex) { return (false, $"backend '{req.Backend}' is unavailable: {ex.Message}"); }

            var job = new TrainJob(req.Name, req.Size, req.Backend, req.Steps);
            _job = job;
            // Background thread: the HTTP loop returns 202 immediately and serves status polls while this runs.
            Task.Run(() => Run(backend, modelsDirectory, req, config, job));
            return (true, "started");
        }
    }

    private static void Run(IComputeBackend backend, string modelsDirectory, TrainStartRequest req, ModelConfig config, TrainJob job)
    {
        try
        {
            var tokenizer = new BpeTokenizer([]); // byte-level
            bool checkpointing = req.Size == "large"; // the only preset that needs it to fit a modest GPU (S3-2)
            var outcome = ModelTrainer.TrainOnText(
                backend, req.Text, config, tokenizer, req.Batch, req.SeqLen, req.Steps, req.LearningRate,
                gradientCheckpointing: checkpointing,
                onStep: (step, loss) => { job.Step = step; job.Loss = loss; });

            Directory.CreateDirectory(modelsDirectory);
            string path = Path.Combine(modelsDirectory, req.Name + ".ckpt");
            Checkpointing.SaveModel(path, outcome.Model, config, tokenizer, outcome.Report.FinalStep, optimizer: null, backend);

            job.Loss = outcome.Report.LastLoss;
            job.Step = outcome.Report.FinalStep;
            job.Status = "done";
        }
        catch (Exception ex)
        {
            job.Error = ex.Message;
            job.Status = "error";
        }
    }
}

// Progress for a training job. The background thread writes; the status endpoint reads. Step/Status/Error use
// volatile (or atomic) access; Loss is a plain float (aligned 32-bit reads are atomic — fine for a progress display).
internal sealed class TrainJob(string name, string size, string backend, int totalSteps)
{
    public string Name { get; } = name;
    public string Size { get; } = size;
    public string Backend { get; } = backend;
    public int TotalSteps { get; } = totalSteps;
    public volatile int Step;
    public float Loss;
    public volatile string Status = "running"; // running | done | error
    public volatile string? Error;
    public bool Done => Status != "running";
}

internal sealed record TrainStartRequest(string Name, string Text, string Size, int Steps, int Batch, int SeqLen, float LearningRate, string Backend);
