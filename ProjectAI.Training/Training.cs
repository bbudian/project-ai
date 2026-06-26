using ProjectAI.Core;
using ProjectAI.Formats;
using ProjectAI.Models;

namespace ProjectAI.Training;

/// <summary>Knobs for a training run.</summary>
public sealed record TrainingConfig
{
    public int BatchSize { get; init; } = 8;
    public int SequenceLength { get; init; } = 512;
    public float LearningRate { get; init; } = 3e-4f;
    public float WeightDecay { get; init; } = 0.01f;
    public int MaxSteps { get; init; } = 1_000;
    public int GradientAccumulationSteps { get; init; } = 1;
    /// <summary>Recompute each transformer block's activations in backward instead of storing them (S3-2): less peak memory, ~one extra forward per block.</summary>
    public bool GradientCheckpointing { get; init; }
    public int WarmupSteps { get; init; } = 100;
    /// <summary>Seed for batch sampling, so a run is reproducible.</summary>
    public ulong Seed { get; init; }
    /// <summary>Write a checkpoint every N steps (in addition to the final one); 0 disables periodic checkpoints.</summary>
    public int CheckpointEverySteps { get; init; }
    public string CheckpointDirectory { get; init; } = "checkpoints";
    /// <summary>Optional per-step progress callback (step, stepLoss); keeps Console out of the training library.</summary>
    public Action<int, float>? OnStep { get; init; }
}

/// <summary>A source of tokenized training sequences.</summary>
public interface IDataset
{
    int Count { get; }
    ReadOnlyMemory<int> GetSequence(int index);
}

/// <summary>Per-step loss history and final step count from a training run.</summary>
public sealed record TrainingReport(IReadOnlyList<float> StepLosses, int FinalStep)
{
    public float FirstLoss => StepLosses.Count > 0 ? StepLosses[0] : float.NaN;
    public float LastLoss => StepLosses.Count > 0 ? StepLosses[^1] : float.NaN;
}

/// <summary>Drives the optimization loop over a dataset.</summary>
public interface ITrainer
{
    TrainingReport Train(LlamaModel model, IDataset dataset, TrainingConfig config);
}

/// <summary>
/// Reference training loop (ticket S1-10): forward → cross-entropy → backward → AdamW step → checkpoint, with
/// gradient accumulation and a warmup+cosine learning-rate schedule. Checkpoints (model weights + optimizer
/// moments + step) are written via <see cref="Checkpoint"/> so a run reproduces identical logits on reload and
/// can resume. The loss-scaling for accumulation relies on parameter grads being leaf-accumulated across the
/// micro-batch <c>Backward()</c> calls (the optimizer's <c>ZeroGrad</c> is what clears them between steps).
/// </summary>
public sealed class Trainer(IComputeBackend backend) : ITrainer
{
    public IComputeBackend Backend { get; } = backend;

    public TrainingReport Train(LlamaModel model, IDataset dataset, TrainingConfig config)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(dataset);
        ArgumentNullException.ThrowIfNull(config);
        if (dataset.Count == 0) throw new ArgumentException("dataset is empty.", nameof(dataset));

        var ag = new Autograd(Backend);
        model.GradientCheckpointing = config.GradientCheckpointing; // S3-2: trade recompute for activation memory
        var parameters = model.Parameters().ToList();
        var optimizer = new AdamW(parameters, Backend, learningRate: config.LearningRate, weightDecay: config.WeightDecay)
        {
            LearningRateSchedule = WarmupCosineSchedule(config),
        };
        var rng = new PcgRng(config.Seed);

        // Gradient accumulation: scale each micro-batch's mean loss by 1/N and sum the (leaf) parameter grads
        // across the N Backward() calls. This equals a single batch of size B*N ONLY when every micro-batch
        // contributes the same number of valid (non-ignored) tokens — which holds here because NextBatch emits
        // dense [B, S] blocks with no padding or ignore-index. Masked / variable-length accumulation would need
        // per-token-count weighting (sum unreduced losses, divide once by the total token count): a follow-up.
        int microSteps = Math.Max(1, config.GradientAccumulationSteps);
        var invMicro = Backend.FromHost([1f / microSteps], new Shape(1), DType.F32);
        var losses = new List<float>(config.MaxSteps);
        var lossHost = new float[1];

        if (!string.IsNullOrEmpty(config.CheckpointDirectory)) Directory.CreateDirectory(config.CheckpointDirectory);

        for (int step = 1; step <= config.MaxSteps; step++)
        {
            optimizer.ZeroGrad();
            float stepLoss = 0f;
            for (int micro = 0; micro < microSteps; micro++)
            {
                // Scope the forward+backward so this micro-batch's activation graph is freed deterministically at
                // the end (S2-3) — before the optimizer (or the next micro-batch) allocates — instead of lingering
                // until the GC runs. Only the accumulated leaf gradients are kept alive past the scope.
                using var scope = Backend.BeginScope();
                var (input, target) = NextBatch(dataset, config, rng);
                var logits = model.Forward(input);
                var loss = Loss.CrossEntropy(ag, logits, target);

                Backend.ToHost(loss, lossHost);
                stepLoss += lossHost[0] / microSteps;

                var scaled = microSteps > 1 ? ag.Mul(loss, invMicro) : loss; // mean over all micro-batches
                scaled.Backward();

                foreach (var p in parameters)
                    if (p.Grad is { } g) Backend.KeepAlive(g);
            }
            optimizer.Step();
            losses.Add(stepLoss);
            config.OnStep?.Invoke(step, stepLoss);

            if (config.CheckpointEverySteps > 0 && step % config.CheckpointEverySteps == 0)
                Save(model, optimizer, step, config);
        }

        Save(model, optimizer, config.MaxSteps, config); // final checkpoint
        return new TrainingReport(losses, config.MaxSteps);
    }

    private (Tensor Input, Tensor Target) NextBatch(IDataset dataset, TrainingConfig config, IRng rng)
    {
        int b = config.BatchSize, s = config.SequenceLength;
        var inBuf = new float[b * s];
        var tgtBuf = new float[b * s];
        for (int row = 0; row < b; row++)
        {
            int idx = (int)(rng.NextUInt32() % (uint)dataset.Count);
            var seq = dataset.GetSequence(idx).Span;
            if (seq.Length < s + 1)
                throw new InvalidOperationException(
                    $"dataset sequence {idx} has length {seq.Length}; need {s + 1} for sequence length {s}.");
            for (int t = 0; t < s; t++)
            {
                inBuf[row * s + t] = seq[t];
                tgtBuf[row * s + t] = seq[t + 1];
            }
        }
        return (Backend.FromHost(inBuf, new Shape(b, s), DType.F32), Backend.FromHost(tgtBuf, new Shape(b, s), DType.F32));
    }

    private void Save(LlamaModel model, AdamW optimizer, int step, TrainingConfig config)
    {
        if (string.IsNullOrEmpty(config.CheckpointDirectory)) return;
        string path = Path.Combine(config.CheckpointDirectory, $"step-{step}.ckpt");
        Checkpoint.Save(path, step, Checkpointing.EnumerateState(model, optimizer), Backend);
    }

    /// <summary>
    /// Restores a resume checkpoint into <paramref name="model"/> (and, if supplied, <paramref name="optimizer"/>),
    /// overwriting parameter data in place so subsequent forwards reproduce the saved logits. Returns the saved
    /// step. The model must have the same architecture the checkpoint was written with. (For inference, prefer
    /// <see cref="Checkpointing.LoadModel"/>, which rebuilds the model from the checkpoint's stored config.)
    /// </summary>
    public static int Restore(string path, LlamaModel model, AdamW? optimizer, IComputeBackend backend)
    {
        var (step, _, dict) = Checkpoint.Load(path, backend);
        Checkpointing.ApplyWeights(dict, model, backend);
        if (optimizer is not null) Checkpointing.ApplyOptimizer(dict, model, optimizer, step);
        return step;
    }

    // Linear warmup to the base LR over WarmupSteps, then cosine decay to a 10% floor by MaxSteps.
    private static Func<int, float> WarmupCosineSchedule(TrainingConfig config)
    {
        float baseLr = config.LearningRate;
        float minLr = baseLr * 0.1f;
        int total = Math.Max(1, config.MaxSteps);
        // Cap warmup below the total so the cosine phase always has range; otherwise WarmupSteps >= MaxSteps
        // would leave the run entirely in linear warmup, never reaching baseLr or decaying.
        int warmup = Math.Clamp(config.WarmupSteps, 0, Math.Max(0, total - 1));
        return step =>
        {
            if (warmup > 0 && step <= warmup) return baseLr * step / warmup;
            float denom = Math.Max(1, total - warmup);
            float progress = Math.Clamp((step - warmup) / (float)denom, 0f, 1f);
            float cosine = 0.5f * (1f + MathF.Cos(MathF.PI * progress));
            return minLr + (baseLr - minLr) * cosine;
        };
    }
}
