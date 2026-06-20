using ProjectAI.Core;
using ProjectAI.Models;

namespace ProjectAI.Training;

/// <summary>Knobs for a training run.</summary>
public sealed record TrainingConfig
{
    public int BatchSize { get; init; } = 8;
    public int SequenceLength { get; init; } = 512;
    public float LearningRate { get; init; } = 3e-4f;
    public int MaxSteps { get; init; } = 1_000;
    public int GradientAccumulationSteps { get; init; } = 1;
    public int WarmupSteps { get; init; } = 100;
    public string CheckpointDirectory { get; init; } = "checkpoints";
}

/// <summary>A source of tokenized training sequences.</summary>
public interface IDataset
{
    int Count { get; }
    ReadOnlyMemory<int> GetSequence(int index);
}

/// <summary>Drives the optimization loop over a dataset.</summary>
public interface ITrainer
{
    void Train(LlamaModel model, IDataset dataset, TrainingConfig config);
}

/// <summary>
/// Reference training loop: forward -&gt; cross-entropy loss -&gt; backward -&gt; AdamW step -&gt; checkpoint,
/// with gradient accumulation and warmup. Implemented in Stage 1 (ticket S1-10).
/// </summary>
public sealed class Trainer(IComputeBackend backend) : ITrainer
{
    public IComputeBackend Backend { get; } = backend;
    public void Train(LlamaModel model, IDataset dataset, TrainingConfig config) =>
        throw new NotImplementedException("ticket S1-10.");
}
