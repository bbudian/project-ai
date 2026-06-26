using ProjectAI.Core;
using ProjectAI.Models;
using ProjectAI.Tokenizers;

namespace ProjectAI.Training;

/// <summary>
/// One-call training orchestration shared by the trainer CLI, the <c>serve</c> training endpoint, and the future
/// in-app Train tab: tokenize the text into next-token blocks, build a model for the given config, run the real
/// <see cref="Trainer"/>, and hand back the trained model + loss report. Backend selection and checkpoint saving
/// stay with the caller (the composition root) — this is just the build-and-train core, kept DRY in one place.
/// </summary>
public static class ModelTrainer
{
    public sealed record TrainOutcome(LlamaModel Model, TrainingReport Report);

    public static TrainOutcome TrainOnText(
        IComputeBackend backend, string text, ModelConfig config, ITokenizer tokenizer,
        int batch, int seqLen, int steps, float learningRate, bool gradientCheckpointing = false, Action<int, float>? onStep = null)
    {
        var dataset = new TextDataset(text, tokenizer, seqLen);
        if (dataset.Count == 0)
            throw new ArgumentException("text is too short for even one training block; use a longer corpus or a smaller sequence length.");

        var model = new LlamaModel(ParameterContext.Create(backend, 0), config);
        var report = new Trainer(backend).Train(model, dataset, new TrainingConfig
        {
            BatchSize = batch,
            SequenceLength = seqLen,
            LearningRate = learningRate,
            MaxSteps = steps,
            WarmupSteps = Math.Clamp(steps / 10, 1, Math.Max(1, steps - 1)),
            GradientCheckpointing = gradientCheckpointing,
            CheckpointDirectory = "", // the caller saves the final, self-describing checkpoint
            OnStep = onStep,
        });
        return new TrainOutcome(model, report);
    }
}
