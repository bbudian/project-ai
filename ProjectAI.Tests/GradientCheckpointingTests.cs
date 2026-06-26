using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using ProjectAI.Models;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>
/// Gradient checkpointing (ticket S3-2) must be transparent to training: recomputing each block's activations in
/// the backward pass has to yield the EXACT same gradients as the normal store-everything path. This compares the
/// per-parameter gradients of one forward+backward with and without checkpointing on the CPU oracle — the decisive
/// correctness proof for the recompute/surrogate machinery in <see cref="Autograd.Checkpoint"/>.
/// </summary>
public class GradientCheckpointingTests
{
    [Fact]
    public void CheckpointedGradientsMatchTheStandardPath()
    {
        using var backend = new CpuComputeBackend();
        var config = new ModelConfig
        {
            VocabSize = 16,
            EmbeddingDim = 32,
            LayerCount = 3, // ≥2 blocks so checkpointing actually exercises the recompute path per block
            HeadCount = 4,
            KvHeadCount = 2,
            FeedForwardHiddenDim = 64,
            MaxSequenceLength = 16,
        };
        var model = new LlamaModel(ParameterContext.Create(backend, seed: 0), config); // deterministic init

        const int batch = 2, seq = 6;
        var inputData = new float[batch * seq];
        var targets = new int[batch * seq];
        for (int i = 0; i < inputData.Length; i++)
        {
            inputData[i] = i % config.VocabSize;            // deterministic token ids
            targets[i] = (i * 7 + 3) % config.VocabSize;    // deterministic next-token targets
        }

        var baseline = ParameterGrads(backend, model, config, inputData, targets, checkpoint: false);
        var checkpointed = ParameterGrads(backend, model, config, inputData, targets, checkpoint: true);

        Assert.Equal(baseline.Count, checkpointed.Count);
        foreach (var (name, expected) in baseline)
        {
            var actual = checkpointed[name];
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(
                    MathF.Abs(expected[i] - actual[i]) <= 1e-4f + 1e-4f * MathF.Abs(expected[i]),
                    $"gradient mismatch for '{name}'[{i}]: standard {expected[i]} vs checkpointed {actual[i]}");
        }
    }

    // One forward → cross-entropy → backward; returns each parameter's gradient as host floats.
    private static Dictionary<string, float[]> ParameterGrads(
        CpuComputeBackend backend, LlamaModel model, ModelConfig config, float[] inputData, int[] targets, bool checkpoint)
    {
        foreach (var (_, p) in model.NamedParameters()) p.Grad = null; // clear before this pass
        model.GradientCheckpointing = checkpoint;

        var ag = new Autograd(backend);
        var logits = model.Forward(backend.FromHost(inputData, new Shape(2, inputData.Length / 2), DType.F32));
        var flat = ag.Reshape(logits, targets.Length, config.VocabSize);
        var loss = ag.CrossEntropy(flat, targets, ignoreIndex: -100);
        loss.Backward();

        var grads = new Dictionary<string, float[]>();
        foreach (var (name, p) in model.NamedParameters())
        {
            var g = new float[p.ElementCount];
            if (p.Grad is not null) backend.ToHost(p.Grad, g);
            grads[name] = g;
        }
        return grads;
    }
}
