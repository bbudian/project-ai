using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using ProjectAI.Models;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>End-to-end tests for the assembled LlamaModel (ticket S1-8).</summary>
public class LlamaModelTests
{
    private static ModelConfig TinyConfig() => new()
    {
        VocabSize = 8, EmbeddingDim = 16, LayerCount = 1, HeadCount = 2, KvHeadCount = 1,
        FeedForwardHiddenDim = 32, MaxSequenceLength = 16,
    };

    [Fact]
    public void Forward_ProducesVocabLogits()
    {
        using var be = new CpuComputeBackend();
        var model = new LlamaModel(ParameterContext.Create(be, 1), TinyConfig());
        var ids = be.FromHost([1, 2, 3, 4, 0, 5, 6, 7], new Shape(2, 4), DType.F32); // [batch=2, seq=4]
        Assert.Equal(new Shape(2, 4, 8), model.Forward(ids).Shape);
    }

    [Fact]
    public void Overfits_TwoLayerModel()
    {
        using var be = new CpuComputeBackend();
        var ctx = ParameterContext.Create(be, 11);
        var model = new LlamaModel(ctx, TinyConfig() with { LayerCount = 2 }); // residual stacking across 2 blocks
        var input = be.FromHost([1, 2, 3, 4, 5], new Shape(1, 5), DType.F32);
        var target = be.FromHost([2, 3, 4, 5, 6], new Shape(1, 5), DType.F32);
        var opt = new AdamW(model.Parameters().ToList(), be, learningRate: 0.02f, weightDecay: 0f);

        var lossHost = new float[1];
        float last = float.NaN;
        for (int step = 0; step < 250; step++)
        {
            opt.ZeroGrad();
            var loss = Loss.CrossEntropy(ctx.Ag, model.Forward(input), target);
            loss.Backward();
            opt.Step();
            be.ToHost(loss, lossHost);
            last = lossHost[0];
        }
        Assert.True(last < 0.1f, $"2-layer model should overfit; final loss {last}");
    }

    [Fact]
    public void Overfits_AND_GreedilyReproduces_AFixedSequence()
    {
        using var be = new CpuComputeBackend();
        var ctx = ParameterContext.Create(be, 7);
        var ag = ctx.Ag;
        var model = new LlamaModel(ctx, TinyConfig());

        // Next-token objective on one fixed sequence: input 1..6 → target 2..7.
        var input = be.FromHost([1, 2, 3, 4, 5, 6], new Shape(1, 6), DType.F32);
        var target = be.FromHost([2, 3, 4, 5, 6, 7], new Shape(1, 6), DType.F32);
        var opt = new AdamW(model.Parameters().ToList(), be, learningRate: 0.02f, weightDecay: 0f);

        var lossHost = new float[1];
        float first = float.NaN, last = float.NaN;
        for (int step = 0; step < 300; step++)
        {
            opt.ZeroGrad();
            var loss = Loss.CrossEntropy(ag, model.Forward(input), target);
            loss.Backward();
            opt.Step();
            be.ToHost(loss, lossHost);
            if (step == 0) first = lossHost[0];
            last = lossHost[0];
        }

        Assert.True(first > 1.5f, $"initial loss {first} should be ≈ ln(8)=2.08 (untrained)");
        Assert.True(last < 0.1f, $"final loss {last} should overfit to near zero (the model must be learning end-to-end)");

        // Greedy argmax over the trained logits must reproduce the memorized next tokens.
        var logits = new float[6 * 8];
        be.ToHost(model.Forward(input), logits);
        int[] expected = [2, 3, 4, 5, 6, 7];
        for (int pos = 0; pos < 6; pos++)
        {
            int arg = 0;
            for (int t = 1; t < 8; t++) if (logits[pos * 8 + t] > logits[pos * 8 + arg]) arg = t;
            Assert.Equal(expected[pos], arg);
        }
    }
}
