using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using ProjectAI.Models;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>Forward/structural tests for S1-3: embedding lookup, scatter-add backward, cross-entropy.</summary>
public class EmbeddingLossTests
{
    private static float[] Host(CpuComputeBackend be, Tensor t)
    {
        var buf = new float[t.ElementCount];
        be.ToHost(t, buf);
        return buf;
    }

    [Fact]
    public void Embedding_GathersRows()
    {
        using var be = new CpuComputeBackend();
        var emb = new Embedding(ParameterContext.Create(be, 1), vocabSize: 5, dim: 3);
        var w = Host(be, emb.Weight); // [5,3]
        var ids = be.FromHost([2f, 0f, 4f], new Shape(3), DType.F32);

        var y = emb.Forward(ids);
        Assert.Equal(new Shape(3, 3), y.Shape);
        var h = Host(be, y);
        int[] expectedRows = [2, 0, 4];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(w[expectedRows[i] * 3 + j], h[i * 3 + j]);
    }

    [Fact]
    public void Embedding_2DInput_ProducesBatchedRows()
    {
        using var be = new CpuComputeBackend();
        var emb = new Embedding(ParameterContext.Create(be, 1), vocabSize: 6, dim: 2);
        var ids = be.FromHost([1f, 2f, 3f, 4f], new Shape(2, 2), DType.F32);
        Assert.Equal(new Shape(2, 2, 2), emb.Forward(ids).Shape);
    }

    [Fact]
    public void Embedding_GradientFlowsOnlyToUsedRows_AndAccumulatesRepeats()
    {
        using var be = new CpuComputeBackend();
        var ctx = ParameterContext.Create(be, 1);
        var emb = new Embedding(ctx, vocabSize: 5, dim: 3);
        var ids = be.FromHost([1f, 1f, 3f], new Shape(3), DType.F32); // row 1 used twice, row 3 once

        var loss = ctx.Ag.Mean(emb.Forward(ids)); // mean over 9 elements → each output grad = 1/9
        loss.Backward();

        var g = Host(be, emb.Weight.Grad!); // [5,3]
        foreach (int unused in new[] { 0, 2, 4 })
            for (int j = 0; j < 3; j++)
                Assert.Equal(0f, g[unused * 3 + j]); // unused rows get no gradient
        for (int j = 0; j < 3; j++)
        {
            Assert.Equal(2f / 9f, g[1 * 3 + j], 5); // row 1: two contributions
            Assert.Equal(1f / 9f, g[3 * 3 + j], 5); // row 3: one
        }
    }

    [Fact]
    public void CrossEntropy_MatchesReference()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var logits = be.FromHost([1f, 2f, 3f, 1f, 1f, 1f], new Shape(2, 3), DType.F32);
        var targets = be.FromHost([2f, 0f], new Shape(2), DType.F32);

        float loss = Host(be, Loss.CrossEntropy(ag, logits, targets))[0];
        // row0: -log softmax([1,2,3])[2]; row1: -log softmax([1,1,1])[0] = -log(1/3).
        float r0 = -MathF.Log(MathF.Exp(3) / (MathF.Exp(1) + MathF.Exp(2) + MathF.Exp(3)));
        float r1 = -MathF.Log(1f / 3f);
        Assert.Equal((r0 + r1) / 2f, loss, 4);
    }

    [Fact]
    public void CrossEntropy_IgnoresPaddingTargets()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var logits = be.FromHost([1f, 2f, 3f, 1f, 1f, 1f], new Shape(2, 3), DType.F32);
        var targets = be.FromHost([2f, 0f], new Shape(2), DType.F32);

        // Ignore rows whose target == 0 → only row0 counts.
        float loss = Host(be, Loss.CrossEntropy(ag, logits, targets, ignoreIndex: 0))[0];
        float r0 = -MathF.Log(MathF.Exp(3) / (MathF.Exp(1) + MathF.Exp(2) + MathF.Exp(3)));
        Assert.Equal(r0, loss, 4);
    }

    [Fact]
    public void CrossEntropy_RejectsOutOfRangeTarget()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var logits = be.FromHost([1f, 2f, 3f, 10f, 20f, 30f], new Shape(2, 3), DType.F32); // vocab 3
        var targets = be.FromHost([4f, 0f], new Shape(2), DType.F32); // target 4 is out of range on a non-last row
        Assert.Throws<ArgumentOutOfRangeException>(() => Loss.CrossEntropy(ag, logits, targets));
    }

    [Fact]
    public void CrossEntropy_Rank3_MatchesFlattenedRank2()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var data = new float[2 * 2 * 3];
        for (int i = 0; i < data.Length; i++) data[i] = (i % 5) * 0.4f - 0.6f;
        float[] tgt = [1f, 0f, 2f, 1f];

        var loss3 = Host(be, Loss.CrossEntropy(ag,
            be.FromHost(data, new Shape(2, 2, 3), DType.F32), be.FromHost(tgt, new Shape(2, 2), DType.F32)))[0];
        var loss2 = Host(be, Loss.CrossEntropy(ag,
            be.FromHost(data, new Shape(4, 3), DType.F32), be.FromHost(tgt, new Shape(4), DType.F32)))[0];
        Assert.Equal(loss2, loss3, 5); // rank-3 flatten must align each (batch,seq) row with its target
    }

    [Fact]
    public void CrossEntropy_PerfectPrediction_NearZeroLoss()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var logits = be.FromHost([100f, 0f, 0f, 0f, 100f, 0f], new Shape(2, 3), DType.F32);
        var targets = be.FromHost([0f, 1f], new Shape(2), DType.F32);
        Assert.True(Host(be, Loss.CrossEntropy(ag, logits, targets))[0] < 1e-3f);
    }
}
