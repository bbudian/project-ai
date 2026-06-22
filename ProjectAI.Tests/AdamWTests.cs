using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using Xunit;

namespace ProjectAI.Tests;

public class AdamWTests
{
    private static float Scalar(CpuComputeBackend be, Tensor t)
    {
        Span<float> s = stackalloc float[1];
        be.ToHost(t, s);
        return s[0];
    }

    [Fact]
    public void MinimizesConvexQuadratic_Under200Steps()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);

        // f(w) = mean((w - c)^2): convex, minimum 0 at w = c.
        var target = be.FromHost([3f, -2f, 0.5f, 1f, -4f], new Shape(5), DType.F32);
        var w = be.FromHost(new float[5], new Shape(5), DType.F32);
        w.RequiresGrad = true;

        var opt = new AdamW([w], be, learningRate: 0.1f, weightDecay: 0f);

        float loss = float.PositiveInfinity;
        for (int step = 0; step < 200; step++)
        {
            opt.ZeroGrad();
            var diff = ag.Sub(w, target);
            var l = ag.Mean(ag.Mul(diff, diff));
            l.Backward();
            opt.Step();
            loss = Scalar(be, l);
        }

        Assert.True(loss < 1e-4f, $"final loss {loss} should be < 1e-4");

        var learned = new float[5];
        be.ToHost(w, learned);
        float[] expected = [3f, -2f, 0.5f, 1f, -4f];
        for (int i = 0; i < 5; i++)
            Assert.True(MathF.Abs(learned[i] - expected[i]) < 1e-2f, $"w[{i}]={learned[i]}, expected≈{expected[i]}");
    }

    [Fact]
    public void DecoupledWeightDecay_MatchesClosedFormStep()
    {
        using var be = new CpuComputeBackend();
        // With a zero gradient, the adaptive term is 0, so the only update is decoupled decay:
        // θ ← θ - lr·λ·θ = θ·(1 - lr·λ).
        const float lr = 0.1f, wd = 0.1f;
        var p = be.FromHost([1f, 2f, -3f], new Shape(3), DType.F32);
        p.RequiresGrad = true;
        p.Grad = be.FromHost([0f, 0f, 0f], new Shape(3), DType.F32);

        var opt = new AdamW([p], be, learningRate: lr, weightDecay: wd);
        opt.Step();

        var after = new float[3];
        be.ToHost(p, after);
        float factor = 1f - lr * wd; // 0.99
        Assert.Equal(1f * factor, after[0], 5);
        Assert.Equal(2f * factor, after[1], 5);
        Assert.Equal(-3f * factor, after[2], 5);
    }

    [Fact]
    public void LearningRateSchedule_OverridesBaseRate()
    {
        using var be = new CpuComputeBackend();
        // Zero gradient + decay: the per-step decay factor reflects the *scheduled* lr.
        const float wd = 0.1f;
        var p = be.FromHost([1f], new Shape(1), DType.F32);
        p.RequiresGrad = true;
        p.Grad = be.FromHost([0f], new Shape(1), DType.F32);

        var opt = new AdamW([p], be, learningRate: 99f, weightDecay: wd) { LearningRateSchedule = _ => 0.2f };
        opt.Step();

        Assert.Equal(1f * (1f - 0.2f * wd), Scalar(be, p), 5); // uses 0.2, not 99
        Assert.Equal(1, opt.StepCount);
    }

    [Fact]
    public void PerParameterTimestep_DeferredParameterGetsFullFirstStep()
    {
        using var be = new CpuComputeBackend();
        // Two params share an optimizer; b's gradient only arrives after 5 steps. With a per-parameter
        // timestep, b's first real update uses t=1 (full bias correction) → it moves by ≈ lr, not the
        // much smaller amount a global timestep (t=6) would produce.
        var a = be.FromHost([0f], new Shape(1), DType.F32);
        var b = be.FromHost([0f], new Shape(1), DType.F32);
        a.RequiresGrad = true;
        b.RequiresGrad = true;
        var opt = new AdamW([a, b], be, learningRate: 0.1f, weightDecay: 0f);

        for (int step = 0; step < 5; step++)
        {
            a.Grad = be.FromHost([1f], new Shape(1), DType.F32);
            b.Grad = null; // b not yet receiving gradients
            opt.Step();
        }

        float bBefore = Scalar(be, b);
        a.Grad = be.FromHost([1f], new Shape(1), DType.F32);
        b.Grad = be.FromHost([1f], new Shape(1), DType.F32);
        opt.Step();
        float bAfter = Scalar(be, b);

        Assert.Equal(0f, bBefore, 5);                  // b untouched until its gradient arrived
        Assert.Equal(-0.1, bAfter - bBefore, 2);       // full t=1 step ≈ -lr
    }
}
