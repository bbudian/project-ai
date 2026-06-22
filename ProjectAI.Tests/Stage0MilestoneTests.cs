using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>
/// The Stage 0 milestone (BUILD_PLAN.md §6): train a tiny <c>y = Wx + b</c> on synthetic data to
/// near-zero loss using only our Tensor + autograd + AdamW on the CPU backend. This is the first
/// end-to-end "it learns" result, asserted so it shows up directly in <c>dotnet test</c>.
/// </summary>
public class Stage0MilestoneTests
{
    [Fact]
    public void LinearRegression_LossDropsToNearZero_AndRecoversTrueParameters()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(0);
        const int n = 128, din = 3, dout = 1;
        float[] wTrue = [2f, -3f, 0.5f];
        const float bTrue = 1.5f;

        var xBuf = new float[n * din];
        var yBuf = new float[n * dout];
        for (int i = 0; i < n; i++)
        {
            float dot = 0f;
            for (int j = 0; j < din; j++)
            {
                float v = (float)(rng.NextDouble() * 2 - 1);
                xBuf[i * din + j] = v;
                dot += v * wTrue[j];
            }
            yBuf[i] = dot + bTrue;
        }

        var x = be.FromHost(xBuf, new Shape(n, din), DType.F32);
        var y = be.FromHost(yBuf, new Shape(n, dout), DType.F32);
        var w = be.FromHost(new float[din * dout], new Shape(din, dout), DType.F32);
        var bias = be.FromHost(new float[dout], new Shape(dout), DType.F32);
        w.RequiresGrad = true;
        bias.RequiresGrad = true;

        var opt = new AdamW([w, bias], be, learningRate: 0.1f, weightDecay: 0f);

        float first = float.NaN, last = float.NaN;
        var lossHost = new float[1];
        for (int step = 1; step <= 200; step++)
        {
            opt.ZeroGrad();
            var pred = ag.Add(ag.MatMul(x, w), bias);
            var diff = ag.Sub(pred, y);
            var loss = ag.Mean(ag.Mul(diff, diff));
            loss.Backward();
            opt.Step();

            be.ToHost(loss, lossHost);
            if (step == 1) first = lossHost[0];
            last = lossHost[0];
        }

        Assert.True(first > 1f, $"expected a non-trivial initial loss, got {first}");
        Assert.True(last < 1e-3f, $"final loss {last} should be < 1e-3 (the model should have learned)");

        var wOut = new float[din];
        be.ToHost(w, wOut);
        for (int j = 0; j < din; j++)
            Assert.True(MathF.Abs(wOut[j] - wTrue[j]) < 1e-2f, $"W[{j}]={wOut[j]}, expected≈{wTrue[j]}");
        Assert.True(MathF.Abs(Bias(be, bias) - bTrue) < 1e-2f);
    }

    private static float Bias(CpuComputeBackend be, Tensor b)
    {
        Span<float> s = stackalloc float[1];
        be.ToHost(b, s);
        return s[0];
    }
}
