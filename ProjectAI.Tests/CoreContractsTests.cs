using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using Xunit;

namespace ProjectAI.Tests;

public class CoreContractsTests
{
    [Fact]
    public void Shape_ElementCount_IsProductOfDimensions()
    {
        var shape = new Shape(2, 3, 4);
        Assert.Equal(24L, shape.ElementCount);
        Assert.Equal(3, shape.Rank);
        Assert.Equal("[2, 3, 4]", shape.ToString());
    }

    [Fact]
    public void CpuBackend_Add_ComputesElementwiseSum()
    {
        using var backend = new CpuComputeBackend();
        var a = backend.FromHost([1f, 2f, 3f], new Shape(3), DType.F32);
        var b = backend.FromHost([10f, 20f, 30f], new Shape(3), DType.F32);

        var sum = backend.Add(a, b);

        var result = new float[3];
        backend.ToHost(sum, result);
        Assert.Equal(new float[] { 11f, 22f, 33f }, result);
    }

    [Fact]
    public void DType_SizeInBytes_IsCorrect()
    {
        Assert.Equal(4, DType.F32.SizeInBytes());
        Assert.Equal(2, DType.BF16.SizeInBytes());
        Assert.Equal(8, DType.I64.SizeInBytes());
    }

    [Fact]
    public void Autograd_NumericGradientCheck_MatchesFiniteDifferences()
    {
        // loss = mean(a * b); analytic d(loss)/d(a_i) = b_i / N — checked against central differences.
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);

        var a = be.FromHost([0.5f, -1.2f, 2.0f, 0.3f], new Shape(2, 2), DType.F32);
        var b = be.FromHost([1f, 2f, 3f, 4f], new Shape(2, 2), DType.F32);
        a.RequiresGrad = true;

        Tensor Forward() => ag.Mean(ag.Mul(a, b));

        a.Grad = null;
        Forward().Backward();
        var analytic = new float[4];
        be.ToHost(a.Grad!, analytic);

        var buf = (float[])a.Handle!;
        const float eps = 1e-2f;
        for (int i = 0; i < buf.Length; i++)
        {
            float original = buf[i];
            buf[i] = original + eps;
            float lossPlus = ScalarNoGrad(be, Forward);
            buf[i] = original - eps;
            float lossMinus = ScalarNoGrad(be, Forward);
            buf[i] = original;

            float numeric = (lossPlus - lossMinus) / (2f * eps);
            Assert.True(Math.Abs(numeric - analytic[i]) <= 2e-3f + 2e-2f * Math.Abs(numeric),
                $"a[{i}]: numeric {numeric}, analytic {analytic[i]}");
        }
    }

    private static float ScalarNoGrad(CpuComputeBackend be, Func<Tensor> forward)
    {
        using (GradMode.NoGrad())
        {
            var t = forward();
            var s = new float[1];
            be.ToHost(t, s);
            return s[0];
        }
    }
}
