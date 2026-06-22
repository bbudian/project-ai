using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>
/// Finite-difference gradient checks for every differentiable op in the <see cref="Autograd"/> facade
/// (ticket S0-6). Each op's loss here is at most quadratic in its inputs, so central differences carry
/// no truncation error — only float32 round-off — and a tight relative tolerance is appropriate.
/// </summary>
public class GradientCheckTests
{
    private static float[] HostOf(CpuComputeBackend be, Tensor t)
    {
        var buf = new float[t.ElementCount];
        be.ToHost(t, buf);
        return buf;
    }

    private static float ScalarNoGrad(CpuComputeBackend be, Func<Tensor> forward)
    {
        using (GradMode.NoGrad())
        {
            var t = forward();
            Span<float> s = stackalloc float[1];
            be.ToHost(t, s);
            return s[0];
        }
    }

    /// <summary>Compares analytic gradients (from Backward) to central differences for each leaf input.</summary>
    private static void GradCheck(
        CpuComputeBackend be, Func<Tensor> forward, Tensor[] inputs,
        float eps = 1e-2f, float rtol = 2e-2f, float atol = 2e-3f)
    {
        foreach (var inp in inputs)
        {
            inp.RequiresGrad = true;
            inp.Grad = null;
        }

        forward().Backward();
        foreach (var inp in inputs)
        {
            Assert.NotNull(inp.Grad);
            Assert.True(inp.Grad!.Shape.Equals(inp.Shape),
                $"gradient shape {inp.Grad.Shape} does not match input shape {inp.Shape}");
        }
        var analytic = inputs.Select(i => HostOf(be, i.Grad!)).ToArray();

        for (int ii = 0; ii < inputs.Length; ii++)
        {
            var buf = (float[])inputs[ii].Handle!;
            for (int e = 0; e < buf.Length; e++)
            {
                float original = buf[e];
                buf[e] = original + eps;
                float lossPlus = ScalarNoGrad(be, forward);
                buf[e] = original - eps;
                float lossMinus = ScalarNoGrad(be, forward);
                buf[e] = original;

                float numeric = (lossPlus - lossMinus) / (2f * eps);
                float a = analytic[ii][e];
                Assert.True(Math.Abs(numeric - a) <= atol + rtol * Math.Abs(numeric),
                    $"input {ii} elem {e}: numeric {numeric}, analytic {a}");
            }
        }
    }

    [Fact]
    public void Add_WithBroadcast()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var a = be.FromHost([0.2f, -0.5f, 1.1f, 0.7f, -1.3f, 0.4f], new Shape(2, 3), DType.F32);
        var b = be.FromHost([0.3f, -0.8f, 1.5f], new Shape(3), DType.F32); // broadcasts over rows
        GradCheck(be, () => ag.Mean(ag.Add(a, b)), [a, b]);
    }

    [Fact]
    public void Sub_Elementwise()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var a = be.FromHost([1f, -2f, 3f, -4f], new Shape(2, 2), DType.F32);
        var b = be.FromHost([0.5f, 0.25f, -0.5f, 2f], new Shape(2, 2), DType.F32);
        GradCheck(be, () => ag.Mean(ag.Sub(a, b)), [a, b]);
    }

    [Fact]
    public void Mul_WithBroadcast()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var a = be.FromHost([0.5f, -1.0f, 2.0f, 0.3f, -0.7f, 1.1f], new Shape(2, 3), DType.F32);
        var b = be.FromHost([1.5f, -0.5f], new Shape(2, 1), DType.F32); // broadcasts over columns
        GradCheck(be, () => ag.Mean(ag.Mul(a, b)), [a, b]);
    }

    [Fact]
    public void MatMul_2D()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(11);
        var a = be.FromHost(Rand(rng, 3 * 4), new Shape(3, 4), DType.F32);
        var b = be.FromHost(Rand(rng, 4 * 2), new Shape(4, 2), DType.F32);
        GradCheck(be, () => ag.Mean(ag.MatMul(a, b)), [a, b]);
    }

    [Fact]
    public void Mean_Reduction()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var x = be.FromHost([1f, 2f, 3f, 4f, 5f, 6f], new Shape(2, 3), DType.F32);
        GradCheck(be, () => ag.Mean(x), [x]);
    }

    [Fact]
    public void Composite_MeanSquaredError()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(5);
        var x = be.FromHost(Rand(rng, 4 * 3), new Shape(4, 3), DType.F32); // constant (no grad)
        var y = be.FromHost(Rand(rng, 4 * 1), new Shape(4, 1), DType.F32); // constant (no grad)
        var w = be.FromHost(Rand(rng, 3 * 1), new Shape(3, 1), DType.F32);
        var b = be.FromHost([0.1f], new Shape(1), DType.F32);

        // loss = mean((X·W + b - Y)^2); checks grads wrt W and b (and exercises diff's fan-out).
        Tensor Forward()
        {
            var diff = ag.Sub(ag.Add(ag.MatMul(x, w), b), y);
            return ag.Mean(ag.Mul(diff, diff));
        }
        GradCheck(be, Forward, [w, b]);
    }

    [Fact]
    public void Add_BroadcastAcrossTwoLeadingAxes()
    {
        // Regression: a [C] operand added into [N, T, C] broadcasts across TWO leading axes — exactly
        // the transformer bias/norm pattern. The grad must sum back to [C].
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(3);
        var big = be.FromHost(Rand(rng, 2 * 4 * 3), new Shape(2, 4, 3), DType.F32);
        var small = be.FromHost(Rand(rng, 3), new Shape(3), DType.F32);
        GradCheck(be, () => ag.Mean(ag.Add(big, small)), [big, small]);
    }

    [Fact]
    public void Mul_BroadcastAcrossThreeLeadingAxes()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(4);
        var big = be.FromHost(Rand(rng, 2 * 2 * 2 * 2), new Shape(2, 2, 2, 2), DType.F32);
        var small = be.FromHost(Rand(rng, 2), new Shape(2), DType.F32);
        GradCheck(be, () => ag.Mean(ag.Mul(big, small)), [big, small]);
    }

    [Fact]
    public void Transpose_InForwardGraph()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var x = be.FromHost([1f, 2f, 3f, 4f], new Shape(2, 2), DType.F32);
        // loss = mean(x * xᵀ): the gradient must flow through the transpose back to the SAME leaf.
        GradCheck(be, () => ag.Mean(ag.Mul(x, ag.Transpose(x, 0, 1))), [x]);
    }

    [Fact]
    public void Reshape_InForwardGraph()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(8);
        var x = be.FromHost(Rand(rng, 6), new Shape(2, 3), DType.F32);
        var c = be.FromHost(Rand(rng, 6), new Shape(6), DType.F32);
        GradCheck(be, () => ag.Mean(ag.Mul(ag.Reshape(x, 6), c)), [x]);
    }

    [Fact]
    public void Slice_InForwardGraph()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var x = be.FromHost([1f, 2f, 3f, 4f, 5f], new Shape(5), DType.F32);
        GradCheck(be, () => ag.Mean(ag.Slice(x, 0, 1, 3)), [x]); // only x[1..4] should get gradient
    }

    [Fact]
    public void FanOut_LeafFeedsTwoSeparateOps()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(6);
        var x = be.FromHost(Rand(rng, 4), new Shape(2, 2), DType.F32);
        var a = be.FromHost(Rand(rng, 4), new Shape(2, 2), DType.F32);
        var b = be.FromHost(Rand(rng, 4), new Shape(2, 2), DType.F32);
        // x is consumed by two independent muls → gradients must accumulate.
        GradCheck(be, () => ag.Mean(ag.Add(ag.Mul(x, a), ag.Mul(x, b))), [x]);
    }

    [Fact]
    public void Backward_Twice_DoesNotDoubleCount()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var x = be.FromHost([1f, 2f, 3f, 4f], new Shape(2, 2), DType.F32);
        x.RequiresGrad = true;

        var loss = ag.Mean(ag.Mul(x, x));
        loss.Backward();
        var first = HostOf(be, x.Grad!);

        x.Grad = null;     // as the optimizer's ZeroGrad would
        loss.Backward();   // re-run on the SAME graph
        var second = HostOf(be, x.Grad!);

        Assert.Equal(first, second); // intermediate grads cleared → no double counting
    }

    [Fact]
    public void DifferentiableMatMul_RejectsNon2D()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var a = be.FromHost(new float[8], new Shape(2, 2, 2), DType.F32);
        Assert.Throws<NotImplementedException>(() => ag.MatMul(a, a));
    }

    [Fact]
    public void NoGrad_SuppressesGraphAndNestsCorrectly()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var x = be.FromHost([1f, 2f], new Shape(2), DType.F32);
        x.RequiresGrad = true;

        Tensor y;
        using (GradMode.NoGrad()) y = ag.Mul(x, x);
        Assert.Null(y.GradFn);
        Assert.False(y.RequiresGrad);

        Assert.True(GradMode.IsEnabled);
        using (GradMode.NoGrad())
        {
            Assert.False(GradMode.IsEnabled);
            using (GradMode.NoGrad()) Assert.False(GradMode.IsEnabled);
            Assert.False(GradMode.IsEnabled); // inner dispose restores "disabled", not "enabled"
        }
        Assert.True(GradMode.IsEnabled);
    }

    private static float[] Rand(Random rng, int count)
    {
        var v = new float[count];
        for (int i = 0; i < count; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
        return v;
    }
}
