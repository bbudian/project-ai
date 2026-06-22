using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using ProjectAI.Models;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>
/// Finite-difference gradient checks for every differentiable op in the <see cref="Autograd"/> facade
/// (ticket S0-6). For the polynomial ops (Add/Sub/Mul/MatMul/Mean/views) the loss is ≤ quadratic in its
/// inputs, so central differences carry no truncation error and the default tight tolerance applies. The
/// transcendental S1-2 ops (Softmax/SiLU/RmsNorm) carry an O(eps²) truncation term, so they pass an
/// explicit, slightly looser tolerance chosen from the measured curvature (still ~10× above the real FD
/// error). RoPE is linear in x, so it stays on the tight default.
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
        float eps = Tolerances.FdStep, float rtol = Tolerances.GradRtol, float atol = Tolerances.GradAtol)
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
    public void MatMul_Batched_GradCheck()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(71);
        var a = be.FromHost(Rand(rng, 2 * 3 * 4), new Shape(2, 3, 4), DType.F32);
        var b = be.FromHost(Rand(rng, 2 * 4 * 5), new Shape(2, 4, 5), DType.F32);
        GradCheck(be, () => ag.Mean(ag.MatMul(a, b)), [a, b]);
    }

    [Fact]
    public void MatMul_BatchedBroadcast_GradCheck()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(72);
        var a = be.FromHost(Rand(rng, 2 * 4 * 3 * 5), new Shape(2, 4, 3, 5), DType.F32); // batch [2,4]
        var b = be.FromHost(Rand(rng, 2 * 1 * 5 * 6), new Shape(2, 1, 5, 6), DType.F32); // batch [2,1] broadcasts
        // grad_b must fold the broadcast head axis back to size 1 via ReduceGradToShape (the GQA reduction).
        GradCheck(be, () => ag.Mean(ag.MatMul(a, b)), [a, b]);
    }

    [Fact]
    public void MatMul_BatchedTransposeB_GradCheck()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(73);
        var a = be.FromHost(Rand(rng, 2 * 2 * 3 * 4), new Shape(2, 2, 3, 4), DType.F32);
        var b = be.FromHost(Rand(rng, 2 * 2 * 5 * 4), new Shape(2, 2, 5, 4), DType.F32); // b [.,n,k]=[5,4] → y [.,3,5]
        GradCheck(be, () => ag.Mean(ag.MatMul(a, b, transposeB: true)), [a, b]);
    }

    [Fact]
    public void MatMul_RankMismatch_GradCheck()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(74);
        var a = be.FromHost(Rand(rng, 2 * 3 * 4 * 5), new Shape(2, 3, 4, 5), DType.F32); // [B,H,M,K]
        var b = be.FromHost(Rand(rng, 5 * 6), new Shape(5, 6), DType.F32);               // [K,N] shared → grad_b sums the [2,3] batch
        GradCheck(be, () => ag.Mean(ag.MatMul(a, b)), [a, b]);
    }

    [Fact]
    public void MatMul_RankMismatch_TransposeB_And_LeftBroadcast_GradCheck()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(75);
        var a = be.FromHost(Rand(rng, 2 * 3 * 4 * 5), new Shape(2, 3, 4, 5), DType.F32);
        var bT = be.FromHost(Rand(rng, 6 * 5), new Shape(6, 5), DType.F32); // transposeB: [N,K]=[6,5] → y [.,4,6]
        GradCheck(be, () => ag.Mean(ag.MatMul(a, bT, transposeB: true)), [a, bT]);

        var a2 = be.FromHost(Rand(rng, 4 * 5), new Shape(4, 5), DType.F32);     // 2-D left, batched right
        var b2 = be.FromHost(Rand(rng, 2 * 5 * 6), new Shape(2, 5, 6), DType.F32);
        GradCheck(be, () => ag.Mean(ag.MatMul(a2, b2)), [a2, b2]); // grad_a sums the [2] batch
    }

    [Fact]
    public void Reshape_BackwardThroughTranspose_GradCheck()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(76);
        var x = be.FromHost(Rand(rng, 2 * 6), new Shape(2, 6), DType.F32);
        // The transpose backward hands a NON-contiguous grad to the reshape backward (the fixed bug class).
        GradCheck(be, () => ag.Mean(ag.Transpose(ag.Reshape(x, 2, 2, 3), 1, 2)), [x]);
    }

    [Fact]
    public void Attention_GradCheck_GroupedQuery()
    {
        using var be = new CpuComputeBackend();
        var ctx = ParameterContext.Create(be, 62);
        var cfg = new ModelConfig
        {
            VocabSize = 16, EmbeddingDim = 8, LayerCount = 1, HeadCount = 4, KvHeadCount = 2, // G = 2, KvH = 2
            FeedForwardHiddenDim = 16, MaxSequenceLength = 16,
        };
        var attn = new Attention(ctx, cfg);
        var rng = new Random(9);
        var x = be.FromHost(Rand(rng, 1 * 3 * 8), new Shape(1, 3, 8), DType.F32);
        var c = be.FromHost(Rand(rng, 1 * 3 * 8), new Shape(1, 3, 8), DType.F32);
        var leaves = new[] { x }.Concat(attn.Parameters()).ToArray();
        GradCheck(be, () => ctx.Ag.Mean(ctx.Ag.Mul(attn.Forward(x), c)), leaves,
            eps: Tolerances.FdStep, rtol: Tolerances.LooseRtol, atol: Tolerances.LooseAtol);
    }

    [Fact]
    public void Attention_GradCheck()
    {
        using var be = new CpuComputeBackend();
        var ctx = ParameterContext.Create(be, 61);
        // Small GQA config with KvHeadCount < HeadCount (G=2) to exercise the group-reduction path.
        var cfg = new ModelConfig
        {
            VocabSize = 16, EmbeddingDim = 4, LayerCount = 1, HeadCount = 2, KvHeadCount = 1,
            FeedForwardHiddenDim = 8, MaxSequenceLength = 16,
        };
        var attn = new Attention(ctx, cfg);
        var rng = new Random(8);
        var x = be.FromHost(Rand(rng, 1 * 3 * 4), new Shape(1, 3, 4), DType.F32);
        var c = be.FromHost(Rand(rng, 1 * 3 * 4), new Shape(1, 3, 4), DType.F32);
        var leaves = new[] { x }.Concat(attn.Parameters()).ToArray();
        GradCheck(be, () => ctx.Ag.Mean(ctx.Ag.Mul(attn.Forward(x), c)), leaves,
            eps: Tolerances.FdStep, rtol: Tolerances.LooseRtol, atol: Tolerances.LooseAtol);
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

    // ---- S1-2 transformer primitives (non-polynomial in x → looser relative tolerance) ----

    [Fact]
    public void Softmax_GradCheck()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(21);
        var x = be.FromHost(Rand(rng, 2 * 3), new Shape(2, 3), DType.F32);
        var c = be.FromHost(Rand(rng, 2 * 3), new Shape(2, 3), DType.F32); // constant weights
        GradCheck(be, () => ag.Mean(ag.Mul(ag.Softmax(x, -1), c)), [x], eps: Tolerances.FdStep, rtol: Tolerances.LooseRtol, atol: Tolerances.LooseAtol);
    }

    [Fact]
    public void Softmax_GradCheck_MiddleAxis()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(31);
        var x = be.FromHost(Rand(rng, 2 * 3 * 2), new Shape(2, 3, 2), DType.F32); // axisLen=3, inner=2
        var c = be.FromHost(Rand(rng, 2 * 3 * 2), new Shape(2, 3, 2), DType.F32);
        // Exercises the backward dot's axis argument off the last axis (inner>1).
        GradCheck(be, () => ag.Mean(ag.Mul(ag.Softmax(x, 1), c)), [x], eps: Tolerances.FdStep, rtol: Tolerances.LooseRtol, atol: Tolerances.LooseAtol);
    }

    [Fact]
    public void Silu_GradCheck()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        // Span the curvature-heavy region incl. the silu minimum near x≈-1.2784 and the saturating tails.
        var x = be.FromHost([-8f, -3f, -1.2784f, -0.5f, 0f, 0.5f, 1f, 3f, 8f], new Shape(9), DType.F32);
        GradCheck(be, () => ag.Mean(ag.Silu(x)), [x], eps: Tolerances.FdStep, rtol: Tolerances.LooseRtol, atol: Tolerances.LooseAtol);
    }

    [Fact]
    public void RmsNorm_GradCheck_X_And_Weight_MultiLeadingAxis()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(22);
        var x = be.FromHost(Rand(rng, 2 * 3 * 4), new Shape(2, 3, 4), DType.F32); // [N, T, D]
        var w = be.FromHost(Rand(rng, 4), new Shape(4), DType.F32);
        var c = be.FromHost(Rand(rng, 2 * 3 * 4), new Shape(2, 3, 4), DType.F32);
        // Checks grad wrt x (non-polynomial) AND weight (reduces over the [2,3] leading axes to [4]).
        GradCheck(be, () => ag.Mean(ag.Mul(ag.RmsNorm(x, w, 1e-5f), c)), [x, w], eps: Tolerances.FdStep, rtol: Tolerances.LooseRtol, atol: Tolerances.LooseAtol);
    }

    [Fact]
    public void RmsNorm_GradCheck_PreNormalizedInput()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        // RMS ≈ 1 (mean(x²) ≈ 1) — the regime where the inv vs inv² centering term is most fragile and
        // where normalized transformer activations actually live.
        var x = be.FromHost([1f, -1f, 0.9f, -1.1f, -1f, 1f, 1.1f, -0.9f], new Shape(2, 4), DType.F32);
        var w = be.FromHost([1.2f, -0.7f, 0.5f, 1f], new Shape(4), DType.F32);
        var c = be.FromHost([0.3f, -0.8f, 1.1f, -0.4f, 0.6f, 0.2f, -1f, 0.5f], new Shape(2, 4), DType.F32);
        GradCheck(be, () => ag.Mean(ag.Mul(ag.RmsNorm(x, w, 1e-5f), c)), [x, w], eps: Tolerances.FdStep, rtol: Tolerances.LooseRtol, atol: Tolerances.LooseAtol);
    }

    [Fact]
    public void Rope_GradCheck()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(23);
        var x = be.FromHost(Rand(rng, 2 * 4), new Shape(2, 4), DType.F32); // 2 positions, headDim 4
        // Constant rotation tables (duplicated rotate-half layout).
        float t0 = 0.4f, t1 = 0.9f;
        var cos = be.FromHost(
            [MathF.Cos(0 * t0), MathF.Cos(0 * t1), MathF.Cos(0 * t0), MathF.Cos(0 * t1),
             MathF.Cos(1 * t0), MathF.Cos(1 * t1), MathF.Cos(1 * t0), MathF.Cos(1 * t1)], new Shape(2, 4), DType.F32);
        var sin = be.FromHost(
            [MathF.Sin(0 * t0), MathF.Sin(0 * t1), MathF.Sin(0 * t0), MathF.Sin(0 * t1),
             MathF.Sin(1 * t0), MathF.Sin(1 * t1), MathF.Sin(1 * t0), MathF.Sin(1 * t1)], new Shape(2, 4), DType.F32);
        var weights = be.FromHost(Rand(rng, 2 * 4), new Shape(2, 4), DType.F32);
        // RoPE is linear in x → tight tolerance is fine.
        GradCheck(be, () => ag.Mean(ag.Mul(ag.RotaryEmbedding(x, cos, sin), weights)), [x]);
    }

    [Fact]
    public void Contiguous_GradCheck()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(9);
        var x = be.FromHost(Rand(rng, 2 * 3), new Shape(2, 3), DType.F32);
        var c = be.FromHost(Rand(rng, 3 * 2), new Shape(3, 2), DType.F32);
        // Densify a strided (transposed) view; gradient flows back through Contiguous (identity) + Transpose.
        GradCheck(be, () => ag.Mean(ag.Mul(ag.Contiguous(ag.Transpose(x, 0, 1)), c)), [x]);
    }

    // ---- S1-6 modules (gradients flow through the module + its parameters) ----

    [Fact]
    public void Linear_Module_GradCheck()
    {
        using var be = new CpuComputeBackend();
        var ctx = ParameterContext.Create(be, 41);
        var lin = new Linear(ctx, 3, 4); // non-square so grad_b is [4,3] — a transposed/swapped term becomes a shape error
        var rng = new Random(1);
        var x = be.FromHost(Rand(rng, 2 * 3), new Shape(2, 3), DType.F32);
        var c = be.FromHost(Rand(rng, 2 * 4), new Shape(2, 4), DType.F32);
        var leaves = new[] { x }.Concat(lin.Parameters()).ToArray();
        GradCheck(be, () => ctx.Ag.Mean(ctx.Ag.Mul(lin.Forward(x), c)), leaves); // bilinear → tight default tol
    }

    [Fact]
    public void RmsNorm_Module_GradCheck()
    {
        using var be = new CpuComputeBackend();
        var ctx = ParameterContext.Create(be, 42);
        var rms = new RmsNorm(ctx, 4, 1e-5f);
        var rng = new Random(2);
        var x = be.FromHost(Rand(rng, 2 * 4), new Shape(2, 4), DType.F32);
        var c = be.FromHost(Rand(rng, 2 * 4), new Shape(2, 4), DType.F32);
        var leaves = new[] { x }.Concat(rms.Parameters()).ToArray();
        GradCheck(be, () => ctx.Ag.Mean(ctx.Ag.Mul(rms.Forward(x), c)), leaves,
            eps: Tolerances.FdStep, rtol: Tolerances.LooseRtol, atol: Tolerances.LooseAtol);
    }

    [Fact]
    public void Rope_Module_GradCheck()
    {
        using var be = new CpuComputeBackend();
        var ctx = ParameterContext.Create(be, 43);
        var rope = new RotaryEmbedding(ctx, 4, 10000f, 8);
        var rng = new Random(3);
        var x = be.FromHost(Rand(rng, 1 * 1 * 2 * 4), new Shape(1, 1, 2, 4), DType.F32);
        var c = be.FromHost(Rand(rng, 1 * 1 * 2 * 4), new Shape(1, 1, 2, 4), DType.F32);
        // RoPE has no trainable params (cos/sin are constants); linear in x → tight default tol.
        GradCheck(be, () => ctx.Ag.Mean(ctx.Ag.Mul(rope.Apply(x, ForwardContext.Inference()), c)), [x]);
    }

    [Fact]
    public void SwiGlu_Module_GradCheck()
    {
        using var be = new CpuComputeBackend();
        var ctx = ParameterContext.Create(be, 44);
        var ffn = new SwiGluFeedForward(ctx, 3, 4);
        var rng = new Random(4);
        var x = be.FromHost(Rand(rng, 2 * 3), new Shape(2, 3), DType.F32);
        var c = be.FromHost(Rand(rng, 2 * 3), new Shape(2, 3), DType.F32);
        var leaves = new[] { x }.Concat(ffn.Parameters()).ToArray(); // x + 3 weight matrices
        GradCheck(be, () => ctx.Ag.Mean(ctx.Ag.Mul(ffn.Forward(x), c)), leaves,
            eps: Tolerances.FdStep, rtol: Tolerances.LooseRtol, atol: Tolerances.LooseAtol);
    }

    // ---- S1-3 embedding + loss ----

    [Fact]
    public void Embedding_GradCheck()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(51);
        var w = be.FromHost(Rand(rng, 5 * 3), new Shape(5, 3), DType.F32);
        int[] ids = [1, 3, 1, 0]; // row 1 repeated, rows 2 & 4 unused (must get zero grad)
        var c = be.FromHost(Rand(rng, 4 * 3), new Shape(4, 3), DType.F32);
        GradCheck(be, () => ag.Mean(ag.Mul(ag.Embedding(w, ids), c)), [w]); // linear in w → tight tol
    }

    [Fact]
    public void CrossEntropy_GradCheck()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(52);
        var logits = be.FromHost(Rand(rng, 3 * 4), new Shape(3, 4), DType.F32);
        int[] targets = [1, 3, 0];
        // CE is the scalar loss directly; non-polynomial (log/exp) → looser tolerance.
        GradCheck(be, () => ag.CrossEntropy(logits, targets, -100), [logits],
            eps: Tolerances.FdStep, rtol: Tolerances.LooseRtol, atol: Tolerances.LooseAtol);
    }

    [Fact]
    public void CrossEntropy_GradCheck_WithIgnoreIndex()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(54);
        var logits = be.FromHost(Rand(rng, 4 * 5), new Shape(4, 5), DType.F32);
        int[] targets = [2, -1, 1, -1]; // rows 1 & 3 ignored → validCount = 2 (exercises the 1/validCount scaling)
        GradCheck(be, () => ag.CrossEntropy(logits, targets, ignoreIndex: -1), [logits],
            eps: Tolerances.FdStep, rtol: Tolerances.LooseRtol, atol: Tolerances.LooseAtol);
    }

    [Fact]
    public void WeightTying_GradCheck_SumsBothPaths()
    {
        using var be = new CpuComputeBackend();
        var ag = new Autograd(be);
        var rng = new Random(53);
        var w = be.FromHost(Rand(rng, 4 * 3), new Shape(4, 3), DType.F32); // shared [vocab=4, dim=3]
        int[] ids = [1, 2];
        var c = be.FromHost(Rand(rng, 2 * 4), new Shape(2, 4), DType.F32);
        // w feeds BOTH the embedding lookup and the tied output projection; the gradient must accumulate
        // across both paths — a finite-difference check over w validates the summed gradient.
        GradCheck(be, () => ag.Mean(ag.Mul(ag.MatMul(ag.Embedding(w, ids), w, transposeB: true), c)), [w]);
    }

    private static float[] Rand(Random rng, int count)
    {
        var v = new float[count];
        for (int i = 0; i < count; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
        return v;
    }
}
