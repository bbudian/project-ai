using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using ProjectAI.Models;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>Forward-correctness tests for the S1-6 transformer modules (wiring on the autograd facade).</summary>
public class ModuleLayerTests
{
    private static float[] Host(CpuComputeBackend be, Tensor t)
    {
        var buf = new float[t.ElementCount];
        be.ToHost(t, buf);
        return buf;
    }

    private static void AssertClose(float[] expected, float[] actual, float tol = 1e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tol, $"index {i}: expected {expected[i]}, got {actual[i]}");
    }

    /// <summary>Reference for y = x · Wᵀ with W stored [out, in].</summary>
    private static float[] LinearRef(float[] x, int rows, int inDim, float[] w, int outDim)
    {
        var y = new float[rows * outDim];
        for (int r = 0; r < rows; r++)
            for (int o = 0; o < outDim; o++)
            {
                float acc = 0;
                for (int i = 0; i < inDim; i++) acc += x[r * inDim + i] * w[o * inDim + i];
                y[r * outDim + o] = acc;
            }
        return y;
    }

    private static float Silu(float v) => v * (1f / (1f + MathF.Exp(-v)));

    [Fact]
    public void Linear_Forward_MatchesReference_2D()
    {
        using var be = new CpuComputeBackend();
        var lin = new Linear(ParameterContext.Create(be, 1), inDim: 3, outDim: 2);
        var w = Host(be, lin.Parameters().First());
        var x = be.FromHost([1, 2, 3, 4, 5, 6], new Shape(2, 3), DType.F32);
        AssertClose(LinearRef([1, 2, 3, 4, 5, 6], 2, 3, w, 2), Host(be, lin.Forward(x)));
    }

    [Fact]
    public void Linear_Forward_FlattensLeadingDims_3D()
    {
        using var be = new CpuComputeBackend();
        var lin = new Linear(ParameterContext.Create(be, 2), inDim: 3, outDim: 4);
        var w = Host(be, lin.Parameters().First());
        var xBuf = new float[2 * 5 * 3];
        for (int i = 0; i < xBuf.Length; i++) xBuf[i] = (i % 7) * 0.1f - 0.3f;
        var x = be.FromHost(xBuf, new Shape(2, 5, 3), DType.F32);

        var y = lin.Forward(x);
        Assert.Equal(new Shape(2, 5, 4), y.Shape);
        AssertClose(LinearRef(xBuf, 2 * 5, 3, w, 4), Host(be, y));
    }

    [Fact]
    public void Linear_HandlesNonContiguousInput()
    {
        using var be = new CpuComputeBackend();
        var lin = new Linear(ParameterContext.Create(be, 7), inDim: 3, outDim: 2);
        var w = Host(be, lin.Parameters().First());
        // [2,3,4] → transpose last two axes → [2,4,3] (non-contiguous, last dim = inDim = 3).
        var raw = new float[2 * 3 * 4];
        for (int i = 0; i < raw.Length; i++) raw[i] = (i % 5) * 0.2f - 0.4f;
        var x = be.FromHost(raw, new Shape(2, 3, 4), DType.F32).Transpose(1, 2);
        Assert.False(x.IsContiguous);

        var y = lin.Forward(x); // must not throw (Linear densifies via Ag.Contiguous)
        Assert.Equal(new Shape(2, 4, 2), y.Shape);
        AssertClose(LinearRef(Host(be, x), 2 * 4, 3, w, 2), Host(be, y));
    }

    [Fact]
    public void Rope_Module_AcceptsFullSequencePositions()
    {
        using var be = new CpuComputeBackend();
        var rope = new RotaryEmbedding(ParameterContext.Create(be, 1), headDim: 4, theta: 10000f, maxSequenceLength: 8);
        var x = be.FromHost(new float[1 * 1 * 2 * 4], new Shape(1, 1, 2, 4), DType.F32);
        var positions = be.FromHost([0f, 1f], new Shape(2), DType.F32);
        // Full-sequence positions are accepted (used as 0..seq-1); only the KV-cache decode offset is deferred.
        var y = rope.Apply(x, ForwardContext.Inference() with { Positions = positions });
        Assert.Equal(new Shape(1, 1, 2, 4), y.Shape);
    }

    [Fact]
    public void RmsNorm_Module_MatchesReference()
    {
        using var be = new CpuComputeBackend();
        var rms = new RmsNorm(ParameterContext.Create(be, 1), dim: 4, epsilon: 1e-5f); // weight = ones
        var x = be.FromHost([1, 2, 3, 4], new Shape(4), DType.F32);
        float inv = 1f / MathF.Sqrt(7.5f + 1e-5f);
        AssertClose([1 * inv, 2 * inv, 3 * inv, 4 * inv], Host(be, rms.Forward(x)), 1e-5f);
    }

    [Fact]
    public void Rope_Module_Position0_IsIdentity_AndPreservesPairNorm()
    {
        using var be = new CpuComputeBackend();
        var rope = new RotaryEmbedding(ParameterContext.Create(be, 1), headDim: 4, theta: 10000f, maxSequenceLength: 8);
        // x: [batch=1, heads=1, seq=2, headDim=4]
        var x = be.FromHost([0.5f, -1f, 2f, 0.3f, 1f, 2f, 3f, 4f], new Shape(1, 1, 2, 4), DType.F32);
        var y = Host(be, rope.Apply(x, ForwardContext.Inference()));

        // Position 0 (seq index 0): cos=1, sin=0 → identity.
        AssertClose([0.5f, -1f, 2f, 0.3f], y[..4], 1e-5f);
        // Per-pair norm preserved for both sequence positions (pairs (i, i+2)).
        for (int s = 0; s < 2; s++)
        {
            float n0 = y[s * 4 + 0] * y[s * 4 + 0] + y[s * 4 + 2] * y[s * 4 + 2];
            float n1 = y[s * 4 + 1] * y[s * 4 + 1] + y[s * 4 + 3] * y[s * 4 + 3];
            var xr = x; var xh = Host(be, xr);
            Assert.Equal(xh[s * 4 + 0] * xh[s * 4 + 0] + xh[s * 4 + 2] * xh[s * 4 + 2], n0, 3);
            Assert.Equal(xh[s * 4 + 1] * xh[s * 4 + 1] + xh[s * 4 + 3] * xh[s * 4 + 3], n1, 3);
        }
    }

    [Fact]
    public void SwiGlu_Module_MatchesReference()
    {
        using var be = new CpuComputeBackend();
        var ff = new SwiGluFeedForward(ParameterContext.Create(be, 3), dim: 3, hiddenDim: 4);
        var named = ff.NamedParameters().ToDictionary(n => n.Name, n => Host(be, n.Param));
        var wg = named["gate.weight"]; // [4,3]
        var wu = named["up.weight"];   // [4,3]
        var wd = named["down.weight"]; // [3,4]

        var xBuf = new float[] { 0.2f, -0.5f, 1.1f, 0.7f, -1.3f, 0.4f }; // [2,3]
        var x = be.FromHost(xBuf, new Shape(2, 3), DType.F32);

        var gate = LinearRef(xBuf, 2, 3, wg, 4);
        var up = LinearRef(xBuf, 2, 3, wu, 4);
        var hidden = new float[2 * 4];
        for (int i = 0; i < hidden.Length; i++) hidden[i] = Silu(gate[i]) * up[i];
        var expected = LinearRef(hidden, 2, 4, wd, 3);

        Assert.Equal(new Shape(2, 3), ff.Forward(x).Shape);
        AssertClose(expected, Host(be, ff.Forward(x)), 1e-4f);
    }
}
