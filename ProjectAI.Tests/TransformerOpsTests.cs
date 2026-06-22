using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>Forward-correctness tests for the S1-2 transformer primitives on the CPU oracle.</summary>
public class TransformerOpsTests
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

    [Fact]
    public void Softmax_MatchesReference_AndSumsToOne()
    {
        using var be = new CpuComputeBackend();
        var x = be.FromHost([1f, 2f, 3f], new Shape(3), DType.F32);
        var y = Host(be, be.Softmax(x, 0));
        AssertClose([0.09003057f, 0.24472847f, 0.66524096f], y);
        Assert.Equal(1f, y[0] + y[1] + y[2], 5);
    }

    [Fact]
    public void Softmax_Along_LastAxis_Of2D()
    {
        using var be = new CpuComputeBackend();
        var x = be.FromHost([1f, 2f, 3f, 1f, 1f, 1f], new Shape(2, 3), DType.F32);
        var y = Host(be, be.Softmax(x, -1));
        AssertClose([0.09003057f, 0.24472847f, 0.66524096f], y[..3]);
        AssertClose([1f / 3, 1f / 3, 1f / 3], y[3..]); // uniform row
    }

    [Fact]
    public void Softmax_StableForLargeLogits()
    {
        using var be = new CpuComputeBackend();
        var y = Host(be, be.Softmax(be.FromHost([80f, 81f, 82f], new Shape(3), DType.F32), 0));
        Assert.All(y, v => Assert.False(float.IsNaN(v) || float.IsInfinity(v)));
        AssertClose([0.09003057f, 0.24472847f, 0.66524096f], y); // shift-invariant
    }

    [Fact]
    public void Softmax_OverMiddleAxis_MatchesPerColumnReference()
    {
        using var be = new CpuComputeBackend();
        var src = new float[12];
        for (int i = 0; i < 12; i++) src[i] = i * 0.3f;
        var x = be.FromHost(src, new Shape(2, 3, 2), DType.F32); // axisLen=3 (axis 1), inner=2

        var y = Host(be, be.Softmax(x, 1));

        // Independent per-column reference: softmax over k=0..2 at src[o*6 + k*2 + c].
        for (int o = 0; o < 2; o++)
            for (int c = 0; c < 2; c++)
            {
                float max = float.NegativeInfinity;
                for (int k = 0; k < 3; k++) max = MathF.Max(max, src[o * 6 + k * 2 + c]);
                float sum = 0f;
                for (int k = 0; k < 3; k++) sum += MathF.Exp(src[o * 6 + k * 2 + c] - max);
                for (int k = 0; k < 3; k++)
                {
                    float expected = MathF.Exp(src[o * 6 + k * 2 + c] - max) / sum;
                    Assert.Equal(expected, y[o * 6 + k * 2 + c], 5);
                }
            }

        // Negative-axis equivalence.
        AssertClose(y, Host(be, be.Softmax(x, -2)));
    }

    [Fact]
    public void Silu_MatchesReference()
    {
        using var be = new CpuComputeBackend();
        var y = Host(be, be.Silu(be.FromHost([0f, 1f, 2f, -1f, -10f], new Shape(5), DType.F32)));
        static float Ref(float v) => v * (1f / (1f + MathF.Exp(-v)));
        AssertClose([Ref(0), Ref(1), Ref(2), Ref(-1), Ref(-10)], y, 1e-5f);
        Assert.Equal(0f, y[0]); // silu(0) = 0
    }

    [Fact]
    public void RmsNorm_MatchesReference()
    {
        using var be = new CpuComputeBackend();
        var x = be.FromHost([1f, 2f, 3f, 4f], new Shape(4), DType.F32);
        var w = be.FromHost([1f, 1f, 1f, 1f], new Shape(4), DType.F32);
        var y = Host(be, be.RmsNorm(x, w, 1e-6f));
        float inv = 1f / MathF.Sqrt(7.5f + 1e-6f); // mean(x^2) = 30/4 = 7.5
        AssertClose([1 * inv, 2 * inv, 3 * inv, 4 * inv], y, 1e-5f);
    }

    [Fact]
    public void RmsNorm_AppliesWeight_AndHandlesZeroRow()
    {
        using var be = new CpuComputeBackend();
        // Two rows: [1,2,3,4] scaled by weight, and an all-zero row (must stay finite, = 0).
        var x = be.FromHost([1f, 2f, 3f, 4f, 0f, 0f, 0f, 0f], new Shape(2, 4), DType.F32);
        var w = be.FromHost([2f, 0.5f, 1f, 3f], new Shape(4), DType.F32);
        var y = Host(be, be.RmsNorm(x, w, 1e-6f));
        float inv = 1f / MathF.Sqrt(7.5f + 1e-6f);
        AssertClose([1 * inv * 2, 2 * inv * 0.5f, 3 * inv * 1, 4 * inv * 3], y[..4], 1e-5f);
        AssertClose([0, 0, 0, 0], y[4..]);
        Assert.All(y, v => Assert.False(float.IsNaN(v)));
    }

    [Fact]
    public void Rope_AtPositionZero_IsIdentity()
    {
        using var be = new CpuComputeBackend();
        var x = be.FromHost([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f], new Shape(2, 4), DType.F32);
        var cos = be.FromHost([1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f], new Shape(2, 4), DType.F32);
        var sin = be.FromHost([0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f], new Shape(2, 4), DType.F32);
        AssertClose([1, 2, 3, 4, 5, 6, 7, 8], Host(be, be.RotaryEmbedding(x, cos, sin)));
    }

    [Fact]
    public void RmsNorm_HandlesStridedInput()
    {
        using var be = new CpuComputeBackend();
        // Transpose [2,3] → strided [3,2]; normalize over the last axis (size 2).
        var x = be.FromHost([1f, 2f, 3f, 4f, 5f, 6f], new Shape(2, 3), DType.F32).Transpose(0, 1); // rows [1,4],[2,5],[3,6]
        Assert.False(x.IsContiguous);
        var w = be.FromHost([1f, 1f], new Shape(2), DType.F32);

        var y = Host(be, be.RmsNorm(x, w, 1e-6f));

        float[][] rows = [[1, 4], [2, 5], [3, 6]];
        var expected = new float[6];
        for (int r = 0; r < 3; r++)
        {
            float inv = 1f / MathF.Sqrt((rows[r][0] * rows[r][0] + rows[r][1] * rows[r][1]) / 2f + 1e-6f);
            expected[r * 2] = rows[r][0] * inv;
            expected[r * 2 + 1] = rows[r][1] * inv;
        }
        AssertClose(expected, y, 1e-5f);
    }

    [Fact]
    public void Rope_BroadcastsCosSinOverLeadingAxes()
    {
        using var be = new CpuComputeBackend();
        // The real attention shape: x = [batch, heads, seq, headDim], cos/sin = [seq, headDim].
        const int batch = 2, heads = 1, seq = 2, d = 4;
        var rng = new Random(77);
        var xBuf = new float[batch * heads * seq * d];
        for (int i = 0; i < xBuf.Length; i++) xBuf[i] = (float)(rng.NextDouble() * 2 - 1);
        var x = be.FromHost(xBuf, new Shape(batch, heads, seq, d), DType.F32);

        float t0 = 0.5f, t1 = 1.1f;
        var cosBuf = new float[seq * d];
        var sinBuf = new float[seq * d];
        for (int p = 0; p < seq; p++)
        {
            float[] angles = [p * t0, p * t1, p * t0, p * t1];
            for (int j = 0; j < d; j++) { cosBuf[p * d + j] = MathF.Cos(angles[j]); sinBuf[p * d + j] = MathF.Sin(angles[j]); }
        }
        var cos = be.FromHost(cosBuf, new Shape(seq, d), DType.F32);
        var sin = be.FromHost(sinBuf, new Shape(seq, d), DType.F32);

        var y = Host(be, be.RotaryEmbedding(x, cos, sin));

        // Reference: apply rotate-half per (batch, head, seq) row using table row = seq index.
        int half = d / 2;
        var expected = new float[xBuf.Length];
        long rows = (long)batch * heads * seq;
        for (long r = 0; r < rows; r++)
        {
            int s = (int)(r % seq); // last leading axis before headDim is seq
            long b = r * d;
            for (int i = 0; i < d; i++)
            {
                float rot = i < half ? -xBuf[b + i + half] : xBuf[b + i - half];
                expected[b + i] = xBuf[b + i] * cosBuf[s * d + i] + rot * sinBuf[s * d + i];
            }
        }
        AssertClose(expected, y, 1e-5f);
    }

    [Fact]
    public void Rope_PreservesPairNorm_AndMatchesRotation()
    {
        using var be = new CpuComputeBackend();
        // headDim=4 (half=2): rotate-half pairs feature j with j+2. One position, angles θ0, θ1.
        float t0 = 0.7f, t1 = 1.3f;
        var x = be.FromHost([1f, 2f, 3f, 4f], new Shape(1, 4), DType.F32); // (a,b,c,e) = (1,2,3,4)
        var cos = be.FromHost([MathF.Cos(t0), MathF.Cos(t1), MathF.Cos(t0), MathF.Cos(t1)], new Shape(1, 4), DType.F32);
        var sin = be.FromHost([MathF.Sin(t0), MathF.Sin(t1), MathF.Sin(t0), MathF.Sin(t1)], new Shape(1, 4), DType.F32);

        var y = Host(be, be.RotaryEmbedding(x, cos, sin));
        // Pair (x0,x2) rotated by θ0; pair (x1,x3) by θ1.
        float y0 = 1 * MathF.Cos(t0) - 3 * MathF.Sin(t0);
        float y2 = 3 * MathF.Cos(t0) + 1 * MathF.Sin(t0);
        float y1 = 2 * MathF.Cos(t1) - 4 * MathF.Sin(t1);
        float y3 = 4 * MathF.Cos(t1) + 2 * MathF.Sin(t1);
        AssertClose([y0, y1, y2, y3], y, 1e-5f);
        // Norm preserved per pair.
        Assert.Equal(1 * 1 + 3 * 3, y0 * y0 + y2 * y2, 3);
        Assert.Equal(2 * 2 + 4 * 4, y1 * y1 + y3 * y3, 3);
    }
}
