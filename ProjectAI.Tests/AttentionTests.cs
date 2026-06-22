using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using ProjectAI.Models;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>Forward/structural tests for S1-7 GQA attention (full-sequence path).</summary>
public class AttentionTests
{
    private static float[] Host(CpuComputeBackend be, Tensor t)
    {
        var buf = new float[t.ElementCount];
        be.ToHost(t, buf);
        return buf;
    }

    private static float[] Rand(Random rng, int count)
    {
        var v = new float[count];
        for (int i = 0; i < count; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    private static ModelConfig Config(int heads, int kvHeads, int dModel = 8) => new()
    {
        VocabSize = 16, EmbeddingDim = dModel, LayerCount = 1, HeadCount = heads, KvHeadCount = kvHeads,
        FeedForwardHiddenDim = 4 * dModel, MaxSequenceLength = 16,
    };

    [Fact]
    public void Attention_ProducesInputShape()
    {
        using var be = new CpuComputeBackend();
        var attn = new Attention(ParameterContext.Create(be, 1), Config(4, 2));
        var x = be.FromHost(Rand(new Random(1), 2 * 3 * 8), new Shape(2, 3, 8), DType.F32);
        Assert.Equal(new Shape(2, 3, 8), attn.Forward(x).Shape);
    }

    [Fact]
    public void Attention_IsCausal()
    {
        using var be = new CpuComputeBackend();
        const int b = 2, s = 4, d = 8;
        var attn = new Attention(ParameterContext.Create(be, 2), Config(4, 2, d));

        var x1 = Rand(new Random(3), b * s * d);
        var x2 = (float[])x1.Clone();
        for (int bi = 0; bi < b; bi++) // perturb only the LAST sequence position
            for (int k = 0; k < d; k++) x2[(bi * s + (s - 1)) * d + k] += 0.7f;

        var y1 = Host(be, attn.Forward(be.FromHost(x1, new Shape(b, s, d), DType.F32)));
        var y2 = Host(be, attn.Forward(be.FromHost(x2, new Shape(b, s, d), DType.F32)));

        // Causality: changing the last token must not affect outputs at earlier positions.
        for (int bi = 0; bi < b; bi++)
            for (int pos = 0; pos < s - 1; pos++)
                for (int k = 0; k < d; k++)
                    Assert.Equal(y1[(bi * s + pos) * d + k], y2[(bi * s + pos) * d + k], 4);
        // And the last position DID change (sanity that the perturbation mattered).
        Assert.NotEqual(y1[((s - 1)) * d], y2[((s - 1)) * d], 4);
    }

    [Theory]
    [InlineData(4, 4)] // multi-head (KvH == H, G = 1)
    [InlineData(4, 2)] // grouped-query (G = 2)
    [InlineData(4, 1)] // multi-query (single shared KV head, G = H)
    public void Attention_GqaConfigurations_Run(int heads, int kvHeads)
    {
        using var be = new CpuComputeBackend();
        var attn = new Attention(ParameterContext.Create(be, 5), Config(heads, kvHeads));
        var x = be.FromHost(Rand(new Random(6), 2 * 3 * 8), new Shape(2, 3, 8), DType.F32);
        var y = attn.Forward(x);
        Assert.Equal(new Shape(2, 3, 8), y.Shape);
        Assert.All(Host(be, y), v => Assert.False(float.IsNaN(v) || float.IsInfinity(v)));
    }

    [Theory]
    [InlineData(2, 2, 4)] // multi-head (G = 1)
    [InlineData(2, 1, 4)] // multi-query (G = 2)
    [InlineData(4, 2, 8)] // grouped-query (KvH = 2, G = 2) — the intermediate grouping
    [InlineData(4, 1, 8)] // multi-query (G = 4)
    public void Attention_MatchesHandRolledReference(int h, int kvh, int d)
    {
        using var be = new CpuComputeBackend();
        const int s = 3;
        int dh = d / h, g = h / kvh, hdh = h * dh;
        const float theta = 10000f;

        var attn = new Attention(ParameterContext.Create(be, 9), Config(h, kvh, d));
        var w = attn.NamedParameters().ToDictionary(n => n.Name, n => Host(be, n.Param));
        float[] wq = w["wq.weight"], wk = w["wk.weight"], wv = w["wv.weight"], wo = w["wo.weight"];

        var x = Rand(new Random(10), s * d);
        var got = Host(be, attn.Forward(be.FromHost(x, new Shape(1, s, d), DType.F32)));

        // Hand-rolled reference: per head, causal scaled-dot-product attention with rotate-half RoPE.
        var merged = new float[s * hdh];
        for (int head = 0; head < h; head++)
        {
            int kvHead = head / g;
            var q = new float[s][]; var k = new float[s][]; var v = new float[s][];
            for (int i = 0; i < s; i++)
            {
                q[i] = Rope(Proj(x, i, d, wq, head * dh, dh), i, dh, theta);
                k[i] = Rope(Proj(x, i, d, wk, kvHead * dh, dh), i, dh, theta);
                v[i] = Proj(x, i, d, wv, kvHead * dh, dh);
            }
            for (int i = 0; i < s; i++)
            {
                var sc = new float[i + 1];
                for (int j = 0; j <= i; j++)
                {
                    float dot = 0;
                    for (int c = 0; c < dh; c++) dot += q[i][c] * k[j][c];
                    sc[j] = dot / MathF.Sqrt(dh);
                }
                float mx = float.NegativeInfinity;
                foreach (var z in sc) mx = MathF.Max(mx, z);
                float sum = 0;
                for (int j = 0; j <= i; j++) { sc[j] = MathF.Exp(sc[j] - mx); sum += sc[j]; }
                for (int c = 0; c < dh; c++)
                {
                    float acc = 0;
                    for (int j = 0; j <= i; j++) acc += (sc[j] / sum) * v[j][c];
                    merged[i * hdh + head * dh + c] = acc;
                }
            }
        }
        var expected = new float[s * d];
        for (int i = 0; i < s; i++)
            for (int o = 0; o < d; o++)
            {
                float acc = 0;
                for (int m = 0; m < hdh; m++) acc += merged[i * hdh + m] * wo[o * hdh + m];
                expected[i * d + o] = acc;
            }

        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - got[i]) <= 1e-4f, $"index {i}: expected {expected[i]}, got {got[i]}");
    }

    private static float[] Proj(float[] x, int pos, int d, float[] w, int rowOffset, int dh)
    {
        var o = new float[dh];
        for (int c = 0; c < dh; c++)
        {
            float acc = 0;
            for (int dd = 0; dd < d; dd++) acc += x[pos * d + dd] * w[(rowOffset + c) * d + dd];
            o[c] = acc;
        }
        return o;
    }

    private static float[] Rope(float[] vec, int pos, int dh, float theta)
    {
        int half = dh / 2;
        var y = new float[dh];
        for (int j = 0; j < half; j++)
        {
            float ang = pos * (1f / MathF.Pow(theta, (2f * j) / dh));
            float c = MathF.Cos(ang), sn = MathF.Sin(ang);
            y[j] = vec[j] * c - vec[j + half] * sn;
            y[j + half] = vec[j + half] * c + vec[j] * sn;
        }
        return y;
    }

    [Fact]
    public void Attention_SingleToken_Runs()
    {
        using var be = new CpuComputeBackend();
        var attn = new Attention(ParameterContext.Create(be, 1), Config(4, 2));
        var x = be.FromHost(Rand(new Random(2), 1 * 1 * 8), new Shape(1, 1, 8), DType.F32); // S = 1
        var y = attn.Forward(x);
        Assert.Equal(new Shape(1, 1, 8), y.Shape);
        Assert.All(Host(be, y), v => Assert.False(float.IsNaN(v)));
    }

    [Theory]
    [InlineData(2, 4)] // KvH > H
    [InlineData(4, 3)] // H not divisible by KvH
    [InlineData(4, 0)] // KvH < 1
    public void Attention_RejectsInvalidConfig(int h, int kvh)
    {
        using var be = new CpuComputeBackend();
        Assert.Throws<ArgumentException>(() => new Attention(ParameterContext.Create(be, 1), Config(h, kvh)));
    }

    [Fact]
    public void Attention_PerBatchMask_AlignsToBatch_NotGroup()
    {
        using var be = new CpuComputeBackend();
        // H=2, KvH=1 → group G=2; batch=2 makes batch == group, the silent-misbroadcast regime.
        var attn = new Attention(ParameterContext.Create(be, 4), Config(2, 1));
        const int b = 2, s = 3, d = 8;
        var x = be.FromHost(Rand(new Random(7), b * s * d), new Shape(b, s, d), DType.F32);

        var yDefault = Host(be, attn.Forward(x)); // built-in rank-2 causal mask

        // The same causal mask supplied explicitly per batch ([b,s,s]) must give an identical result.
        var maskBuf = new float[b * s * s];
        for (int bi = 0; bi < b; bi++)
            for (int i = 0; i < s; i++)
                for (int j = 0; j < s; j++)
                    maskBuf[(bi * s + i) * s + j] = j <= i ? 0f : -1e9f;
        var mask = be.FromHost(maskBuf, new Shape(b, s, s), DType.F32);
        var yMasked = Host(be, attn.Forward(x, ForwardContext.Inference() with { AttentionMask = mask }));

        for (int i = 0; i < yDefault.Length; i++)
            Assert.Equal(yDefault[i], yMasked[i], 4);
    }

    [Fact]
    public void Attention_TrainContext_WithIdentityPositions_EqualsInference()
    {
        using var be = new CpuComputeBackend();
        var attn = new Attention(ParameterContext.Create(be, 4), Config(4, 2));
        const int s = 3, d = 8;
        var x = be.FromHost(Rand(new Random(8), 1 * s * d), new Shape(1, s, d), DType.F32);

        var causal = new float[s * s];
        for (int i = 0; i < s; i++)
            for (int j = 0; j < s; j++)
                causal[i * s + j] = j <= i ? 0f : -1e9f;
        var ctx = ForwardContext.Train(
            be.FromHost(causal, new Shape(s, s), DType.F32),
            be.FromHost([0f, 1f, 2f], new Shape(s), DType.F32)); // identity positions

        var yInfer = Host(be, attn.Forward(x));
        var yTrain = Host(be, attn.Forward(x, ctx)); // must not throw (RoPE gates on Cache, not Positions)
        for (int i = 0; i < yInfer.Length; i++) Assert.Equal(yInfer[i], yTrain[i], 4);
    }

    [Fact]
    public void Attention_RejectsKvCacheDecode_UntilFollowUp()
    {
        using var be = new CpuComputeBackend();
        var ctx = ParameterContext.Create(be, 1);
        var attn = new Attention(ctx, Config(4, 2));
        var x = be.FromHost(Rand(new Random(1), 1 * 1 * 8), new Shape(1, 1, 8), DType.F32);
        var cache = new KvCache(Config(4, 2), maxBatch: 1, maxSequenceLength: 16);
        Assert.Throws<NotImplementedException>(() => attn.Forward(x, ForwardContext.Decode(cache, x)));
    }
}
