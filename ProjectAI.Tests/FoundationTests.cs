using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using ProjectAI.Models;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>S0-7 (RNG), S0-8 (init), S0-9 (Module contract) foundation tests.</summary>
public class FoundationTests
{
    private static float[] Host(CpuComputeBackend be, Tensor t)
    {
        var buf = new float[t.ElementCount];
        be.ToHost(t, buf);
        return buf;
    }

    // ---- S0-7: RNG ----

    [Fact]
    public void Rng_SameSeed_ReproducesStream()
    {
        var a = new PcgRng(12345);
        var b = new PcgRng(12345);
        for (int i = 0; i < 100; i++) Assert.Equal(a.NextUInt32(), b.NextUInt32());
    }

    [Fact]
    public void Rng_DifferentSeeds_Diverge()
    {
        var a = new PcgRng(1);
        var b = new PcgRng(2);
        bool anyDifferent = false;
        for (int i = 0; i < 20; i++) anyDifferent |= a.NextUInt32() != b.NextUInt32();
        Assert.True(anyDifferent);
    }

    [Fact]
    public void Rng_MatchesCanonicalPcgVector()
    {
        // Pins the implementation to PCG-XSH-RR 64/32: pcg32_srandom_r(initstate=42, initseq=54).
        var rng = new PcgRng(seed: 42, stream: 54);
        uint[] expected = [0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e];
        foreach (var e in expected) Assert.Equal(e, rng.NextUInt32());
    }

    [Fact]
    public void Rng_Gaussian_HasUnitMeanAndStd()
    {
        var rng = new PcgRng(2024);
        const int n = 200_000;
        double sum = 0, sumSq = 0;
        for (int i = 0; i < n; i++) { float g = rng.NextGaussian(); sum += g; sumSq += (double)g * g; }
        double mean = sum / n;
        double std = Math.Sqrt(sumSq / n - mean * mean);
        Assert.True(Math.Abs(mean) < 0.02, $"mean {mean}");
        Assert.True(Math.Abs(std - 1.0) < 0.02, $"std {std}");
    }

    [Fact]
    public void Rng_Float_IsInUnitInterval()
    {
        var rng = new PcgRng(3);
        for (int i = 0; i < 1000; i++) { float f = rng.NextFloat(); Assert.InRange(f, 0f, 0.9999999f); }
    }

    // ---- S0-8: Init ----

    [Fact]
    public void Init_ZerosAndOnes()
    {
        using var be = new CpuComputeBackend();
        var z = be.Allocate(new Shape(5), DType.F32);
        Init.Zeros.Fill(z, be, new PcgRng(0));
        Assert.All(Host(be, z), v => Assert.Equal(0f, v));

        var o = be.Allocate(new Shape(5), DType.F32);
        Init.Ones.Fill(o, be, new PcgRng(0));
        Assert.All(Host(be, o), v => Assert.Equal(1f, v));
    }

    [Fact]
    public void Init_Normal_MatchesRequestedMeanAndStd_AndIsReproducible()
    {
        using var be = new CpuComputeBackend();
        var t1 = be.Allocate(new Shape(50_000), DType.F32);
        Init.Normal(0.3f, 0.5f).Fill(t1, be, new PcgRng(11));
        var h1 = Host(be, t1);
        double mean = h1.Average();
        double std = Math.Sqrt(h1.Select(v => ((double)v - mean) * (v - mean)).Average());
        Assert.True(Math.Abs(mean - 0.3) < 0.01, $"mean {mean}");
        Assert.True(Math.Abs(std - 0.5) < 0.01, $"std {std}");

        var t2 = be.Allocate(new Shape(50_000), DType.F32);
        Init.Normal(0.3f, 0.5f).Fill(t2, be, new PcgRng(11)); // same seed
        Assert.Equal(h1, Host(be, t2));
    }

    [Fact]
    public void Init_Kaiming_StdMatchesFormula()
    {
        using var be = new CpuComputeBackend();
        var w = be.Allocate(new Shape(64, 256), DType.F32); // fanIn = 256
        Init.Kaiming().Fill(w, be, new PcgRng(5));
        var h = Host(be, w);
        double std = Math.Sqrt(h.Select(v => (double)v * v).Average());
        double expected = 1.41421356 / Math.Sqrt(256); // gain sqrt(2) / sqrt(fanIn)
        Assert.True(Math.Abs(std - expected) < 0.01, $"std {std} vs {expected}");
    }

    // ---- S0-9: Module contract ----

    [Fact]
    public void Module_Param_AppearsInParametersAndNamedParameters()
    {
        using var be = new CpuComputeBackend();
        var lin = new Linear(ParameterContext.Create(be, 1), inDim: 3, outDim: 2);
        Assert.Single(lin.Parameters());
        var named = lin.NamedParameters().ToList();
        Assert.Equal("weight", named[0].Name);
        Assert.Equal(new Shape(2, 3), named[0].Param.Shape);
        Assert.True(named[0].Param.RequiresGrad);
    }

    [Fact]
    public void Module_NamedParameters_UseDottedChildPaths()
    {
        using var be = new CpuComputeBackend();
        var ffn = new SwiGluFeedForward(ParameterContext.Create(be, 1), dim: 3, hiddenDim: 4);
        var names = ffn.NamedParameters().Select(n => n.Name).OrderBy(s => s).ToArray();
        Assert.Equal(["down.weight", "gate.weight", "up.weight"], names);
        Assert.Equal(3, ffn.Parameters().Count());
    }

    [Fact]
    public void Module_SiblingParameters_AreDistinct()
    {
        using var be = new CpuComputeBackend();
        var ffn = new SwiGluFeedForward(ParameterContext.Create(be, 1), dim: 3, hiddenDim: 4);
        var named = ffn.NamedParameters().ToDictionary(n => n.Name, n => Host(be, n.Param));
        // gate and up have the same shape [4,3] but must be initialized to different values (sequential RNG).
        Assert.NotEqual(named["gate.weight"], named["up.weight"]);
    }

    [Fact]
    public void Module_Construction_IsReproducibleFromSeed()
    {
        using var be = new CpuComputeBackend();
        var a = new SwiGluFeedForward(ParameterContext.Create(be, 99), 3, 4);
        var b = new SwiGluFeedForward(ParameterContext.Create(be, 99), 3, 4);
        foreach (var (pa, pb) in a.Parameters().Zip(b.Parameters()))
            Assert.Equal(Host(be, pa), Host(be, pb));
    }

    [Fact]
    public void Module_ForwardConvenience_EqualsInferenceContext()
    {
        using var be = new CpuComputeBackend();
        var lin = new Linear(ParameterContext.Create(be, 5), 3, 2);
        var x = be.FromHost([1, 2, 3, 4, 5, 6], new Shape(2, 3), DType.F32);
        Assert.Equal(Host(be, lin.Forward(x)), Host(be, lin.Forward(x, ForwardContext.Inference())));
    }

    [Fact]
    public void Module_ParametersTrainWithAdamW()
    {
        using var be = new CpuComputeBackend();
        var ctx = ParameterContext.Create(be, 7);
        var ag = ctx.Ag;
        var lin = new Linear(ctx, 3, 1);

        // Fit a fixed linear target with MSE; loss must drop (validates Param tensors flow through AdamW).
        var rng = new Random(0);
        var xBuf = new float[16 * 3];
        var yBuf = new float[16];
        for (int i = 0; i < 16; i++)
        {
            float dot = 0;
            for (int j = 0; j < 3; j++) { float v = (float)(rng.NextDouble() * 2 - 1); xBuf[i * 3 + j] = v; dot += v * (j + 1); }
            yBuf[i] = dot;
        }
        var x = be.FromHost(xBuf, new Shape(16, 3), DType.F32);
        var y = be.FromHost(yBuf, new Shape(16, 1), DType.F32);
        var opt = new AdamW(lin.Parameters().ToList(), be, learningRate: 0.1f, weightDecay: 0f);

        var lossHost = new float[1];
        float first = float.NaN, last = float.NaN;
        for (int step = 0; step < 100; step++)
        {
            opt.ZeroGrad();
            var diff = ag.Sub(lin.Forward(x), y);
            var loss = ag.Mean(ag.Mul(diff, diff));
            loss.Backward();
            opt.Step();
            be.ToHost(loss, lossHost);
            if (step == 0) first = lossHost[0];
            last = lossHost[0];
        }
        Assert.True(last < first * 0.1f, $"loss should drop substantially: {first} → {last}");
    }
}
