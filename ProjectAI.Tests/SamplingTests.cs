using ProjectAI.Core;
using ProjectAI.Models;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>Unit tests for S1-9 samplers (greedy / temperature / top-k / top-p, seedable for reproducibility).</summary>
public class SamplingTests
{
    [Fact]
    public void Greedy_ReturnsArgmax()
    {
        var s = new GreedySampler();
        Assert.Equal(1, s.Sample([0.1f, 0.5f, 0.3f, -2f]));
        Assert.Equal(3, s.Sample([0.1f, 0.5f, 0.3f, 9f]));
    }

    [Fact]
    public void Greedy_BreaksTiesToLowestIndex()
    {
        var s = new GreedySampler();
        Assert.Equal(0, s.Sample([0.5f, 0.5f, 0.1f]));
    }

    [Fact]
    public void Temperature_Zero_IsGreedy()
    {
        var s = new TopKTopPSampler(new PcgRng(1), temperature: 0f);
        for (int trial = 0; trial < 20; trial++)
            Assert.Equal(2, s.Sample([1f, 2f, 5f, 0f])); // always the argmax, no randomness
    }

    [Fact]
    public void TopK_RestrictsSupportToTopKIndices()
    {
        // Logits 0..4 = [1,5,3,2,4]; top-2 by value are indices 1 (5) and 4 (4).
        var s = new TopKTopPSampler(new PcgRng(42), temperature: 1f, topK: 2);
        var allowed = new HashSet<int> { 1, 4 };
        for (int i = 0; i < 500; i++)
            Assert.Contains(s.Sample([1f, 5f, 3f, 2f, 4f]), allowed);
    }

    [Fact]
    public void TopP_RestrictsSupportToNucleus()
    {
        // softmax([10,9,0,0,0]) ≈ [0.731, 0.269, ~0, ~0, ~0]; nucleus for p=0.9 is {0,1} (0.731 then 1.0 ≥ 0.9).
        var s = new TopKTopPSampler(new PcgRng(7), temperature: 1f, topP: 0.9f);
        var nucleus = new HashSet<int> { 0, 1 };
        for (int i = 0; i < 500; i++)
            Assert.Contains(s.Sample([10f, 9f, 0f, 0f, 0f]), nucleus);
    }

    [Fact]
    public void TopP_KeepsAtLeastOneToken_WhenTopProbExceedsP()
    {
        // Top token already exceeds p — the nucleus must still contain (only) it, never empty.
        var s = new TopKTopPSampler(new PcgRng(3), temperature: 1f, topP: 0.3f);
        for (int i = 0; i < 100; i++)
            Assert.Equal(0, s.Sample([12f, 0f, 0f, 0f])); // index 0 has prob ≈ 1 > 0.3
    }

    [Fact]
    public void FixedSeed_ReproducesSequence()
    {
        float[] logits = [0.2f, 1.1f, 0.7f, -0.5f, 2.0f, 0.3f];
        var a = new TopKTopPSampler(new PcgRng(123), temperature: 1f);
        var b = new TopKTopPSampler(new PcgRng(123), temperature: 1f);
        for (int i = 0; i < 64; i++)
            Assert.Equal(a.Sample(logits), b.Sample(logits)); // same seed → identical draws

        // A different seed should not lock-step with the first (sanity that draws actually depend on the RNG).
        var c = new TopKTopPSampler(new PcgRng(999), temperature: 1f);
        var aSeq = new int[64];
        var cSeq = new int[64];
        var a2 = new TopKTopPSampler(new PcgRng(123), temperature: 1f);
        for (int i = 0; i < 64; i++) { aSeq[i] = a2.Sample(logits); cSeq[i] = c.Sample(logits); }
        Assert.NotEqual(aSeq, cSeq);
    }

    [Fact]
    public void Temperature_PeakedLogits_SampleArgmaxMostOften()
    {
        // With a clear peak at index 1 and T=1, index 1 should dominate the empirical distribution.
        var s = new TopKTopPSampler(new PcgRng(5), temperature: 1f);
        var counts = new int[3];
        for (int i = 0; i < 3000; i++) counts[s.Sample([0f, 3f, 0f])]++;
        Assert.True(counts[1] > counts[0] && counts[1] > counts[2], $"peak should dominate: {counts[0]},{counts[1]},{counts[2]}");
    }

    [Fact]
    public void LowTemperature_SharpensTowardArgmax()
    {
        // Cooler temperature concentrates mass on the top logit relative to T=1.
        float[] logits = [0f, 1f, 2f];
        var hot = new TopKTopPSampler(new PcgRng(9), temperature: 1.0f);
        var cold = new TopKTopPSampler(new PcgRng(9), temperature: 0.2f);
        int hotPeak = 0, coldPeak = 0;
        for (int i = 0; i < 2000; i++)
        {
            if (hot.Sample(logits) == 2) hotPeak++;
            if (cold.Sample(logits) == 2) coldPeak++;
        }
        Assert.True(coldPeak > hotPeak, $"cold should pick the peak more often: cold {coldPeak} vs hot {hotPeak}");
    }

    [Fact]
    public void Sampler_NaNLogit_FallsBackToFiniteArgmax()
    {
        // A NaN must not poison the distribution: fall back to the finite argmax (index 0 here), every draw.
        var s = new TopKTopPSampler(new PcgRng(1), temperature: 1f);
        for (int i = 0; i < 50; i++)
            Assert.Equal(0, s.Sample([5f, float.NaN, 1f]));
    }

    [Fact]
    public void Sampler_PositiveInfLogit_SelectsThatToken()
    {
        // +inf means "certainly this token"; the fallback argmax picks it rather than returning NaN garbage.
        var s = new TopKTopPSampler(new PcgRng(2), temperature: 1f);
        Assert.Equal(1, s.Sample([2f, float.PositiveInfinity, 1f]));
    }

    [Fact]
    public void Greedy_IgnoresNaN_AndReturnsFiniteMax()
    {
        Assert.Equal(1, new GreedySampler().Sample([float.NaN, 5f, 1f]));
    }

    [Fact]
    public void Samplers_RejectEmptyLogits()
    {
        Assert.Throws<ArgumentException>(() => new GreedySampler().Sample([]));
        Assert.Throws<ArgumentException>(() => new TopKTopPSampler(new PcgRng(1)).Sample([]));
    }
}
