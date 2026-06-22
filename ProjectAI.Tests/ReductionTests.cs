using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using Xunit;

namespace ProjectAI.Tests;

public class ReductionTests
{
    private static float[] Host(CpuComputeBackend be, Tensor t)
    {
        var buf = new float[t.ElementCount];
        be.ToHost(t, buf);
        return buf;
    }

    // x = [[1,2,3],[4,5,6]]
    private static Tensor Sample(CpuComputeBackend be) =>
        be.FromHost([1, 2, 3, 4, 5, 6], new Shape(2, 3), DType.F32);

    [Fact]
    public void Sum_OverAxis0_AddsRows()
    {
        using var be = new CpuComputeBackend();
        var s = be.Sum(Sample(be), axis: 0);
        Assert.Equal(new Shape(3), s.Shape);
        Assert.Equal(new float[] { 5, 7, 9 }, Host(be, s));
    }

    [Fact]
    public void Sum_OverAxis1_AddsColumns()
    {
        using var be = new CpuComputeBackend();
        var s = be.Sum(Sample(be), axis: 1);
        Assert.Equal(new Shape(2), s.Shape);
        Assert.Equal(new float[] { 6, 15 }, Host(be, s));
    }

    [Fact]
    public void Mean_OverAxis1()
    {
        using var be = new CpuComputeBackend();
        var s = be.Mean(Sample(be), axis: 1);
        Assert.Equal(new float[] { 2, 5 }, Host(be, s));
    }

    [Fact]
    public void Max_OverAxis0_And_Axis1()
    {
        using var be = new CpuComputeBackend();
        Assert.Equal(new float[] { 4, 5, 6 }, Host(be, be.Max(Sample(be), axis: 0)));
        Assert.Equal(new float[] { 3, 6 }, Host(be, be.Max(Sample(be), axis: 1)));
    }

    [Fact]
    public void Sum_KeepDims_RetainsReducedAxisAsSize1()
    {
        using var be = new CpuComputeBackend();
        var s = be.Sum(Sample(be), axis: 1, keepDims: true);
        Assert.Equal(new Shape(2, 1), s.Shape);
        Assert.Equal(new float[] { 6, 15 }, Host(be, s));
    }

    [Fact]
    public void NegativeAxis_CountsFromEnd()
    {
        using var be = new CpuComputeBackend();
        Assert.Equal(Host(be, be.Sum(Sample(be), axis: 1)), Host(be, be.Sum(Sample(be), axis: -1)));
    }

    [Fact]
    public void Reduce_HandlesNonContiguousInput()
    {
        using var be = new CpuComputeBackend();
        // Transpose to [3,2] = [[1,4],[2,5],[3,6]] (non-contiguous), then sum over axis 1.
        var xt = Sample(be).Transpose(0, 1);
        Assert.False(xt.IsContiguous);

        var s = be.Sum(xt, axis: 1);
        Assert.Equal(new Shape(3), s.Shape);
        Assert.Equal(new float[] { 5, 7, 9 }, Host(be, s));
    }

    [Fact]
    public void SumOverSingleAxisVector_ProducesScalar()
    {
        using var be = new CpuComputeBackend();
        var v = be.FromHost([1, 2, 3, 4], new Shape(4), DType.F32);
        var s = be.Sum(v, axis: 0);
        Assert.Equal(0, s.Shape.Rank);
        Assert.Equal(new float[] { 10 }, Host(be, s));
    }

    // ---- Rank-3 / strided coverage against an independent reference reducer ----

    private enum Kind { Sum, Mean, Max }

    /// <summary>Independent reference reduction: gathers via row-major unravel, not the kernel's decomposition.</summary>
    private static double[] RefReduce(float[] src, int[] dims, int axis, Kind kind)
    {
        int rank = dims.Length;
        int total = 1;
        foreach (var d in dims) total *= d;
        int outCount = 1;
        for (int i = 0; i < rank; i++) if (i != axis) outCount *= dims[i];

        var sum = new double[outCount];
        var max = new double[outCount];
        Array.Fill(max, double.NegativeInfinity);

        var idx = new int[rank];
        for (int lin = 0; lin < total; lin++)
        {
            long outLin = 0;
            for (int i = 0; i < rank; i++) if (i != axis) outLin = outLin * dims[i] + idx[i];
            double v = src[lin];
            sum[outLin] += v;
            if (v > max[outLin]) max[outLin] = v;
            for (int a = rank - 1; a >= 0; a--) { if (++idx[a] < dims[a]) break; idx[a] = 0; }
        }

        var o = new double[outCount];
        for (int i = 0; i < outCount; i++)
            o[i] = kind == Kind.Sum ? sum[i] : kind == Kind.Mean ? sum[i] / dims[axis] : max[i];
        return o;
    }

    private static void AssertClose(double[] expected, float[] actual, double rtol = 1e-4, double atol = 1e-5)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= atol + rtol * Math.Abs(expected[i]),
                $"index {i}: expected {expected[i]}, got {actual[i]}");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)] // the middle axis — the case the [outer, axisLen, inner] decomposition exists for
    [InlineData(2)]
    public void Reduce3D_MatchesReference_AllAxes(int axis)
    {
        using var be = new CpuComputeBackend();
        int[] dims = [2, 3, 4];
        var raw = new float[24];
        for (int i = 0; i < 24; i++) raw[i] = i;
        var x = be.FromHost(raw, new Shape(dims), DType.F32);

        AssertClose(RefReduce(raw, dims, axis, Kind.Sum), Host(be, be.Sum(x, axis)));
        AssertClose(RefReduce(raw, dims, axis, Kind.Mean), Host(be, be.Mean(x, axis)));
        AssertClose(RefReduce(raw, dims, axis, Kind.Max), Host(be, be.Max(x, axis)));
    }

    [Fact]
    public void Reduce3D_Random_MatchesReference()
    {
        using var be = new CpuComputeBackend();
        int[] dims = [5, 7, 4];
        var rng = new Random(2024);
        var raw = new float[5 * 7 * 4];
        for (int i = 0; i < raw.Length; i++) raw[i] = (float)(rng.NextDouble() * 10 - 5);
        var x = be.FromHost(raw, new Shape(dims), DType.F32);

        for (int axis = 0; axis < 3; axis++)
        {
            AssertClose(RefReduce(raw, dims, axis, Kind.Sum), Host(be, be.Sum(x, axis)));
            AssertClose(RefReduce(raw, dims, axis, Kind.Max), Host(be, be.Max(x, axis)));
        }
    }

    [Fact]
    public void Reduce_StridedMeanAndMax_MatchReference()
    {
        using var be = new CpuComputeBackend();
        var raw = new float[24];
        for (int i = 0; i < 24; i++) raw[i] = i;
        // Permute [2,3,4] → strided [4,3,2], then reduce over the (now) middle axis.
        var x = be.FromHost(raw, new Shape(2, 3, 4), DType.F32).Permute(2, 1, 0);
        Assert.False(x.IsContiguous);
        Assert.Equal(new Shape(4, 3, 2), x.Shape);

        int[] dims = [4, 3, 2];
        var mat = Host(be, x); // logical [4,3,2] row-major
        AssertClose(RefReduce(mat, dims, 1, Kind.Mean), Host(be, be.Mean(x, 1)));
        AssertClose(RefReduce(mat, dims, 1, Kind.Max), Host(be, be.Max(x, 1)));
    }

    [Fact]
    public void Sum_KeepDims_On3D_RetainsRank()
    {
        using var be = new CpuComputeBackend();
        var raw = new float[24];
        for (int i = 0; i < 24; i++) raw[i] = i;
        var x = be.FromHost(raw, new Shape(2, 3, 4), DType.F32);

        var s = be.Sum(x, axis: 1, keepDims: true);
        Assert.Equal(new Shape(2, 1, 4), s.Shape);
        AssertClose(RefReduce(raw, [2, 3, 4], 1, Kind.Sum), Host(be, s));
    }
}
