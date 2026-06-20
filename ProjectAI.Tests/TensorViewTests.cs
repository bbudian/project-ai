using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using Xunit;

namespace ProjectAI.Tests;

public class TensorViewTests
{
    private static float[] Host(CpuComputeBackend backend, Tensor t)
    {
        var buffer = new float[t.ElementCount];
        backend.ToHost(t, buffer);
        return buffer;
    }

    [Fact]
    public void NewTensor_IsContiguousRowMajor()
    {
        using var be = new CpuComputeBackend();
        var t = be.FromHost([0f, 1f, 2f, 3f, 4f, 5f], new Shape(2, 3), DType.F32);
        Assert.True(t.IsContiguous);
        Assert.Equal(new long[] { 3, 1 }, t.Strides);
        Assert.Equal(0L, t.Offset);
    }

    [Fact]
    public void Transpose_IsNonContiguous_AndRoundTripsThroughToHost()
    {
        using var be = new CpuComputeBackend();
        var t = be.FromHost([0f, 1f, 2f, 3f, 4f, 5f], new Shape(2, 3), DType.F32);

        var tt = t.Transpose(0, 1);

        Assert.Equal(new Shape(3, 2), tt.Shape);
        Assert.False(tt.IsContiguous);
        Assert.Equal(new float[] { 0, 3, 1, 4, 2, 5 }, Host(be, tt));
    }

    [Fact]
    public void Slice_AlongInnerAxis_GathersStridedRows()
    {
        using var be = new CpuComputeBackend();
        var t = be.FromHost([0f, 1f, 2f, 3f, 4f, 5f], new Shape(2, 3), DType.F32);

        var s = t.Slice(axis: 1, start: 1, length: 2);

        Assert.Equal(new Shape(2, 2), s.Shape);
        Assert.False(s.IsContiguous);
        Assert.Equal(new float[] { 1, 2, 4, 5 }, Host(be, s));
    }

    [Fact]
    public void Permute_ReordersAxes_AndGathersCorrectly()
    {
        using var be = new CpuComputeBackend();
        var src = new float[24];
        for (int i = 0; i < 24; i++) src[i] = i;
        var t = be.FromHost(src, new Shape(2, 3, 4), DType.F32);

        var p = t.Permute(2, 0, 1);

        Assert.Equal(new Shape(4, 2, 3), p.Shape);
        Assert.Equal(new long[] { 1, 12, 4 }, p.Strides);
        Assert.Equal(
            new float[] { 0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23 },
            Host(be, p));
    }

    [Fact]
    public void Reshape_Contiguous_PreservesData_AndRejectsNonContiguous()
    {
        using var be = new CpuComputeBackend();
        var t = be.FromHost([0f, 1f, 2f, 3f, 4f, 5f], new Shape(2, 3), DType.F32);

        var r = t.Reshape(6);
        Assert.Equal(new Shape(6), r.Shape);
        Assert.True(r.IsContiguous);
        Assert.Equal(new float[] { 0, 1, 2, 3, 4, 5 }, Host(be, r));

        Assert.Throws<InvalidOperationException>(() => t.Transpose(0, 1).Reshape(6));
    }
}
