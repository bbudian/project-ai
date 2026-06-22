using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using Xunit;

namespace ProjectAI.Tests;

public class ElementwiseBroadcastTests
{
    private static float[] Host(CpuComputeBackend be, Tensor t)
    {
        var buf = new float[t.ElementCount];
        be.ToHost(t, buf);
        return buf;
    }

    [Fact]
    public void Add_BroadcastsRowVectorAcrossRows()
    {
        using var be = new CpuComputeBackend();
        var a = be.FromHost([1, 2, 3, 4, 5, 6], new Shape(2, 3), DType.F32); // [[1,2,3],[4,5,6]]
        var b = be.FromHost([10, 20, 30], new Shape(3), DType.F32);

        var c = be.Add(a, b);

        Assert.Equal(new Shape(2, 3), c.Shape);
        Assert.Equal(new float[] { 11, 22, 33, 14, 25, 36 }, Host(be, c));
    }

    [Fact]
    public void Add_BidirectionalBroadcast_OuterSum()
    {
        using var be = new CpuComputeBackend();
        var a = be.FromHost([1, 2], new Shape(2, 1), DType.F32);  // column
        var b = be.FromHost([10, 20, 30], new Shape(1, 3), DType.F32); // row

        var c = be.Add(a, b);

        Assert.Equal(new Shape(2, 3), c.Shape);
        Assert.Equal(new float[] { 11, 21, 31, 12, 22, 32 }, Host(be, c));
    }

    [Fact]
    public void Mul_BroadcastsColumnVector()
    {
        using var be = new CpuComputeBackend();
        var a = be.FromHost([1, 2, 3, 4, 5, 6], new Shape(2, 3), DType.F32);
        var b = be.FromHost([2, 3], new Shape(2, 1), DType.F32);

        var c = be.Mul(a, b);

        Assert.Equal(new float[] { 2, 4, 6, 12, 15, 18 }, Host(be, c));
    }

    [Fact]
    public void Add_EqualShapes_FastPath_StillCorrect()
    {
        using var be = new CpuComputeBackend();
        var a = be.FromHost([1, 2, 3], new Shape(3), DType.F32);
        var b = be.FromHost([10, 20, 30], new Shape(3), DType.F32);
        Assert.Equal(new float[] { 11, 22, 33 }, Host(be, be.Add(a, b)));
    }

    [Fact]
    public void Add_HandlesNonContiguousOperand()
    {
        using var be = new CpuComputeBackend();
        // at is [3,2] = [[1,4],[2,5],[3,6]] (non-contiguous transpose of [2,3]).
        var at = be.FromHost([1, 2, 3, 4, 5, 6], new Shape(2, 3), DType.F32).Transpose(0, 1);
        Assert.False(at.IsContiguous);
        var b = be.FromHost([10, 20, 30, 40, 50, 60], new Shape(3, 2), DType.F32);

        var c = be.Add(at, b);

        Assert.Equal(new Shape(3, 2), c.Shape);
        Assert.Equal(new float[] { 11, 24, 32, 45, 53, 66 }, Host(be, c));
    }

    [Fact]
    public void AddScalar_And_MulScalar_HandleNonContiguous()
    {
        using var be = new CpuComputeBackend();
        var at = be.FromHost([1, 2, 3, 4, 5, 6], new Shape(2, 3), DType.F32).Transpose(0, 1); // [3,2]: 1,4,2,5,3,6

        Assert.Equal(new float[] { 2, 5, 3, 6, 4, 7 }, Host(be, be.AddScalar(at, 1f)));
        Assert.Equal(new float[] { 10, 40, 20, 50, 30, 60 }, Host(be, be.MulScalar(at, 10f)));
    }

    [Fact]
    public void Add_IncompatibleShapes_Throws()
    {
        using var be = new CpuComputeBackend();
        var a = be.FromHost([1, 2, 3], new Shape(3), DType.F32);
        var b = be.FromHost([1, 2, 3, 4], new Shape(4), DType.F32);
        Assert.Throws<ArgumentException>(() => be.Add(a, b));
    }

    [Fact]
    public void Mul_HandlesNonContiguousOperand()
    {
        using var be = new CpuComputeBackend();
        var at = be.FromHost([1, 2, 3, 4, 5, 6], new Shape(2, 3), DType.F32).Transpose(0, 1); // [3,2]: 1,4,2,5,3,6
        var b = be.FromHost([10, 10, 10, 10, 10, 10], new Shape(3, 2), DType.F32);
        Assert.Equal(new float[] { 10, 40, 20, 50, 30, 60 }, Host(be, be.Mul(at, b)));
    }

    // ---- Regression: a contiguous, zero-offset head-slice is shorter than its backing buffer ----

    [Fact]
    public void HeadSlice_1D_ThroughScalarAndElementwiseOps()
    {
        using var be = new CpuComputeBackend();
        var head = be.FromHost([1, 2, 3, 4], new Shape(4), DType.F32).Slice(0, 0, 2); // [1,2]
        Assert.True(head.IsContiguous);
        Assert.Equal(0L, head.Offset);
        Assert.Equal(2L, head.ElementCount);

        Assert.Equal(new float[] { 11, 12 }, Host(be, be.AddScalar(head, 10f)));
        Assert.Equal(new float[] { 10, 20 }, Host(be, be.MulScalar(head, 10f)));
        // Equal-shape elementwise: must fall through the fast path to the gather path.
        Assert.Equal(new float[] { 101, 202 }, Host(be, be.Add(head, be.FromHost([100, 200], new Shape(2), DType.F32))));
        Assert.Equal(new float[] { 10, 20 }, Host(be, be.Mul(head, be.FromHost([10, 10], new Shape(2), DType.F32))));
        Assert.Equal(new float[] { 2, 4 }, Host(be, be.Add(head, head)));
    }

    [Fact]
    public void HeadSlice_2D_FirstRows_ThroughElementwise()
    {
        using var be = new CpuComputeBackend();
        var rows = be.FromHost([1, 2, 3, 4, 5, 6], new Shape(3, 2), DType.F32).Slice(0, 0, 2); // [[1,2],[3,4]]
        Assert.True(rows.IsContiguous);
        Assert.Equal(new Shape(2, 2), rows.Shape);

        Assert.Equal(new float[] { 2, 3, 4, 5 }, Host(be, be.AddScalar(rows, 1f)));
        var b = be.FromHost([10, 20, 30, 40], new Shape(2, 2), DType.F32);
        Assert.Equal(new float[] { 11, 22, 33, 44 }, Host(be, be.Add(rows, b)));
    }
}
