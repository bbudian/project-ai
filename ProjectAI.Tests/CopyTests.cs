using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using Xunit;

namespace ProjectAI.Tests;

public class CopyTests
{
    private static float[] Host(CpuComputeBackend be, Tensor t)
    {
        var buf = new float[t.ElementCount];
        be.ToHost(t, buf);
        return buf;
    }

    [Fact]
    public void Copy_StridedSource_GathersIntoContiguousDestination()
    {
        using var be = new CpuComputeBackend();
        var src = be.FromHost([1, 2, 3, 4, 5, 6], new Shape(2, 3), DType.F32).Transpose(0, 1); // [3,2]: 1,4,2,5,3,6
        var dst = be.Allocate(new Shape(3, 2), DType.F32);

        be.Copy(src, dst);

        Assert.Equal(new float[] { 1, 4, 2, 5, 3, 6 }, Host(be, dst));
    }

    [Fact]
    public void Copy_IntoStridedDestination_ScattersCorrectly()
    {
        using var be = new CpuComputeBackend();
        var src = be.FromHost([1, 2, 3, 4, 5, 6], new Shape(3, 2), DType.F32); // [[1,2],[3,4],[5,6]]
        var dstBase = be.Allocate(new Shape(2, 3), DType.F32);
        var dst = dstBase.Transpose(0, 1); // [3,2] strided view sharing dstBase's buffer

        be.Copy(src, dst);

        // Logical read-back of the strided destination matches the source.
        Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6 }, Host(be, dst));
        // And the underlying buffer holds the transposed (scattered) layout.
        Assert.Equal(new float[] { 1, 3, 5, 2, 4, 6 }, Host(be, dstBase));
    }

    [Fact]
    public void Copy_HeadSliceSource()
    {
        using var be = new CpuComputeBackend();
        var head = be.FromHost([1, 2, 3, 4], new Shape(4), DType.F32).Slice(0, 0, 2); // contiguous, shorter than buffer
        var dst = be.Allocate(new Shape(2), DType.F32);

        be.Copy(head, dst);

        Assert.Equal(new float[] { 1, 2 }, Host(be, dst));
    }
}
