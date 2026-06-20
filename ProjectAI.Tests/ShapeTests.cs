using ProjectAI.Core;
using Xunit;

namespace ProjectAI.Tests;

public class ShapeTests
{
    [Fact]
    public void ContiguousStrides_AreRowMajor()
    {
        Assert.Equal(new long[] { 12, 4, 1 }, new Shape(2, 3, 4).ContiguousStrides());
        Assert.Equal(new long[] { 1 }, new Shape(5).ContiguousStrides());
        Assert.Equal(System.Array.Empty<long>(), Shape.Scalar.ContiguousStrides());
    }

    [Fact]
    public void Scalar_HasRankZeroAndOneElement()
    {
        Assert.Equal(0, Shape.Scalar.Rank);
        Assert.Equal(1L, Shape.Scalar.ElementCount);
    }

    [Fact]
    public void Ravel_Unravel_RoundTrip()
    {
        var shape = new Shape(2, 3, 4);
        Assert.Equal(23L, shape.RavelIndex(new[] { 1, 2, 3 }));
        Assert.Equal(new[] { 1, 2, 3 }, shape.UnravelIndex(23));

        for (long linear = 0; linear < shape.ElementCount; linear++)
            Assert.Equal(linear, shape.RavelIndex(shape.UnravelIndex(linear)));
    }

    [Fact]
    public void RavelIndex_RejectsOutOfRange()
    {
        var shape = new Shape(2, 3);
        Assert.Throws<IndexOutOfRangeException>(() => shape.RavelIndex(new[] { 2, 0 }));
        Assert.Throws<ArgumentException>(() => shape.RavelIndex(new[] { 0 }));
    }

    [Theory]
    [InlineData(new[] { 3, 1 }, new[] { 1, 4 }, new[] { 3, 4 })]
    [InlineData(new[] { 2, 3, 4 }, new[] { 4 }, new[] { 2, 3, 4 })]
    [InlineData(new[] { 256 }, new[] { 256 }, new[] { 256 })]
    [InlineData(new[] { 5, 1, 6 }, new[] { 1, 7, 1 }, new[] { 5, 7, 6 })]
    public void Broadcast_FollowsNumpyRules(int[] a, int[] b, int[] expected)
    {
        Assert.Equal(new Shape(expected), new Shape(a).BroadcastWith(new Shape(b)));
    }

    [Fact]
    public void Broadcast_IncompatibleShapes_Fail()
    {
        Assert.False(new Shape(5).TryBroadcastWith(new Shape(6), out _));
        Assert.Throws<ArgumentException>(() => new Shape(5).BroadcastWith(new Shape(6)));
    }

    [Fact]
    public void NegativeDimension_Throws() =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new Shape(2, -1));
}
