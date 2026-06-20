using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using Xunit;

namespace ProjectAI.Tests;

public class CoreContractsTests
{
    [Fact]
    public void Shape_ElementCount_IsProductOfDimensions()
    {
        var shape = new Shape(2, 3, 4);
        Assert.Equal(24L, shape.ElementCount);
        Assert.Equal(3, shape.Rank);
        Assert.Equal("[2, 3, 4]", shape.ToString());
    }

    [Fact]
    public void CpuBackend_Add_ComputesElementwiseSum()
    {
        using var backend = new CpuComputeBackend();
        var a = backend.FromHost([1f, 2f, 3f], new Shape(3), DType.F32);
        var b = backend.FromHost([10f, 20f, 30f], new Shape(3), DType.F32);

        var sum = backend.Add(a, b);

        var result = new float[3];
        backend.ToHost(sum, result);
        Assert.Equal(new float[] { 11f, 22f, 33f }, result);
    }

    [Fact]
    public void DType_SizeInBytes_IsCorrect()
    {
        Assert.Equal(4, DType.F32.SizeInBytes());
        Assert.Equal(2, DType.BF16.SizeInBytes());
        Assert.Equal(8, DType.I64.SizeInBytes());
    }

    [Fact(Skip = "Autograd backward pass is implemented in Stage 0 (ticket S0-4).")]
    public void Autograd_NumericGradientCheck_MatchesFiniteDifferences()
    {
        // Placeholder for the Stage 0 gradient-check harness.
    }
}
