using ProjectAI.Core;
// Import only the static helper we use; importing the whole System.Numerics.Tensors
// namespace would pull in its own `Tensor` type and make the name ambiguous with Core's.
using TensorPrimitives = System.Numerics.Tensors.TensorPrimitives;

namespace ProjectAI.Backends.Cpu;

/// <summary>
/// Fully-managed reference backend built on <see cref="TensorPrimitives"/> (SIMD span ops).
/// It is the correctness oracle for the accelerated backends: every GPU kernel is validated by
/// comparing against this implementation. Storage is a plain <see cref="float"/> array per tensor.
/// Not performance-critical.
/// </summary>
public sealed class CpuComputeBackend : IComputeBackend
{
    public string Name => "cpu-reference";
    public Device Device => new(DeviceKind.Cpu);

    private static float[] Buffer(Tensor t) => (float[])t.Handle!;

    /// <summary>Fast access to a tensor's full backing array; rejects non-contiguous views (handled in S0-3).</summary>
    private static float[] Contiguous(Tensor t) =>
        t is { IsContiguous: true, Offset: 0 }
            ? Buffer(t)
            : throw new NotSupportedException(
                "Elementwise kernels require contiguous, zero-offset tensors; strided compute lands in ticket S0-3.");

    public Tensor Allocate(Shape shape, DType dtype) =>
        new(shape, dtype, Device, handle: new float[shape.ElementCount]);

    public Tensor FromHost(ReadOnlySpan<float> data, Shape shape, DType dtype)
    {
        var storage = new float[shape.ElementCount];
        data.CopyTo(storage);
        return new Tensor(shape, dtype, Device, handle: storage);
    }

    /// <summary>
    /// Copies a tensor's logical elements to the host in row-major order. Handles arbitrary views by
    /// gathering through the view's strides/offset, so e.g. a transpose round-trips correctly.
    /// </summary>
    public void ToHost(Tensor source, Span<float> destination)
    {
        var buf = Buffer(source);
        if (source is { IsContiguous: true, Offset: 0 })
        {
            buf.AsSpan(0, (int)source.ElementCount).CopyTo(destination);
            return;
        }

        int i = 0;
        foreach (var offset in source.EnumerateOffsets())
            destination[i++] = buf[(int)offset];
    }

    public void Copy(Tensor source, Tensor destination) =>
        Contiguous(source).AsSpan().CopyTo(Contiguous(destination));

    // --- Elementwise ops: implemented against TensorPrimitives to anchor the reference path. ---
    public Tensor Add(Tensor a, Tensor b)
    {
        var dst = Allocate(a.Shape, a.DType);
        TensorPrimitives.Add(Contiguous(a), Contiguous(b), Buffer(dst));
        return dst;
    }

    public Tensor Mul(Tensor a, Tensor b)
    {
        var dst = Allocate(a.Shape, a.DType);
        TensorPrimitives.Multiply(Contiguous(a), Contiguous(b), Buffer(dst));
        return dst;
    }

    public Tensor AddScalar(Tensor a, float scalar)
    {
        var dst = Allocate(a.Shape, a.DType);
        TensorPrimitives.Add(Contiguous(a), scalar, Buffer(dst));
        return dst;
    }

    public Tensor MulScalar(Tensor a, float scalar)
    {
        var dst = Allocate(a.Shape, a.DType);
        TensorPrimitives.Multiply(Contiguous(a), scalar, Buffer(dst));
        return dst;
    }

    // --- Heavier kernels land in Stage 0 / Stage 1 (see docs/BUILD_PLAN.md). ---
    public Tensor MatMul(Tensor a, Tensor b, bool transposeB = false) =>
        throw new NotImplementedException("Tiled GEMM — ticket S0-3.");

    public Tensor Softmax(Tensor x, int axis) => throw new NotImplementedException("ticket S1-2.");
    public Tensor RmsNorm(Tensor x, Tensor weight, float eps) => throw new NotImplementedException("ticket S1-2.");
    public Tensor Silu(Tensor x) => throw new NotImplementedException("ticket S1-2.");
    public Tensor RotaryEmbedding(Tensor x, Tensor cos, Tensor sin) => throw new NotImplementedException("ticket S1-2.");

    public void Dispose() { }
}
