namespace ProjectAI.Core;

/// <summary>
/// The single abstraction every compute path implements. This is the seam that makes the
/// two-backend strategy work: swapping the managed CPU reference, TorchSharp/libtorch, or the
/// hand-written Vulkan compute backend must require no change to any code above this interface.
/// Every op consumes and produces <see cref="Tensor"/> handles allocated by the same backend.
/// </summary>
public interface IComputeBackend : IDisposable
{
    string Name { get; }
    Device Device { get; }

    // --- Allocation & host transfer ---
    Tensor Allocate(Shape shape, DType dtype);
    Tensor FromHost(ReadOnlySpan<float> data, Shape shape, DType dtype);
    void ToHost(Tensor source, Span<float> destination);
    void Copy(Tensor source, Tensor destination);

    // --- Elementwise ---
    Tensor Add(Tensor a, Tensor b);
    Tensor Mul(Tensor a, Tensor b);
    Tensor AddScalar(Tensor a, float scalar);
    Tensor MulScalar(Tensor a, float scalar);

    // --- Linear algebra ---
    /// <summary>Batched matrix multiply. <paramref name="transposeB"/> fuses the common weight transpose.</summary>
    Tensor MatMul(Tensor a, Tensor b, bool transposeB = false);

    // --- Reductions (along a single axis; negative axis counts from the end) ---
    /// <summary>Sum along <paramref name="axis"/>. When <paramref name="keepDims"/> is false the reduced axis is removed.</summary>
    Tensor Sum(Tensor x, int axis, bool keepDims = false);
    /// <summary>Arithmetic mean along <paramref name="axis"/>.</summary>
    Tensor Mean(Tensor x, int axis, bool keepDims = false);
    /// <summary>Maximum along <paramref name="axis"/>.</summary>
    Tensor Max(Tensor x, int axis, bool keepDims = false);

    // --- Transformer primitives (kept on the seam so each backend can fuse them) ---
    Tensor Softmax(Tensor x, int axis);
    Tensor RmsNorm(Tensor x, Tensor weight, float eps);
    Tensor Silu(Tensor x);
    Tensor RotaryEmbedding(Tensor x, Tensor cos, Tensor sin);
}
