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

    // --- Elementwise (NumPy-style broadcasting) ---
    Tensor Add(Tensor a, Tensor b);
    Tensor Mul(Tensor a, Tensor b);
    Tensor Sub(Tensor a, Tensor b);
    Tensor Div(Tensor a, Tensor b);
    Tensor AddScalar(Tensor a, float scalar);
    Tensor MulScalar(Tensor a, float scalar);
    /// <summary>Elementwise square root.</summary>
    Tensor Sqrt(Tensor x);
    /// <summary>Elementwise logistic sigmoid 1/(1+e^-x).</summary>
    Tensor Sigmoid(Tensor x);

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

    // --- Indexing & loss (ticket S1-3) ---
    /// <summary>Gathers rows of <paramref name="table"/> ([vocab, dim]) by <paramref name="ids"/> → [ids.Length, dim].</summary>
    Tensor Gather(Tensor table, int[] ids);
    /// <summary>Scatter-adds rows of <paramref name="rows"/> ([N, dim]) into a zeroed [rowCount, dim] keyed by <paramref name="ids"/> (the embedding backward; repeated ids accumulate).</summary>
    Tensor ScatterAddRows(Tensor rows, int[] ids, int rowCount);
    /// <summary>Mean cross-entropy of <paramref name="logits"/> ([N, vocab]) vs integer <paramref name="targets"/>, skipping rows whose target == <paramref name="ignoreIndex"/>.</summary>
    Tensor CrossEntropy(Tensor logits, int[] targets, int ignoreIndex);
    /// <summary>Gradient of <see cref="CrossEntropy"/> w.r.t. logits: (softmax − onehot)/validCount, with ignored rows zeroed.</summary>
    Tensor CrossEntropyGrad(Tensor logits, int[] targets, int ignoreIndex);
}
