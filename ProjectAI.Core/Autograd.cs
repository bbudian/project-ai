namespace ProjectAI.Core;

/// <summary>
/// The differentiable op layer (reverse-mode autodiff). Each method calls a non-differentiable
/// <see cref="IComputeBackend"/> op to compute the forward value, then — while grad mode is on and an
/// input requires grad — records a <see cref="GradNode"/> capturing the local backward rule.
/// <see cref="Tensor.Backward"/> walks those nodes. Backends never build the tape; this facade is the
/// single place the graph is constructed, so model <c>Forward</c> code should route through it rather
/// than calling the backend directly (ticket S0-4).
/// </summary>
public sealed class Autograd(IComputeBackend backend)
{
    public IComputeBackend Backend { get; } = backend;

    /// <summary>Elementwise add, y = a + b (NumPy broadcasting).</summary>
    public Tensor Add(Tensor a, Tensor b)
    {
        var y = Backend.Add(a, b);
        return Record(y, "add", [a, b], g =>
        [
            ReduceGradToShape(g, a.Shape),
            ReduceGradToShape(g, b.Shape),
        ]);
    }

    /// <summary>Elementwise subtract, y = a - b (NumPy broadcasting).</summary>
    public Tensor Sub(Tensor a, Tensor b)
    {
        var y = Backend.Sub(a, b);
        return Record(y, "sub", [a, b], g =>
        [
            ReduceGradToShape(g, a.Shape),
            ReduceGradToShape(Backend.MulScalar(g, -1f), b.Shape),
        ]);
    }

    /// <summary>Elementwise multiply, y = a * b (NumPy broadcasting).</summary>
    public Tensor Mul(Tensor a, Tensor b)
    {
        var y = Backend.Mul(a, b);
        return Record(y, "mul", [a, b], g =>
        [
            ReduceGradToShape(Backend.Mul(g, b), a.Shape),
            ReduceGradToShape(Backend.Mul(g, a), b.Shape),
        ]);
    }

    /// <summary>2-D matrix multiply, y = a · b. (Batched/attention autograd arrives in ticket S1-7.)</summary>
    public Tensor MatMul(Tensor a, Tensor b)
    {
        if (a.Shape.Rank != 2 || b.Shape.Rank != 2)
            throw new NotImplementedException("Differentiable MatMul is 2-D for now (batched grad: ticket S1-7).");

        var y = Backend.MatMul(a, b);
        return Record(y, "matmul", [a, b], g =>
        [
            Backend.MatMul(g, b, transposeB: true),    // grad_a = g · bᵀ
            Backend.MatMul(a.Transpose(0, 1), g),      // grad_b = aᵀ · g
        ]);
    }

    /// <summary>Mean over all elements, producing a scalar.</summary>
    public Tensor Mean(Tensor x)
    {
        long count = x.ElementCount;
        var y = x;
        for (int axis = x.Shape.Rank - 1; axis >= 0; axis--) y = Backend.Mean(y, axis); // → scalar; strided-safe
        return Record(y, "mean", [x], g =>
        {
            // d(mean)/d(x_i) = 1/count; g is the scalar upstream gradient.
            float scaled = ToScalar(g) / count;
            return [Backend.AddScalar(Backend.Allocate(x.Shape, x.DType), scaled)];
        });
    }

    // --- Differentiable view ops (raw Tensor views are detached; route through here to keep gradients) ---

    /// <summary>Swaps two axes; gradient transposes back.</summary>
    public Tensor Transpose(Tensor x, int axis0, int axis1)
    {
        var y = x.Transpose(axis0, axis1);
        return Record(y, "transpose", [x], g => [g.Transpose(axis0, axis1)]);
    }

    /// <summary>Reorders axes; gradient permutes by the inverse.</summary>
    public Tensor Permute(Tensor x, params int[] permutation)
    {
        var inverse = new int[permutation.Length];
        for (int i = 0; i < permutation.Length; i++) inverse[permutation[i]] = i;
        var y = x.Permute(permutation);
        return Record(y, "permute", [x], g => [g.Permute(inverse)]);
    }

    /// <summary>Reshapes a contiguous tensor; gradient reshapes back to the original shape.</summary>
    public Tensor Reshape(Tensor x, params int[] newDimensions)
    {
        var original = x.Shape.Dimensions.ToArray();
        var y = x.Reshape(newDimensions);
        return Record(y, "reshape", [x], g => [g.Reshape(original)]);
    }

    /// <summary>Slices along an axis; gradient scatters back into a zero tensor of the input's shape.</summary>
    public Tensor Slice(Tensor x, int axis, int start, int length)
    {
        var shape = x.Shape;
        var dtype = x.DType;
        var y = x.Slice(axis, start, length);
        return Record(y, "slice", [x], g =>
        {
            var grad = Backend.Allocate(shape, dtype);            // zeros
            Backend.Copy(g, grad.Slice(axis, start, length));     // scatter g into the sliced region
            return [grad];
        });
    }

    /// <summary>Broadcasts to a larger shape; gradient sums back over the broadcast axes.</summary>
    public Tensor BroadcastTo(Tensor x, Shape target)
    {
        var y = x.BroadcastTo(target);
        return Record(y, "broadcast", [x], g => [ReduceGradToShape(g, x.Shape)]);
    }

    /// <summary>Stamps autograd metadata onto a freshly-computed forward result.</summary>
    private Tensor Record(Tensor y, string op, IReadOnlyList<Tensor> inputs, Func<Tensor, IReadOnlyList<Tensor>> backward)
    {
        y.Backend = Backend;
        if (GradMode.IsEnabled && inputs.Any(static i => i.RequiresGrad))
        {
            y.RequiresGrad = true;
            y.GradFn = new GradNode(op, inputs, backward);
        }
        return y;
    }

    /// <summary>Sums a gradient back down to <paramref name="target"/> over any axes that were broadcast.</summary>
    private Tensor ReduceGradToShape(Tensor grad, Shape target)
    {
        if (grad.Shape.Equals(target)) return grad;

        var g = grad;
        // Capture the count once: each Sum(axis:0) drops a rank, so reading g.Shape.Rank in the
        // loop bound would under-iterate and leave leading broadcast axes un-reduced.
        int leadingToDrop = grad.Shape.Rank - target.Rank;
        for (int i = 0; i < leadingToDrop; i++) g = Backend.Sum(g, axis: 0);
        for (int axis = 0; axis < target.Rank; axis++)
            if (target[axis] == 1 && g.Shape[axis] != 1)
                g = Backend.Sum(g, axis, keepDims: true);                                 // collapse broadcast axes
        return g;
    }

    private float ToScalar(Tensor t)
    {
        Span<float> one = stackalloc float[1];
        Backend.ToHost(t, one);
        return one[0];
    }
}
