namespace ProjectAI.Core;

/// <summary>
/// A multi-dimensional array handle. The numeric storage is owned by the <see cref="IComputeBackend"/>
/// that produced it; <see cref="Handle"/> is an opaque, backend-specific reference (a managed array,
/// a device pointer, or a libtorch tensor).
///
/// <para>Layout follows NumPy/PyTorch: a tensor is a <em>view</em> over storage described by
/// <see cref="Strides"/> (element strides per axis) and <see cref="Offset"/> (element offset into the
/// storage). View operations (<see cref="Reshape"/>, <see cref="Transpose"/>, <see cref="Permute"/>,
/// <see cref="Slice"/>) only rewrite this metadata and share the same <see cref="Handle"/> — no data
/// is copied.</para>
///
/// <para>Autograd metadata lives on the same type so one object flows through both forward and
/// backward passes.</para>
/// </summary>
public sealed class Tensor
{
    /// <summary>Creates a contiguous, zero-offset tensor over the given storage handle.</summary>
    public Tensor(Shape shape, DType dtype, Device device, object? handle = null, bool requiresGrad = false)
        : this(shape, shape.ContiguousStrides(), offset: 0, dtype, device, handle, requiresGrad)
    {
    }

    private Tensor(Shape shape, long[] strides, long offset, DType dtype, Device device, object? handle, bool requiresGrad)
    {
        Shape = shape;
        Strides = strides;
        Offset = offset;
        DType = dtype;
        Device = device;
        Handle = handle;
        RequiresGrad = requiresGrad;
    }

    public Shape Shape { get; }

    /// <summary>Element strides aligned to <see cref="Shape"/>'s axes (not byte strides).</summary>
    public long[] Strides { get; }

    /// <summary>Element offset of this view's first element within <see cref="Handle"/>.</summary>
    public long Offset { get; }

    public DType DType { get; }
    public Device Device { get; }

    /// <summary>Opaque, backend-owned storage handle. Interpreted only by the producing backend.</summary>
    public object? Handle { get; internal set; }

    public long ElementCount => Shape.ElementCount;

    /// <summary>True if this view's strides describe a densely-packed, row-major layout.</summary>
    public bool IsContiguous
    {
        get
        {
            var dims = Shape.Dimensions;
            long acc = 1;
            for (int i = dims.Length - 1; i >= 0; i--)
            {
                if (dims[i] == 1) continue;          // size-1 axes never break contiguity
                if (Strides[i] != acc) return false;
                acc *= dims[i];
            }
            return true;
        }
    }

    // --- View operations (metadata only; share Handle) ---
    // These are NON-differentiable primitives: the returned view is detached (RequiresGrad = false).
    // For a view inside an autograd graph, use the matching op on the Autograd facade, which records a
    // GradNode. This keeps "the tape is built in one place" (the facade) and avoids dangling views.

    /// <summary>Returns a view with a new shape. Requires a contiguous tensor (the common case).</summary>
    public Tensor Reshape(params int[] newDimensions)
    {
        if (!IsContiguous)
            throw new InvalidOperationException("Reshape requires a contiguous tensor; materialize it first.");

        var newShape = new Shape(newDimensions);
        if (newShape.ElementCount != Shape.ElementCount)
            throw new ArgumentException(
                $"Cannot reshape {Shape} ({Shape.ElementCount} elements) to {newShape} ({newShape.ElementCount} elements).");

        return new Tensor(newShape, newShape.ContiguousStrides(), Offset, DType, Device, Handle, requiresGrad: false);
    }

    /// <summary>Returns a view with two axes swapped (produces a non-contiguous view in general).</summary>
    public Tensor Transpose(int axis0, int axis1)
    {
        int rank = Shape.Rank;
        if ((uint)axis0 >= (uint)rank) throw new ArgumentOutOfRangeException(nameof(axis0));
        if ((uint)axis1 >= (uint)rank) throw new ArgumentOutOfRangeException(nameof(axis1));

        var dims = Shape.Dimensions.ToArray();
        var strides = (long[])Strides.Clone();
        (dims[axis0], dims[axis1]) = (dims[axis1], dims[axis0]);
        (strides[axis0], strides[axis1]) = (strides[axis1], strides[axis0]);
        return new Tensor(new Shape(dims), strides, Offset, DType, Device, Handle, requiresGrad: false);
    }

    /// <summary>Returns a view whose axes are reordered by <paramref name="permutation"/>.</summary>
    public Tensor Permute(params int[] permutation)
    {
        int rank = Shape.Rank;
        if (permutation.Length != rank)
            throw new ArgumentException($"Permutation length {permutation.Length} does not match rank {rank}.", nameof(permutation));

        var srcDims = Shape.Dimensions;
        var newDims = new int[rank];
        var newStrides = new long[rank];
        var seen = new bool[rank];
        for (int i = 0; i < rank; i++)
        {
            int p = permutation[i];
            if ((uint)p >= (uint)rank || seen[p])
                throw new ArgumentException("Permutation must be a rearrangement of 0..rank-1.", nameof(permutation));
            seen[p] = true;
            newDims[i] = srcDims[p];
            newStrides[i] = Strides[p];
        }
        return new Tensor(new Shape(newDims), newStrides, Offset, DType, Device, Handle, requiresGrad: false);
    }

    /// <summary>Returns a view of <paramref name="length"/> elements along <paramref name="axis"/> starting at <paramref name="start"/>.</summary>
    public Tensor Slice(int axis, int start, int length)
    {
        int rank = Shape.Rank;
        if ((uint)axis >= (uint)rank) throw new ArgumentOutOfRangeException(nameof(axis));
        int dim = Shape[axis];
        if (start < 0 || length < 0 || start + length > dim)
            throw new ArgumentOutOfRangeException(nameof(start), $"Slice [{start}, {start + length}) is out of range for axis {axis} (size {dim}).");

        var dims = Shape.Dimensions.ToArray();
        dims[axis] = length;
        long newOffset = Offset + (long)start * Strides[axis];
        return new Tensor(new Shape(dims), (long[])Strides.Clone(), newOffset, DType, Device, Handle, requiresGrad: false);
    }

    /// <summary>
    /// Returns a view broadcast to <paramref name="target"/> following NumPy rules: axes are aligned on
    /// the right, and each size-1 axis (or newly-introduced leading axis) is given a stride of 0 so reads
    /// repeat that element. No data is copied. Throws if this shape is not broadcastable to the target.
    /// </summary>
    public Tensor BroadcastTo(Shape target)
    {
        var src = Shape.Dimensions;
        var tgt = target.Dimensions;
        if (src.Length > tgt.Length)
            throw new ArgumentException($"Cannot broadcast {Shape} to fewer dimensions {target}.");

        int lead = tgt.Length - src.Length;            // new leading axes the source lacks
        var newStrides = new long[tgt.Length];
        for (int i = 0; i < tgt.Length; i++)
        {
            if (i < lead)
            {
                newStrides[i] = 0;                      // brand-new leading axis: repeat
                continue;
            }
            int s = src[i - lead];
            if (s == tgt[i]) newStrides[i] = Strides[i - lead];
            else if (s == 1) newStrides[i] = 0;         // size-1 axis: repeat
            else throw new ArgumentException($"Cannot broadcast {Shape} to {target}: axis {i} ({s} vs {tgt[i]}).");
        }
        return new Tensor(target, newStrides, Offset, DType, Device, Handle, requiresGrad: false);
    }

    /// <summary>
    /// Enumerates the physical element offsets into <see cref="Handle"/> in logical row-major order.
    /// Backends use this to gather a (possibly strided) view into a contiguous buffer.
    /// </summary>
    public IEnumerable<long> EnumerateOffsets()
    {
        var dims = Shape.Dimensions.ToArray();   // materialize: ReadOnlySpan can't cross a yield
        int rank = dims.Length;
        if (rank == 0)
        {
            yield return Offset;
            yield break;
        }

        long total = Shape.ElementCount;
        var counter = new int[rank];
        for (long n = 0; n < total; n++)
        {
            long off = Offset;
            for (int i = 0; i < rank; i++) off += counter[i] * Strides[i];
            yield return off;

            for (int axis = rank - 1; axis >= 0; axis--)
            {
                if (++counter[axis] < dims[axis]) break;
                counter[axis] = 0;
            }
        }
    }

    // --- Autograd (reverse-mode) ---
    public bool RequiresGrad { get; set; }
    public Tensor? Grad { get; set; }

    /// <summary>The op that produced this tensor, used to walk the backward graph. Null for leaf tensors.</summary>
    public GradNode? GradFn { get; internal set; }

    /// <summary>The backend that produced this tensor; drives the backward pass. Set by the autograd facade.</summary>
    internal IComputeBackend? Backend { get; set; }

    /// <summary>
    /// Seeds this tensor's gradient with ones and runs reverse-mode autodiff over the recorded graph,
    /// accumulating <see cref="Grad"/> on every tensor that requires grad. Call on a scalar loss.
    /// </summary>
    public void Backward()
    {
        var backend = Backend
            ?? throw new InvalidOperationException(
                "Backward requires a tensor produced by the Autograd facade (no graph/backend recorded).");
        if (Shape.ElementCount != 1)
            throw new InvalidOperationException(
                $"Backward expects a scalar loss; this tensor has shape {Shape}. Reduce to a scalar first.");

        // Topological order: every input appears before the op that consumes it.
        var topo = new List<Tensor>();
        var visited = new HashSet<Tensor>();
        BuildTopo(this, visited, topo);

        // Clear gradients on intermediate (non-leaf) nodes so a second Backward over the same graph
        // doesn't accumulate onto stale values. Leaf grads are left alone (their accumulation across
        // graphs is intentional and managed by the optimizer's ZeroGrad).
        foreach (var node in topo)
            if (node.GradFn is not null)
                node.Grad = null;

        // Seed d(self)/d(self) = 1 (for a scalar loss this is the usual 1.0).
        Grad = backend.AddScalar(backend.Allocate(Shape, DType), 1f);

        for (int i = topo.Count - 1; i >= 0; i--)
        {
            var node = topo[i];
            if (node.GradFn is null || node.Grad is null) continue;

            var inputs = node.GradFn.Inputs;
            var grads = node.GradFn.Backward(node.Grad);
            for (int j = 0; j < inputs.Count; j++)
            {
                var input = inputs[j];
                if (!input.RequiresGrad) continue;
                // Accumulate so a tensor used by multiple ops (fan-out) sums its gradients.
                input.Grad = input.Grad is null ? grads[j] : backend.Add(input.Grad, grads[j]);
            }
        }
    }

    private static void BuildTopo(Tensor t, HashSet<Tensor> visited, List<Tensor> topo)
    {
        if (!visited.Add(t)) return;
        if (t.GradFn is not null)
            foreach (var input in t.GradFn.Inputs)
                BuildTopo(input, visited, topo);
        topo.Add(t);
    }

    public override string ToString() =>
        $"Tensor(shape={Shape}, dtype={DType}, device={Device}, contiguous={IsContiguous}, requiresGrad={RequiresGrad})";
}
