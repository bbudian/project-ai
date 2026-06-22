using ProjectAI.Core;
// Import only the static helper we use; importing the whole System.Numerics.Tensors
// namespace would pull in its own `Tensor` type and make the name ambiguous with Core's.
using TensorPrimitives = System.Numerics.Tensors.TensorPrimitives;

namespace ProjectAI.Backends.Cpu;

/// <summary>
/// Fully-managed reference backend built on <see cref="TensorPrimitives"/> (SIMD span ops).
/// It is the correctness oracle for the accelerated backends: every GPU kernel is validated by
/// comparing against this implementation. Storage is a plain <see cref="float"/> array per tensor.
/// Correctness over speed — kernels favour clarity and materialize strided views into contiguous
/// buffers rather than running fully stride-aware loops.
/// </summary>
public sealed class CpuComputeBackend : IComputeBackend
{
    public string Name => "cpu-reference";
    public Device Device => new(DeviceKind.Cpu);

    private static float[] Buffer(Tensor t) => (float[])t.Handle!;

    /// <summary>
    /// Returns a tensor's logical elements as a dense, row-major <see cref="float"/> array. Contiguous,
    /// zero-offset tensors are returned without copying; any other view is gathered through its
    /// strides/offset. This is how the oracle "handles non-contiguous inputs" (ticket S0-3).
    /// </summary>
    private static float[] Materialize(Tensor t)
    {
        var buf = Buffer(t);
        // Only hand back the raw buffer when it is exactly the logical extent. A contiguous,
        // zero-offset *head-slice* (e.g. Slice(0,0,2) of a length-4 tensor) is contiguous with
        // offset 0 yet shorter than its backing array, so it must take the gather path.
        if (t is { IsContiguous: true, Offset: 0 } && buf.Length == t.ElementCount) return buf;

        var dst = new float[t.ElementCount];
        int i = 0;
        foreach (var offset in t.EnumerateOffsets()) dst[i++] = buf[(int)offset];
        return dst;
    }

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

    public void Copy(Tensor source, Tensor destination)
    {
        var src = Materialize(source);
        if (destination is { IsContiguous: true, Offset: 0 })
        {
            src.AsSpan(0, (int)source.ElementCount).CopyTo(Buffer(destination));
            return;
        }

        var dst = Buffer(destination);
        int i = 0;
        foreach (var offset in destination.EnumerateOffsets()) dst[(int)offset] = src[i++];
    }

    // --- Elementwise (NumPy-style broadcasting; strided inputs are gathered) ---
    public Tensor Add(Tensor a, Tensor b) => Elementwise(a, b, static (x, y) => x + y, isAdd: true);
    public Tensor Mul(Tensor a, Tensor b) => Elementwise(a, b, static (x, y) => x * y, isAdd: false);

    private Tensor Elementwise(Tensor a, Tensor b, Func<float, float, float> op, bool isAdd)
    {
        var shape = a.Shape.BroadcastWith(b.Shape);
        var dst = Allocate(shape, a.DType);
        var outBuf = Buffer(dst);

        // Fast path: identical shapes, both densely packed over a full-extent buffer → SIMD.
        // The buffer-length checks exclude contiguous head-slices (backing array longer than the
        // logical extent), which must fall through to the lockstep gather path below.
        if (a.Shape.Equals(shape) && b.Shape.Equals(shape) &&
            a is { IsContiguous: true, Offset: 0 } && b is { IsContiguous: true, Offset: 0 } &&
            Buffer(a).Length == shape.ElementCount && Buffer(b).Length == shape.ElementCount)
        {
            if (isAdd) TensorPrimitives.Add(Buffer(a), Buffer(b), outBuf);
            else TensorPrimitives.Multiply(Buffer(a), Buffer(b), outBuf);
            return dst;
        }

        // General path: broadcast both operands to the result shape and gather in lockstep.
        var bufA = Buffer(a);
        var bufB = Buffer(b);
        var av = a.BroadcastTo(shape);
        var bv = b.BroadcastTo(shape);
        int i = 0;
        using var ea = av.EnumerateOffsets().GetEnumerator();
        using var eb = bv.EnumerateOffsets().GetEnumerator();
        while (ea.MoveNext() && eb.MoveNext())
            outBuf[i++] = op(bufA[(int)ea.Current], bufB[(int)eb.Current]);
        return dst;
    }

    public Tensor AddScalar(Tensor a, float scalar)
    {
        var dst = Allocate(a.Shape, a.DType);
        TensorPrimitives.Add(Materialize(a), scalar, Buffer(dst));
        return dst;
    }

    public Tensor MulScalar(Tensor a, float scalar)
    {
        var dst = Allocate(a.Shape, a.DType);
        TensorPrimitives.Multiply(Materialize(a), scalar, Buffer(dst));
        return dst;
    }

    // --- Linear algebra: batched GEMM (ticket S0-3) ---
    public Tensor MatMul(Tensor a, Tensor b, bool transposeB = false)
    {
        int ra = a.Shape.Rank, rb = b.Shape.Rank;
        if (ra < 2 || rb < 2)
            throw new ArgumentException($"MatMul requires rank ≥ 2 operands; got {a.Shape} and {b.Shape}.");

        int m = a.Shape[ra - 2];
        int k = a.Shape[ra - 1];
        int kb = transposeB ? b.Shape[rb - 1] : b.Shape[rb - 2];
        int n = transposeB ? b.Shape[rb - 2] : b.Shape[rb - 1];
        if (k != kb)
            throw new ArgumentException($"MatMul inner dimensions disagree: {a.Shape} x {b.Shape} (transposeB={transposeB}).");

        // Broadcast the batch dims (everything but the trailing two axes).
        var aBatch = a.Shape.Dimensions[..(ra - 2)].ToArray();
        var bBatch = b.Shape.Dimensions[..(rb - 2)].ToArray();
        var batch = new Shape(aBatch).BroadcastWith(new Shape(bBatch));
        var batchDims = batch.Dimensions.ToArray();
        long batchCount = batch.ElementCount;

        // Materialize to contiguous, then treat each operand as a stack of 2D blocks.
        var bufA = Materialize(a);
        var bufB = Materialize(b);
        long aBlock = (long)m * k;
        long bBlock = (long)kb * n; // (transposeB) ? n*k : k*n — kb==k either way

        var outBuf = new float[batchCount * m * n];
        var counter = new int[batchDims.Length];
        for (long bi = 0; bi < batchCount; bi++)
        {
            long aOff = BatchIndex(counter, aBatch) * aBlock;
            long bOff = BatchIndex(counter, bBatch) * bBlock;
            long oOff = bi * m * n;

            for (int i = 0; i < m; i++)
            {
                long aRow = aOff + (long)i * k;
                for (int j = 0; j < n; j++)
                {
                    float acc = 0f;
                    if (transposeB)
                    {
                        long bRow = bOff + (long)j * k; // b is [n, k]
                        for (int p = 0; p < k; p++) acc += bufA[aRow + p] * bufB[bRow + p];
                    }
                    else
                    {
                        for (int p = 0; p < k; p++) acc += bufA[aRow + p] * bufB[bOff + (long)p * n + j];
                    }
                    outBuf[oOff + (long)i * n + j] = acc;
                }
            }
            Increment(counter, batchDims);
        }

        var outDims = new int[batchDims.Length + 2];
        Array.Copy(batchDims, outDims, batchDims.Length);
        outDims[^2] = m;
        outDims[^1] = n;
        return new Tensor(new Shape(outDims), a.DType, Device, handle: outBuf);
    }

    /// <summary>Linear index into an input's batch given the broadcasted batch counter (size-1 dims repeat).</summary>
    private static long BatchIndex(int[] counter, int[] inBatch)
    {
        int lead = counter.Length - inBatch.Length;
        long idx = 0;
        for (int i = 0; i < inBatch.Length; i++)
        {
            int dim = inBatch[i];
            int use = dim == 1 ? 0 : counter[lead + i];
            idx = idx * dim + use;
        }
        return idx;
    }

    /// <summary>Advances a row-major multi-index counter in place.</summary>
    private static void Increment(int[] counter, int[] dims)
    {
        for (int axis = counter.Length - 1; axis >= 0; axis--)
        {
            if (++counter[axis] < dims[axis]) return;
            counter[axis] = 0;
        }
    }

    // --- Reductions along a single axis (ticket S0-3) ---
    private enum ReduceKind { Sum, Mean, Max }

    public Tensor Sum(Tensor x, int axis, bool keepDims = false) => Reduce(x, axis, keepDims, ReduceKind.Sum);
    public Tensor Mean(Tensor x, int axis, bool keepDims = false) => Reduce(x, axis, keepDims, ReduceKind.Mean);
    public Tensor Max(Tensor x, int axis, bool keepDims = false) => Reduce(x, axis, keepDims, ReduceKind.Max);

    private Tensor Reduce(Tensor x, int axis, bool keepDims, ReduceKind kind)
    {
        int rank = x.Shape.Rank;
        if (rank == 0) throw new ArgumentException("Cannot reduce a scalar tensor.");
        int ax = axis < 0 ? axis + rank : axis;
        if ((uint)ax >= (uint)rank)
            throw new ArgumentOutOfRangeException(nameof(axis), axis, $"Axis out of range for shape {x.Shape}.");

        var dims = x.Shape.Dimensions;
        long outer = 1;
        for (int i = 0; i < ax; i++) outer *= dims[i];
        int axisLen = dims[ax];
        long inner = 1;
        for (int i = ax + 1; i < rank; i++) inner *= dims[i];

        // Materialized layout is [outer, axisLen, inner] in row-major order.
        var src = Materialize(x);
        var outBuf = new float[outer * inner];
        for (long o = 0; o < outer; o++)
        {
            long blockBase = o * axisLen * inner;
            for (long c = 0; c < inner; c++)
            {
                float acc = kind == ReduceKind.Max ? float.NegativeInfinity : 0f;
                for (int t = 0; t < axisLen; t++)
                {
                    float v = src[blockBase + (long)t * inner + c];
                    acc = kind == ReduceKind.Max ? MathF.Max(acc, v) : acc + v;
                }
                if (kind == ReduceKind.Mean) acc /= axisLen;
                outBuf[o * inner + c] = acc;
            }
        }

        int outRank = keepDims ? rank : rank - 1;
        var outDims = new int[outRank];
        int w = 0;
        for (int i = 0; i < rank; i++)
        {
            if (i == ax)
            {
                if (keepDims) outDims[w++] = 1;
            }
            else
            {
                outDims[w++] = dims[i];
            }
        }
        return new Tensor(new Shape(outDims), x.DType, Device, handle: outBuf);
    }

    // --- Transformer primitives land in Stage 1 (see docs/BUILD_PLAN.md). ---
    public Tensor Softmax(Tensor x, int axis) => throw new NotImplementedException("ticket S1-2.");
    public Tensor RmsNorm(Tensor x, Tensor weight, float eps) => throw new NotImplementedException("ticket S1-2.");
    public Tensor Silu(Tensor x) => throw new NotImplementedException("ticket S1-2.");
    public Tensor RotaryEmbedding(Tensor x, Tensor cos, Tensor sin) => throw new NotImplementedException("ticket S1-2.");

    public void Dispose() { }
}
