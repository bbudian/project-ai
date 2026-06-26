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
    private enum BinOp { Add, Mul, Sub, Div }

    public Tensor Add(Tensor a, Tensor b) => Elementwise(a, b, BinOp.Add);
    public Tensor Mul(Tensor a, Tensor b) => Elementwise(a, b, BinOp.Mul);
    public Tensor Sub(Tensor a, Tensor b) => Elementwise(a, b, BinOp.Sub);
    public Tensor Div(Tensor a, Tensor b) => Elementwise(a, b, BinOp.Div);

    private Tensor Elementwise(Tensor a, Tensor b, BinOp kind)
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
            switch (kind)
            {
                case BinOp.Add: TensorPrimitives.Add(Buffer(a), Buffer(b), outBuf); break;
                case BinOp.Mul: TensorPrimitives.Multiply(Buffer(a), Buffer(b), outBuf); break;
                case BinOp.Sub: TensorPrimitives.Subtract(Buffer(a), Buffer(b), outBuf); break;
                case BinOp.Div: TensorPrimitives.Divide(Buffer(a), Buffer(b), outBuf); break;
            }
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
        {
            float x = bufA[(int)ea.Current], y = bufB[(int)eb.Current];
            outBuf[i++] = kind switch
            {
                BinOp.Add => x + y,
                BinOp.Mul => x * y,
                BinOp.Sub => x - y,
                BinOp.Div => x / y,
                _ => throw new ArgumentOutOfRangeException(nameof(kind)),
            };
        }
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

    public Tensor Sqrt(Tensor x)
    {
        var dst = Allocate(x.Shape, x.DType);
        TensorPrimitives.Sqrt(Materialize(x), Buffer(dst));
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
        long bBlock = (long)kb * n; // b's 2-D block size = kb*n (== k*n; for transposeB, b is [n,k] so kb==k)

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

    /// <summary>Numerically-stable elementwise logistic sigmoid (sign-split to keep exp arguments ≤ 0).</summary>
    public Tensor Sigmoid(Tensor x)
    {
        var src = Materialize(x);
        var dst = new float[src.Length];
        for (int i = 0; i < src.Length; i++) dst[i] = SigmoidScalar(src[i]);
        return new Tensor(x.Shape, x.DType, Device, handle: dst);
    }

    private static float SigmoidScalar(float v)
    {
        if (v >= 0f) return 1f / (1f + MathF.Exp(-v));
        float t = MathF.Exp(v);
        return t / (1f + t);
    }

    // --- Transformer primitives (ticket S1-2) ---

    /// <summary>Numerically-stable softmax along <paramref name="axis"/> (subtract the max before exp).</summary>
    public Tensor Softmax(Tensor x, int axis)
    {
        int rank = x.Shape.Rank;
        if (rank == 0) throw new ArgumentException("Softmax requires rank ≥ 1.");
        int ax = axis < 0 ? axis + rank : axis;
        if ((uint)ax >= (uint)rank) throw new ArgumentOutOfRangeException(nameof(axis));

        var dims = x.Shape.Dimensions;
        long outer = 1;
        for (int i = 0; i < ax; i++) outer *= dims[i];
        int axisLen = dims[ax];
        long inner = 1;
        for (int i = ax + 1; i < rank; i++) inner *= dims[i];

        var src = Materialize(x);
        var dst = new float[src.Length];
        for (long o = 0; o < outer; o++)
        {
            long blockBase = o * axisLen * inner;
            for (long c = 0; c < inner; c++)
            {
                long b = blockBase + c;
                float max = float.NegativeInfinity;
                for (int k = 0; k < axisLen; k++) { float v = src[b + (long)k * inner]; if (v > max) max = v; }
                float sum = 0f;
                for (int k = 0; k < axisLen; k++)
                {
                    float e = MathF.Exp(src[b + (long)k * inner] - max);
                    dst[b + (long)k * inner] = e;
                    sum += e;
                }
                float invSum = 1f / sum;
                for (int k = 0; k < axisLen; k++) dst[b + (long)k * inner] *= invSum;
            }
        }
        return new Tensor(x.Shape, x.DType, Device, handle: dst);
    }

    /// <summary>RMS normalization over the last axis: y = x / sqrt(mean(x²)+eps) * weight (weight is [D]).</summary>
    public Tensor RmsNorm(Tensor x, Tensor weight, float eps)
    {
        int rank = x.Shape.Rank;
        if (rank == 0) throw new ArgumentException("RmsNorm requires rank ≥ 1.");
        int d = x.Shape[rank - 1];
        if (weight.Shape.Rank != 1 || weight.Shape[0] != d)
            throw new ArgumentException($"RmsNorm weight must be [{d}] to match the last axis of {x.Shape}.");

        var src = Materialize(x);
        var w = Materialize(weight);
        var dst = new float[src.Length];
        long rows = x.ElementCount / d;
        for (long r = 0; r < rows; r++)
        {
            long b = r * d;
            float sumSq = 0f;
            for (int j = 0; j < d; j++) { float v = src[b + j]; sumSq += v * v; }
            float inv = 1f / MathF.Sqrt(sumSq / d + eps);
            for (int j = 0; j < d; j++) dst[b + j] = src[b + j] * inv * w[j];
        }
        return new Tensor(x.Shape, x.DType, Device, handle: dst);
    }

    /// <summary>SiLU / swish activation: y = x · sigmoid(x), elementwise.</summary>
    public Tensor Silu(Tensor x)
    {
        var src = Materialize(x);
        var dst = new float[src.Length];
        for (int i = 0; i < src.Length; i++) dst[i] = src[i] * SigmoidScalar(src[i]);
        return new Tensor(x.Shape, x.DType, Device, handle: dst);
    }

    /// <summary>
    /// Rotary position embedding (rotate-half / Llama convention) on the last axis (head dim, even).
    /// <paramref name="cos"/>/<paramref name="sin"/> are the duplicated tables, broadcastable to x's shape.
    /// </summary>
    public Tensor RotaryEmbedding(Tensor x, Tensor cos, Tensor sin)
    {
        int rank = x.Shape.Rank;
        if (rank == 0) throw new ArgumentException("RotaryEmbedding requires rank ≥ 1.");
        int d = x.Shape[rank - 1];
        if (d % 2 != 0) throw new ArgumentException($"RoPE head dim must be even; got {d}.");
        int half = d / 2;

        var src = Materialize(x);
        var c = Materialize(cos.BroadcastTo(x.Shape));
        var s = Materialize(sin.BroadcastTo(x.Shape));
        var dst = new float[src.Length];
        long rows = x.ElementCount / d;
        for (long r = 0; r < rows; r++)
        {
            long b = r * d;
            for (int i = 0; i < d; i++)
            {
                float rotated = i < half ? -src[b + i + half] : src[b + i - half];
                dst[b + i] = src[b + i] * c[b + i] + rotated * s[b + i];
            }
        }
        return new Tensor(x.Shape, x.DType, Device, handle: dst);
    }

    // --- Indexing & loss (ticket S1-3) ---

    public Tensor Gather(Tensor table, int[] ids)
    {
        int vocab = table.Shape[0], dim = table.Shape[1];
        var src = Materialize(table);
        var dst = new float[(long)ids.Length * dim];
        for (int i = 0; i < ids.Length; i++)
        {
            int id = ids[i];
            if ((uint)id >= (uint)vocab) throw new ArgumentOutOfRangeException(nameof(ids), id, $"id out of range [0,{vocab}).");
            Array.Copy(src, (long)id * dim, dst, (long)i * dim, dim);
        }
        return new Tensor(new Shape(ids.Length, dim), table.DType, Device, handle: dst);
    }

    public Tensor ScatterAddRows(Tensor rows, int[] ids, int rowCount)
    {
        int dim = rows.Shape[rows.Shape.Rank - 1];
        var src = Materialize(rows);
        var dst = new float[(long)rowCount * dim]; // zero-initialized → unused rows get no gradient
        for (int i = 0; i < ids.Length; i++)
        {
            int id = ids[i];
            if ((uint)id >= (uint)rowCount) throw new ArgumentOutOfRangeException(nameof(ids), id, $"id out of range [0,{rowCount}).");
            long rowBase = (long)id * dim, srcBase = (long)i * dim;
            for (int j = 0; j < dim; j++) dst[rowBase + j] += src[srcBase + j];
        }
        return new Tensor(new Shape(rowCount, dim), rows.DType, Device, handle: dst);
    }

    public Tensor CrossEntropy(Tensor logits, int[] targets, int ignoreIndex)
    {
        int n = logits.Shape[0], vocab = logits.Shape[1];
        var src = Materialize(logits);
        double total = 0;
        int count = 0;
        for (int i = 0; i < n; i++)
        {
            int t = targets[i];
            if (t == ignoreIndex) continue;
            if ((uint)t >= (uint)vocab) throw new ArgumentOutOfRangeException(nameof(targets), t, $"target out of range [0,{vocab}).");
            long b = (long)i * vocab;
            float max = float.NegativeInfinity;
            for (int v = 0; v < vocab; v++) if (src[b + v] > max) max = src[b + v];
            double sumExp = 0;
            for (int v = 0; v < vocab; v++) sumExp += Math.Exp(src[b + v] - max);
            total += (max + Math.Log(sumExp)) - src[b + t]; // logsumexp − logit[target] = −log softmax[target]
            count++;
        }
        float loss = count > 0 ? (float)(total / count) : 0f;
        return new Tensor(Shape.Scalar, logits.DType, Device, handle: new[] { loss });
    }

    public Tensor CrossEntropyGrad(Tensor logits, int[] targets, int ignoreIndex)
    {
        int n = logits.Shape[0], vocab = logits.Shape[1];
        var src = Materialize(logits);
        int count = 0;
        for (int i = 0; i < n; i++) if (targets[i] != ignoreIndex) count++;
        float invCount = count > 0 ? 1f / count : 0f;

        var dst = new float[(long)n * vocab]; // ignored rows stay zero
        for (int i = 0; i < n; i++)
        {
            int t = targets[i];
            if (t == ignoreIndex) continue;
            if ((uint)t >= (uint)vocab) throw new ArgumentOutOfRangeException(nameof(targets), t, $"target out of range [0,{vocab}).");
            long b = (long)i * vocab;
            float max = float.NegativeInfinity;
            for (int v = 0; v < vocab; v++) max = MathF.Max(max, src[b + v]);
            float sumExp = 0;
            for (int v = 0; v < vocab; v++) sumExp += MathF.Exp(src[b + v] - max);
            float invSum = 1f / sumExp;
            for (int v = 0; v < vocab; v++) dst[b + v] = MathF.Exp(src[b + v] - max) * invSum * invCount; // softmax/count
            dst[b + t] -= invCount; // − onehot/count
        }
        return new Tensor(new Shape(n, vocab), logits.DType, Device, handle: dst);
    }

    // --- Structural ---

    /// <summary>
    /// Concatenates two tensors along <paramref name="axis"/> (negative counts from the end). All other dimensions
    /// must match. Strided inputs are gathered (materialized) first, so a transposed/sliced view concatenates
    /// correctly — the KV cache appends a transposed value view this way.
    /// </summary>
    public Tensor Cat(Tensor a, Tensor b, int axis)
    {
        int rank = a.Shape.Rank;
        if (b.Shape.Rank != rank)
            throw new ArgumentException($"Cat rank mismatch: {a.Shape} vs {b.Shape}.");
        int ax = axis < 0 ? axis + rank : axis;
        if ((uint)ax >= (uint)rank)
            throw new ArgumentOutOfRangeException(nameof(axis), axis, $"Axis out of range for shape {a.Shape}.");
        for (int i = 0; i < rank; i++)
            if (i != ax && a.Shape[i] != b.Shape[i])
                throw new ArgumentException($"Cat: dimensions must match except on axis {ax}: {a.Shape} vs {b.Shape}.");

        long outer = 1;
        for (int i = 0; i < ax; i++) outer *= a.Shape[i];
        long inner = 1;
        for (int i = ax + 1; i < rank; i++) inner *= a.Shape[i];
        int la = a.Shape[ax], lb = b.Shape[ax], lt = la + lb;

        var sa = Materialize(a);
        var sb = Materialize(b);
        var dst = new float[outer * lt * inner];
        for (long o = 0; o < outer; o++)
        {
            Array.Copy(sa, o * la * inner, dst, o * lt * inner, la * inner);
            Array.Copy(sb, o * lb * inner, dst, o * lt * inner + la * inner, lb * inner);
        }

        var outDims = new int[rank];
        for (int i = 0; i < rank; i++) outDims[i] = a.Shape[i];
        outDims[ax] = lt;
        return new Tensor(new Shape(outDims), a.DType, Device, handle: dst);
    }

    public void Dispose() { }
}
