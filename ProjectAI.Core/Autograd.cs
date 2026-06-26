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

    /// <summary>
    /// Batched matrix multiply over the trailing two axes, y = a · b (or a · bᵀ when <paramref name="transposeB"/>),
    /// with NumPy batch-dim broadcasting. The transposeB form fuses the weight transpose for <c>Linear</c>
    /// (weights stored [out, in]). The backward transposes the last two axes and folds broadcasted batch dims
    /// back to each operand via <see cref="ReduceGradToShape"/> (this is what makes GQA's shared-KV gradient sum).
    /// </summary>
    public Tensor MatMul(Tensor a, Tensor b, bool transposeB = false)
    {
        if (a.Shape.Rank < 2 || b.Shape.Rank < 2)
            throw new ArgumentException($"MatMul requires rank ≥ 2 operands; got {a.Shape} and {b.Shape}.");

        var y = Backend.MatMul(a, b, transposeB);
        if (!transposeB)
            return Record(y, "matmul", [a, b], g =>
            [
                ReduceGradToShape(Backend.MatMul(g, b, transposeB: true), a.Shape), // grad_a = g · bᵀ
                ReduceGradToShape(Backend.MatMul(LastT(a), g), b.Shape),            // grad_b = aᵀ · g
            ]);

        // y = a · bᵀ, with b's trailing axes [n, k]: grad_a = g · b ; grad_b = gᵀ · a.
        return Record(y, "matmul_tb", [a, b], g =>
        [
            ReduceGradToShape(Backend.MatMul(g, b), a.Shape),        // grad_a = g · b
            ReduceGradToShape(Backend.MatMul(LastT(g), a), b.Shape), // grad_b = gᵀ · a
        ]);
    }

    /// <summary>Transposes the trailing two axes (the matrix axes), leaving batch axes in place.</summary>
    private static Tensor LastT(Tensor t) => t.Transpose(t.Shape.Rank - 2, t.Shape.Rank - 1);

    /// <summary>
    /// Returns a dense, contiguous copy of <paramref name="x"/> (a no-op if it is already contiguous and
    /// zero-offset). Needed before <see cref="Reshape"/> of a strided view (e.g. a transposed attention
    /// output). The copy preserves logical order, so the backward is the identity.
    /// </summary>
    public Tensor Contiguous(Tensor x)
    {
        if (x is { IsContiguous: true, Offset: 0 }) return x;
        var dst = Backend.Allocate(x.Shape, x.DType);
        Backend.Copy(x, dst); // gathers the strided view into dense row-major order
        return Record(dst, "contiguous", [x], g => [g]);
    }

    /// <summary>Sum over all elements, producing a scalar. Backward: grad_x = g (the scalar) broadcast to x.</summary>
    public Tensor SumAll(Tensor x)
    {
        var y = x;
        for (int axis = x.Shape.Rank - 1; axis >= 0; axis--) y = Backend.Sum(y, axis); // → scalar; strided-safe
        return Record(y, "sum_all", [x], g =>
            [Backend.AddScalar(Backend.Allocate(x.Shape, x.DType), ToScalar(g))]); // d(sum)/d(x_i) = 1, so fill with g
    }

    /// <summary>
    /// Gradient checkpointing (ticket S3-2). Runs <paramref name="segment"/> on <paramref name="input"/> WITHOUT
    /// building a graph — so its intermediate activations are NOT retained — then records a node that RE-RUNS the
    /// segment during backward to regenerate the graph on demand, trading a recompute for the activation memory.
    /// <para>The backward recomputes from a detached copy of the input (so it doesn't extend the outer graph) with
    /// grad forced on, then builds the surrogate scalar Σ(recomputed ⊙ upstream): its gradient w.r.t. the recomputed
    /// output is exactly <c>upstream</c>, so a nested <see cref="Tensor.Backward"/> propagates that grad through the
    /// local graph — yielding the input gradient and accumulating the segment's parameter gradients identically to
    /// the non-checkpointed path (verified by a gradient-check test).</para>
    /// <para>CALLER CONTRACT: this promotes the recomputed grads exactly ONE scope level (out of the recompute
    /// scope into the caller's enclosing scope). When the caller runs <see cref="Tensor.Backward"/> inside its own
    /// <see cref="IComputeBackend.BeginScope"/> — as the trainer does per micro-batch — it MUST, after Backward,
    /// <see cref="IComputeBackend.KeepAlive"/> every model parameter's grad to promote it the rest of the way; the
    /// trainer does this over the full parameter set. Backward with no enclosing scope needs nothing extra.</para>
    /// <para>The whole recompute runs in its own <see cref="IComputeBackend.BeginScope"/> so the regenerated
    /// activations are freed as soon as this segment's grads are extracted (only the input + the listed
    /// <paramref name="parameters"/> gradients are kept) — without that, the per-step S2-3 scope would hold every
    /// segment's recompute alive until the step ends, defeating the memory saving. <paramref name="parameters"/>
    /// must list exactly the grad-requiring tensors the segment touches (e.g. <c>block.Parameters()</c>).</para>
    /// </summary>
    public Tensor Checkpoint(Func<Tensor, Tensor> segment, Tensor input, IReadOnlyList<Tensor> parameters)
    {
        Tensor output;
        using (GradMode.NoGrad())
        {
            // Scope the forward too, so this segment's intermediate activations are freed immediately — only the
            // (small) block output is kept. Without this, the no-grad forward still piles every block's activations
            // into the enclosing per-step scope, and checkpointing saves nothing on the forward pass.
            using var forwardScope = Backend.BeginScope();
            output = segment(input);   // forward value only — no graph
            Backend.KeepAlive(output); // the block output survives; its intermediates are freed at scope close
        }

        return Record(output, "checkpoint", [input], upstream =>
        {
            using var scope = Backend.BeginScope(); // free this segment's recompute as soon as its grads are out
            var leaf = DetachLeaf(input);           // a fresh leaf with the input's values; its grad becomes grad-wrt-input
            Tensor recomputed;
            using (GradMode.Enabled()) recomputed = segment(leaf); // rebuild the local graph from the detached input

            SumAll(Mul(recomputed, upstream)).Backward(); // seeds `recomputed` with `upstream`; fills leaf + param grads

            var inputGrad = leaf.Grad ?? Backend.Allocate(input.Shape, input.DType);
            Backend.KeepAlive(inputGrad);                                       // survives the recompute scope…
            foreach (var p in parameters) if (p.Grad is not null) Backend.KeepAlive(p.Grad); // …as do this segment's param grads
            return [inputGrad];
        });
    }

    // A detached copy of x as a grad-requiring leaf (no GradFn), so recomputing a segment on it doesn't link back
    // into the outer graph; the leaf's accumulated gradient is the gradient with respect to the original input.
    private Tensor DetachLeaf(Tensor x)
    {
        var copy = Backend.Allocate(x.Shape, x.DType);
        Backend.Copy(x, copy);
        copy.Backend = Backend;
        copy.RequiresGrad = true;
        return copy;
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

    // --- Indexing & loss (ticket S1-3) ---

    /// <summary>Embedding lookup: gathers rows of <paramref name="weight"/> ([vocab, dim]) by id. Backward
    /// scatter-adds the upstream gradient into the used rows (ids are constants, not differentiated).</summary>
    public Tensor Embedding(Tensor weight, int[] ids)
    {
        var y = Backend.Gather(weight, ids);
        int vocab = weight.Shape[0];
        return Record(y, "embedding", [weight], g => [Backend.ScatterAddRows(g, ids, vocab)]);
    }

    /// <summary>Mean cross-entropy of <paramref name="logits"/> ([N, vocab]) vs integer targets. Backward is
    /// (softmax − onehot)/validCount scaled by the upstream scalar gradient.</summary>
    public Tensor CrossEntropy(Tensor logits, int[] targets, int ignoreIndex)
    {
        var loss = Backend.CrossEntropy(logits, targets, ignoreIndex);
        return Record(loss, "cross_entropy", [logits], g =>
            [Backend.MulScalar(Backend.CrossEntropyGrad(logits, targets, ignoreIndex), ToScalar(g))]);
    }

    // --- Differentiable transformer primitives (forward via fused backend kernels, closed-form backward) ---

    /// <summary>Softmax along <paramref name="axis"/>. Backward: grad_x = y ⊙ (g - Σ_axis(g⊙y)).</summary>
    public Tensor Softmax(Tensor x, int axis)
    {
        var y = Backend.Softmax(x, axis);
        return Record(y, "softmax", [x], g =>
        {
            var dot = Backend.Sum(Backend.Mul(g, y), axis, keepDims: true);
            return [Backend.Mul(y, Backend.Sub(g, dot))];
        });
    }

    /// <summary>SiLU activation x·σ(x). Backward: grad_x = g ⊙ (s + y - s⊙y), s = σ(x).</summary>
    public Tensor Silu(Tensor x)
    {
        var y = Backend.Silu(x);
        return Record(y, "silu", [x], g =>
        {
            var s = Backend.Sigmoid(x);
            var derivative = Backend.Sub(Backend.Add(s, y), Backend.Mul(s, y));
            return [Backend.Mul(g, derivative)];
        });
    }

    /// <summary>
    /// RMSNorm over the last axis with learned per-feature scale. Backward (per the verified VJP):
    /// grad_x = inv·(g⊙w) - (inv²/D)·x·Σ(g⊙y); grad_w = Σ_leading(g⊙(x·inv)), inv = 1/sqrt(mean(x²)+eps).
    /// </summary>
    public Tensor RmsNorm(Tensor x, Tensor weight, float eps)
    {
        int lastAxis = x.Shape.Rank - 1;
        int d = x.Shape[lastAxis];
        var y = Backend.RmsNorm(x, weight, eps);
        return Record(y, "rmsnorm", [x, weight], g =>
        {
            var meanSq = Backend.Mean(Backend.Mul(x, x), lastAxis, keepDims: true);
            var denom = Backend.Sqrt(Backend.AddScalar(meanSq, eps));
            var ones = Backend.AddScalar(Backend.Allocate(denom.Shape, x.DType), 1f);
            var inv = Backend.Div(ones, denom);                                 // [.., 1]
            var dot = Backend.Sum(Backend.Mul(g, y), lastAxis, keepDims: true); // [.., 1]

            var direct = Backend.Mul(Backend.Mul(g, weight), inv);              // inv·(g⊙w)
            var coef = Backend.MulScalar(Backend.Mul(inv, inv), 1f / d);        // inv²/D
            var center = Backend.Mul(Backend.Mul(coef, x), dot);               // (inv²/D)·x·dot
            var gradX = Backend.Sub(direct, center);

            var gradW = ReduceGradToShape(Backend.Mul(g, Backend.Mul(x, inv)), weight.Shape);
            return [gradX, gradW];
        });
    }

    /// <summary>
    /// Rotary position embedding (rotate-half). cos/sin are constants. Since the forward is an
    /// orthogonal rotation, the backward is the same op with sin negated (the inverse rotation).
    /// </summary>
    public Tensor RotaryEmbedding(Tensor x, Tensor cos, Tensor sin)
    {
        var negSin = Backend.MulScalar(sin, -1f);
        var y = Backend.RotaryEmbedding(x, cos, sin);
        return Record(y, "rope", [x], g => [Backend.RotaryEmbedding(g, cos, negSin)]);
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
        return Record(y, "reshape", [x], g =>
        {
            // The upstream grad may be a non-contiguous view (e.g. from a Transpose backward feeding a
            // reshape, as in attention's head merge); densify it before the metadata-only reshape.
            if (g is not { IsContiguous: true, Offset: 0 })
            {
                var dense = Backend.Allocate(g.Shape, g.DType);
                Backend.Copy(g, dense);
                g = dense;
            }
            return [g.Reshape(original)];
        });
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
