using ProjectAI.Core;
using TorchSharp;
using TT = TorchSharp.torch.Tensor;

namespace ProjectAI.Backends.Torch;

/// <summary>
/// libtorch-backed <see cref="IComputeBackend"/> via TorchSharp: CUDA on Windows/RTX 4090, MPS on Apple Silicon,
/// CPU anywhere (ticket S2-2). The managed bindings compile everywhere; the native libtorch runtime is a separate
/// per-machine package (see the .csproj), so first use throws a clear load error if no bundle is installed.
///
/// <para>Bridge to our tensor model: <see cref="Tensor.Handle"/> holds a *contiguous* base torch tensor, while
/// our <see cref="Tensor"/> carries its own shape/strides/offset (view ops rewrite those without touching the
/// backend). So every op reconstructs the logical view with <c>as_strided</c> before computing — that reuses
/// libtorch's own striding and broadcasting instead of reimplementing them, and keeps results bit-comparable to
/// the CPU oracle (enforced by the S2-1 conformance suite).</para>
///
/// <para>Memory: each op runs inside a <c>DisposeScope</c> so its <c>as_strided</c> views and intermediates are
/// freed deterministically; only the contiguous result escapes (via <c>MoveToOuterDisposeScope</c>) into the
/// returned <see cref="Tensor.Handle"/>. That bounds per-op churn — without it a single transformer step leaks
/// thousands of native tensors and OOMs a GPU. The *result* tensors (the live autograd graph) are still released
/// only by the GC/finalizer between steps; deterministic handle lifetime + pooling is ticket S2-3.</para>
/// </summary>
public sealed class TorchComputeBackend : IComputeBackend
{
    private readonly torch.Device _device;

    public TorchComputeBackend(Device device)
    {
        Device = device;
        _device = new torch.Device(ToTorchDeviceType(device.Kind), device.Index);
    }

    public string Name => "torchsharp";
    public Device Device { get; }

    /// <summary>
    /// Probes whether this backend can actually run on the given device on THIS machine — that a native libtorch
    /// bundle is installed and, for CUDA/Metal, that the device is present. The composition root uses this to offer
    /// only the backends that will work (and to explain why one won't). Never throws.
    /// </summary>
    public static bool IsAvailable(DeviceKind kind, out string? reason)
    {
        try
        {
            using var probe = torch.zeros(1); // forces the native libtorch load
            switch (kind)
            {
                case DeviceKind.Cpu:
                    reason = null;
                    return true;
                case DeviceKind.Cuda when torch.cuda.is_available():
                    reason = null;
                    return true;
                case DeviceKind.Cuda:
                    reason = "no CUDA device/driver detected";
                    return false;
                case DeviceKind.Metal when OperatingSystem.IsMacOS():
                    reason = null;
                    return true;
                case DeviceKind.Metal:
                    reason = "Metal/MPS is only available on macOS";
                    return false;
                default:
                    reason = $"unsupported device {kind}";
                    return false;
            }
        }
        catch (Exception ex)
        {
            reason = $"libtorch runtime not installed ({ex.Message})";
            return false;
        }
    }

    // --- Allocation & host transfer ---

    public Tensor Allocate(Shape shape, DType dtype)
        => new(shape, dtype, Device, handle: torch.zeros(Sizes(shape), ToScalarType(dtype), _device));

    public Tensor FromHost(ReadOnlySpan<float> data, Shape shape, DType dtype)
    {
        using var scope = torch.NewDisposeScope();
        var sized = torch.tensor(data.ToArray(), device: _device).reshape(Sizes(shape)); // float32, then reshape
        if (dtype != DType.F32) sized = sized.to_type(ToScalarType(dtype));
        return new Tensor(shape, dtype, Device, handle: sized.contiguous().MoveToOuterDisposeScope());
    }

    public void ToHost(Tensor source, Span<float> destination)
    {
        using var scope = torch.NewDisposeScope();
        var dense = Bind(source).contiguous().to_type(torch.ScalarType.Float32).cpu();
        float[] data = dense.data<float>().ToArray();
        data.AsSpan(0, destination.Length).CopyTo(destination);
    }

    public void Copy(Tensor source, Tensor destination)
    {
        using var scope = torch.NewDisposeScope();
        Bind(destination).copy_(Bind(source)); // writes into destination's base storage; the views are scope-disposed
    }

    // --- Elementwise (libtorch broadcasts following the same NumPy rules) ---

    public Tensor Add(Tensor a, Tensor b) => Run(a.DType, () => Bind(a) + Bind(b));
    public Tensor Mul(Tensor a, Tensor b) => Run(a.DType, () => Bind(a) * Bind(b));
    public Tensor Sub(Tensor a, Tensor b) => Run(a.DType, () => Bind(a) - Bind(b));
    public Tensor Div(Tensor a, Tensor b) => Run(a.DType, () => Bind(a) / Bind(b));
    public Tensor AddScalar(Tensor a, float scalar) => Run(a.DType, () => Bind(a) + scalar);
    public Tensor MulScalar(Tensor a, float scalar) => Run(a.DType, () => Bind(a) * scalar);
    public Tensor Sqrt(Tensor x) => Run(x.DType, () => Bind(x).sqrt());
    public Tensor Sigmoid(Tensor x) => Run(x.DType, () => Bind(x).sigmoid());

    // --- Linear algebra ---

    public Tensor MatMul(Tensor a, Tensor b, bool transposeB = false)
        => Run(a.DType, () => torch.matmul(Bind(a), transposeB ? Bind(b).transpose(-2, -1) : Bind(b)));

    // --- Reductions (negative axis counts from the end, matching the oracle) ---

    public Tensor Sum(Tensor x, int axis, bool keepDims = false) => Run(x.DType, () => Bind(x).sum(axis, keepdim: keepDims));
    public Tensor Mean(Tensor x, int axis, bool keepDims = false) => Run(x.DType, () => Bind(x).mean([axis], keepdim: keepDims));
    public Tensor Max(Tensor x, int axis, bool keepDims = false) => Run(x.DType, () => Bind(x).max(axis, keepDims).values);

    // --- Transformer primitives ---

    public Tensor Softmax(Tensor x, int axis) => Run(x.DType, () => Bind(x).softmax(axis));

    public Tensor RmsNorm(Tensor x, Tensor weight, float eps) => Run(x.DType, () =>
    {
        var t = Bind(x);
        var ms = (t * t).mean([-1L], keepdim: true);           // mean square over the last axis
        return t / (ms + eps).sqrt() * Bind(weight);           // weight [d] broadcasts over the last axis
    });

    public Tensor Silu(Tensor x) => Run(x.DType, () =>
    {
        var t = Bind(x);
        return t * t.sigmoid();
    });

    public Tensor RotaryEmbedding(Tensor x, Tensor cos, Tensor sin) => Run(x.DType, () =>
    {
        int d = x.Shape[x.Shape.Rank - 1];
        if (d % 2 != 0) throw new ArgumentException($"RoPE head dim must be even; got {d}.");
        long half = d / 2;
        var t = Bind(x);
        var rotated = torch.cat([t.narrow(-1, half, half).neg(), t.narrow(-1, 0, half)], dim: -1); // rotate-half [-x2, x1]
        return t * Bind(cos) + rotated * Bind(sin);
    });

    // --- Indexing & loss ---

    public Tensor Gather(Tensor table, int[] ids)
        => Run(table.DType, () => Bind(table).index_select(0, LongTensor(ids)));

    public Tensor ScatterAddRows(Tensor rows, int[] ids, int rowCount) => Run(rows.DType, () =>
    {
        int dim = rows.Shape[rows.Shape.Rank - 1];
        var dst = torch.zeros([rowCount, dim], ToScalarType(rows.DType), _device);
        dst.index_add_(0, LongTensor(ids), Bind(rows).contiguous(), 1); // repeated ids accumulate (alpha = 1)
        return dst;
    });

    public Tensor CrossEntropy(Tensor logits, int[] targets, int ignoreIndex)
    {
        using var scope = torch.NewDisposeScope();
        var lg = Bind(logits);                                  // [n, vocab]
        var tgt = LongTensor(targets);                          // [n]
        var valid = tgt.ne(ignoreIndex);
        var validF = valid.to_type(torch.ScalarType.Float32);
        var safe = torch.where(valid, tgt, torch.zeros_like(tgt));            // clamp ignored → row 0 (masked out)
        var perRow = (lg.logsumexp(1) - lg.gather(1, safe.unsqueeze(1)).squeeze(1)) * validF; // −log softmax[target]
        var count = validF.sum();
        var loss = count.item<float>() > 0 ? perRow.sum() / count : torch.tensor(0f, device: _device);
        return new Tensor(Shape.Scalar, logits.DType, Device, handle: loss.contiguous().MoveToOuterDisposeScope());
    }

    public Tensor CrossEntropyGrad(Tensor logits, int[] targets, int ignoreIndex) => Run(logits.DType, () =>
    {
        int n = logits.Shape[0];
        var lg = Bind(logits);                                  // [n, vocab]
        var tgt = LongTensor(targets);                          // [n]
        var valid = tgt.ne(ignoreIndex);
        var validRows = valid.to_type(torch.ScalarType.Float32).unsqueeze(1);   // [n, 1]
        float count = valid.sum().to_type(torch.ScalarType.Float32).item<float>();
        float invCount = count > 0 ? 1f / count : 0f;

        var sm = lg.softmax(1);
        var safe = torch.where(valid, tgt, torch.zeros_like(tgt));
        var onehot = torch.zeros_like(sm);
        onehot.scatter_(1, safe.unsqueeze(1), torch.ones([n, 1L], torch.ScalarType.Float32, _device));
        // valid rows: (softmax − onehot)/count; ignored rows zeroed by validRows (their spurious col-0 onehot too).
        return (sm - onehot) * invCount * validRows;
    });

    // --- Structural ---

    // torch.cat materializes the (possibly strided) Bind views into one contiguous device tensor — no host
    // round-trip and no synchronization, which is the whole point for the per-token KV-cache growth path.
    public Tensor Cat(Tensor a, Tensor b, int axis) => Run(a.DType, () => torch.cat([Bind(a), Bind(b)], dim: axis));

    // --- Transient memory scoping (S2-3): free a step's native tensors deterministically instead of via the GC ---

    // A DisposeScope tracks every torch tensor created while it's open; disposing frees all of them at once,
    // except those moved out via KeepAlive. Op results are already moved to the enclosing scope (see Run), so an
    // enclosing step scope here captures the whole step's graph and releases it the moment the step ends.
    public IDisposable BeginScope() => torch.NewDisposeScope();

    public void KeepAlive(Tensor tensor)
    {
        if (tensor.Handle is TT t) t.MoveToOuterDisposeScope(); // promote out of the current scope so it survives
    }

    public void Release(Tensor tensor)
    {
        if (tensor.Handle is TT t) t.Dispose(); // free the native/VRAM storage now (superseded optimizer state)
    }

    public void Dispose() { }

    // --- bridge helpers ---

    private static TT Base(Tensor t) => (TT)(t.Handle
        ?? throw new InvalidOperationException("tensor has no torch handle (was it produced by a different backend?)."));

    // Reconstruct the (possibly strided / broadcast) logical view our metadata describes over the base storage.
    private static TT Bind(Tensor t) => Base(t).as_strided(Sizes(t.Shape), t.Strides, t.Offset);

    // Runs an op inside a DisposeScope so the as_strided views + intermediates are freed deterministically; only
    // the contiguous result escapes into the returned Tensor.Handle.
    private Tensor Run(DType dtype, Func<TT> compute)
    {
        using var scope = torch.NewDisposeScope();
        var result = compute().contiguous();
        result.MoveToOuterDisposeScope();
        long[] s = result.shape;
        var dims = new int[s.Length];
        for (int i = 0; i < s.Length; i++) dims[i] = (int)s[i];
        return new Tensor(new Shape(dims), dtype, Device, handle: result);
    }

    private TT LongTensor(int[] values)
    {
        var longs = new long[values.Length];
        for (int i = 0; i < values.Length; i++) longs[i] = values[i];
        return torch.tensor(longs, device: _device);
    }

    private static long[] Sizes(Shape shape)
    {
        var s = new long[shape.Rank];
        for (int i = 0; i < s.Length; i++) s[i] = shape[i];
        return s;
    }

    private static torch.ScalarType ToScalarType(DType dtype) => dtype switch
    {
        DType.F32 => torch.ScalarType.Float32,
        DType.F16 => torch.ScalarType.Float16,
        DType.BF16 => torch.ScalarType.BFloat16,
        DType.I64 => torch.ScalarType.Int64,
        DType.I32 => torch.ScalarType.Int32,
        DType.U8 => torch.ScalarType.Byte,
        DType.Bool => torch.ScalarType.Bool,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, "unmapped dtype"),
    };

    private static DeviceType ToTorchDeviceType(DeviceKind kind) => kind switch
    {
        DeviceKind.Cpu => DeviceType.CPU,
        DeviceKind.Cuda => DeviceType.CUDA,
        DeviceKind.Metal => DeviceType.MPS,
        DeviceKind.Vulkan => throw new NotSupportedException("TorchSharp has no Vulkan device; use ProjectAI.Backends.Vulkan."),
        _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "unmapped device kind"),
    };
}
