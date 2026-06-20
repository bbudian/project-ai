namespace ProjectAI.Core;

/// <summary>
/// Immutable tensor shape with row-major (C-contiguous) semantics. Holds dimensions only;
/// per-view strides and offset live on <see cref="Tensor"/>. Provides the index/stride/broadcast
/// math the rest of the system relies on.
/// </summary>
public readonly struct Shape : IEquatable<Shape>
{
    private readonly int[]? _dims;

    public Shape(params int[] dims)
    {
        ArgumentNullException.ThrowIfNull(dims);
        foreach (var d in dims)
            if (d < 0)
                throw new ArgumentOutOfRangeException(nameof(dims), d, "Dimensions must be non-negative.");
        _dims = dims;
    }

    /// <summary>The rank-0 (scalar) shape.</summary>
    public static Shape Scalar => new();

    public ReadOnlySpan<int> Dimensions => _dims is null ? ReadOnlySpan<int>.Empty : _dims;
    public int Rank => Dimensions.Length;
    public int this[int axis] => Dimensions[axis];

    /// <summary>Total number of elements (product of all dimensions; 1 for a scalar).</summary>
    public long ElementCount
    {
        get
        {
            long n = 1;
            foreach (var d in Dimensions) n *= d;
            return n;
        }
    }

    /// <summary>
    /// Row-major (C-contiguous) strides in <em>elements</em>: <c>strides[i]</c> is the product of all
    /// dimensions after axis <c>i</c>. Example: shape [2,3,4] -&gt; [12,4,1].
    /// </summary>
    public long[] ContiguousStrides()
    {
        var dims = Dimensions;
        var strides = new long[dims.Length];
        long acc = 1;
        for (int i = dims.Length - 1; i >= 0; i--)
        {
            strides[i] = acc;
            acc *= dims[i];
        }
        return strides;
    }

    /// <summary>Linear element offset of a multi-index into a C-contiguous buffer.</summary>
    public long RavelIndex(ReadOnlySpan<int> indices)
    {
        var dims = Dimensions;
        if (indices.Length != dims.Length)
            throw new ArgumentException($"Expected {dims.Length} indices for shape {this}, got {indices.Length}.", nameof(indices));

        long linear = 0, stride = 1;
        for (int i = dims.Length - 1; i >= 0; i--)
        {
            int idx = indices[i];
            if ((uint)idx >= (uint)dims[i])
                throw new IndexOutOfRangeException($"Index {idx} is out of range for axis {i} (size {dims[i]}).");
            linear += idx * stride;
            stride *= dims[i];
        }
        return linear;
    }

    /// <summary>Inverse of <see cref="RavelIndex"/> for a C-contiguous buffer.</summary>
    public int[] UnravelIndex(long linear)
    {
        if (linear < 0 || linear >= ElementCount)
            throw new ArgumentOutOfRangeException(nameof(linear), linear, $"Out of range for shape {this} ({ElementCount} elements).");

        var dims = Dimensions;
        var idx = new int[dims.Length];
        for (int i = dims.Length - 1; i >= 0; i--)
        {
            idx[i] = (int)(linear % dims[i]);
            linear /= dims[i];
        }
        return idx;
    }

    /// <summary>NumPy-style broadcast of two shapes. Throws if the shapes are not broadcast-compatible.</summary>
    public Shape BroadcastWith(Shape other) =>
        TryBroadcastWith(other, out var result)
            ? result
            : throw new ArgumentException($"Shapes {this} and {other} are not broadcast-compatible.");

    /// <summary>NumPy-style broadcast: align trailing dimensions; each pair must be equal or one must be 1.</summary>
    public bool TryBroadcastWith(Shape other, out Shape result)
    {
        var a = Dimensions;
        var b = other.Dimensions;
        int n = Math.Max(a.Length, b.Length);
        var dims = new int[n];
        for (int i = 0; i < n; i++)
        {
            int da = i < a.Length ? a[a.Length - 1 - i] : 1;
            int db = i < b.Length ? b[b.Length - 1 - i] : 1;
            int d;
            if (da == db) d = da;
            else if (da == 1) d = db;
            else if (db == 1) d = da;
            else { result = default; return false; }
            dims[n - 1 - i] = d;
        }
        result = new Shape(dims);
        return true;
    }

    public bool Equals(Shape other) => Dimensions.SequenceEqual(other.Dimensions);
    public override bool Equals(object? obj) => obj is Shape s && Equals(s);

    public override int GetHashCode()
    {
        var hc = new HashCode();
        foreach (var d in Dimensions) hc.Add(d);
        return hc.ToHashCode();
    }

    public override string ToString() => $"[{string.Join(", ", Dimensions.ToArray())}]";
}
