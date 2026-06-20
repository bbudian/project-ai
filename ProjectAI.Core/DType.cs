namespace ProjectAI.Core;

/// <summary>Numeric element type stored in a <see cref="Tensor"/>.</summary>
public enum DType
{
    F32,
    F16,
    BF16,
    I64,
    I32,
    U8,
    Bool
}

public static class DTypeExtensions
{
    /// <summary>Size in bytes of a single element of the given dtype.</summary>
    public static int SizeInBytes(this DType dtype) => dtype switch
    {
        DType.F32 or DType.I32 => 4,
        DType.F16 or DType.BF16 => 2,
        DType.I64 => 8,
        DType.U8 or DType.Bool => 1,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, null)
    };
}
