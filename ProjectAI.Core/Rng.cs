namespace ProjectAI.Core;

/// <summary>
/// Seedable deterministic random source (ticket S0-7). Uses an explicit algorithm (PCG-XSH-RR 64/32)
/// rather than <see cref="System.Random"/> so a seed reproduces the same stream byte-for-byte across
/// runs, processes, and platforms — the basis for reproducible parameter initialization.
/// </summary>
public interface IRng
{
    uint NextUInt32();
    /// <summary>Uniform float in [0, 1).</summary>
    float NextFloat();
    /// <summary>Standard normal sample (Box–Muller).</summary>
    float NextGaussian();
}

/// <summary>PCG-XSH-RR 64/32 generator with a Box–Muller Gaussian. Deterministic and platform-stable.</summary>
public sealed class PcgRng : IRng
{
    private const ulong Multiplier = 6364136223846793005UL;
    private ulong _state;
    private readonly ulong _inc;
    private bool _hasSpare;
    private float _spare;

    public PcgRng(ulong seed, ulong stream = 0xDA3E39CB94B95BDBUL)
    {
        _inc = (stream << 1) | 1UL;
        _state = 0UL;
        NextUInt32();
        _state += seed;
        NextUInt32();
    }

    public uint NextUInt32()
    {
        ulong old = _state;
        _state = old * Multiplier + _inc;
        uint xorshifted = (uint)(((old >> 18) ^ old) >> 27);
        int rot = (int)(old >> 59);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    public float NextFloat() => (NextUInt32() >> 8) * (1.0f / (1u << 24)); // 24-bit mantissa → [0,1)

    public float NextGaussian()
    {
        if (_hasSpare)
        {
            _hasSpare = false;
            return _spare;
        }
        float u1 = MathF.Max(NextFloat(), 1e-7f); // avoid log(0)
        float u2 = NextFloat();
        float magnitude = MathF.Sqrt(-2f * MathF.Log(u1));
        _spare = magnitude * MathF.Sin(2f * MathF.PI * u2);
        _hasSpare = true;
        return magnitude * MathF.Cos(2f * MathF.PI * u2);
    }
}
