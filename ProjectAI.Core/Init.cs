namespace ProjectAI.Core;

/// <summary>A parameter-initialization strategy (ticket S0-8). Fills an already-allocated tensor in place.</summary>
public interface IInitializer
{
    void Fill(Tensor target, IComputeBackend backend, IRng rng);
}

/// <summary>
/// Standard parameter initializers. Gaussian strategies draw from the supplied <see cref="IRng"/> so a
/// fixed seed reproduces identical weights. For a 2-D weight stored <c>[out, in]</c>, fan-in is the
/// trailing-dim product and fan-out is the leading dim.
/// </summary>
public static class Init
{
    public static readonly IInitializer Zeros = new ZerosInit();
    public static readonly IInitializer Ones = new OnesInit();
    public static IInitializer Normal(float mean = 0f, float std = 0.02f) => new NormalInit(mean, std);
    public static IInitializer Xavier(float gain = 1f) => new XavierInit(gain);
    public static IInitializer Kaiming(float gain = 1.41421356f /* sqrt(2) */) => new KaimingInit(gain);

    private static (long fanIn, long fanOut) Fans(Shape shape)
    {
        if (shape.Rank <= 1)
        {
            long n = shape.Rank == 0 ? 1 : shape[0];
            return (n, n);
        }
        long fanOut = shape[0];
        long fanIn = 1;
        for (int i = 1; i < shape.Rank; i++) fanIn *= shape[i];
        return (fanIn, fanOut);
    }

    private static void FillGaussian(Tensor target, IComputeBackend backend, IRng rng, float mean, float std)
    {
        var buffer = new float[target.ElementCount];
        for (int i = 0; i < buffer.Length; i++) buffer[i] = mean + std * rng.NextGaussian();
        backend.Copy(backend.FromHost(buffer, target.Shape, target.DType), target);
    }

    private sealed class ZerosInit : IInitializer
    {
        public void Fill(Tensor target, IComputeBackend backend, IRng rng) { } // Allocate already zeroes
    }

    private sealed class OnesInit : IInitializer
    {
        public void Fill(Tensor target, IComputeBackend backend, IRng rng) =>
            backend.Copy(backend.AddScalar(target, 1f), target);
    }

    private sealed class NormalInit(float mean, float std) : IInitializer
    {
        public void Fill(Tensor target, IComputeBackend backend, IRng rng) =>
            FillGaussian(target, backend, rng, mean, std);
    }

    private sealed class XavierInit(float gain) : IInitializer
    {
        public void Fill(Tensor target, IComputeBackend backend, IRng rng)
        {
            var (fanIn, fanOut) = Fans(target.Shape);
            FillGaussian(target, backend, rng, 0f, gain * MathF.Sqrt(2f / (fanIn + fanOut)));
        }
    }

    private sealed class KaimingInit(float gain) : IInitializer
    {
        public void Fill(Tensor target, IComputeBackend backend, IRng rng)
        {
            var (fanIn, _) = Fans(target.Shape);
            FillGaussian(target, backend, rng, 0f, gain / MathF.Sqrt(fanIn));
        }
    }
}
