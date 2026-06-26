using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>
/// Backend conformance suite (ticket S2-1) — the safety net for every GPU backend. It runs each
/// <see cref="IComputeBackend"/> op on every candidate backend and asserts the result matches the CPU oracle
/// (<see cref="CpuComputeBackend"/>) within tolerance, so a backend that diverges can't be shipped. Adding a
/// backend (TorchSharp/CUDA, Vulkan) is one factory entry in <see cref="BackendFactories"/> — every op below is
/// then checked against the oracle with no new test code. (Today the only backend is the CPU oracle, which
/// conforms to itself; the value is the systematic op coverage + the harness ready for backend #2.)
/// </summary>
public class BackendConformanceTests
{
    // Cross-backend tolerance: tight enough to catch real op bugs, loose enough for float-order/fused-kernel
    // differences a GPU backend will have. CPU-vs-CPU is exact, so any positive tolerance passes today.
    private const float Atol = 1e-4f;
    private const float Rtol = 1e-4f;

    // The backends validated against the oracle. Add `() => new TorchComputeBackend()` etc. here.
    private static readonly Func<IComputeBackend>[] BackendFactories = [() => new CpuComputeBackend()];

    public static IEnumerable<object[]> OpNames() => Cases().Select(c => new object[] { c.Name });

    [Theory]
    [MemberData(nameof(OpNames))]
    public void EveryBackendMatchesTheCpuOracle(string opName)
    {
        var run = Cases().First(c => c.Name == opName).Run;
        using var oracle = new CpuComputeBackend();
        foreach (var factory in BackendFactories)
        {
            using var backend = factory();
            AssertMatchesOracle(backend, oracle, opName, run);
        }
    }

    /// <summary>
    /// Runs one op-case on <paramref name="candidate"/> and the oracle with identical inputs and asserts a match
    /// within tolerance. Shared with backend-specific suites (e.g. <c>TorchConformanceTests</c>) so a new backend
    /// reuses the exact same op coverage.
    /// </summary>
    internal static void AssertMatchesOracle(IComputeBackend candidate, IComputeBackend oracle, string opName, Func<IComputeBackend, float[]> run)
    {
        float[] expected = run(oracle);
        float[] got = run(candidate);
        Assert.True(got.Length == expected.Length, $"[{candidate.Name}] '{opName}': length {got.Length} != oracle {expected.Length}");
        for (int i = 0; i < expected.Length; i++)
        {
            float tol = Atol + Rtol * MathF.Abs(expected[i]);
            Assert.True(MathF.Abs(expected[i] - got[i]) <= tol,
                $"[{candidate.Name}] '{opName}' index {i}: {got[i]} vs oracle {expected[i]} (tol {tol})");
        }
    }

    // Each case builds its inputs deterministically (fixed seeds), runs one op, and returns the host result, so
    // the oracle and a candidate backend are fed identical data and their outputs can be compared directly.
    internal static IEnumerable<(string Name, Func<IComputeBackend, float[]> Run)> Cases()
    {
        // --- Elementwise + broadcasting ---
        yield return ("Add", be => Host(be, be.Add(T(be, 1, 2, 3, 4), T(be, 2, 2, 3, 4))));
        yield return ("Add[broadcast]", be => Host(be, be.Add(T(be, 1, 2, 3, 4), T(be, 2, 4))));
        yield return ("Sub", be => Host(be, be.Sub(T(be, 1, 2, 3, 4), T(be, 2, 2, 3, 4))));
        yield return ("Mul", be => Host(be, be.Mul(T(be, 1, 2, 3, 4), T(be, 2, 2, 3, 4))));
        yield return ("Mul[broadcast]", be => Host(be, be.Mul(T(be, 1, 2, 3, 4), T(be, 2, 1, 1, 4))));
        yield return ("Div", be => Host(be, be.Div(T(be, 1, 2, 3, 4), TPos(be, 2, 2, 3, 4))));
        yield return ("AddScalar", be => Host(be, be.AddScalar(T(be, 1, 2, 3, 4), 0.75f)));
        yield return ("MulScalar", be => Host(be, be.MulScalar(T(be, 1, 2, 3, 4), -1.5f)));
        yield return ("Sqrt", be => Host(be, be.Sqrt(TPos(be, 1, 2, 3, 4))));
        yield return ("Sigmoid", be => Host(be, be.Sigmoid(T(be, 1, 2, 3, 4))));

        // --- MatMul (2D, batched, transposeB) ---
        yield return ("MatMul[2D]", be => Host(be, be.MatMul(T(be, 1, 3, 4), T(be, 2, 4, 5))));
        yield return ("MatMul[batched]", be => Host(be, be.MatMul(T(be, 1, 2, 3, 4), T(be, 2, 2, 4, 5))));
        yield return ("MatMul[transposeB]", be => Host(be, be.MatMul(T(be, 1, 3, 4), T(be, 2, 5, 4), transposeB: true)));

        // --- Reductions ---
        yield return ("Sum[axis=-1]", be => Host(be, be.Sum(T(be, 1, 2, 3, 4), axis: -1)));
        yield return ("Sum[axis=0,keepDims]", be => Host(be, be.Sum(T(be, 1, 2, 3, 4), axis: 0, keepDims: true)));
        yield return ("Mean[axis=-1]", be => Host(be, be.Mean(T(be, 1, 2, 3, 4), axis: -1)));
        yield return ("Max[axis=1]", be => Host(be, be.Max(T(be, 1, 2, 3, 4), axis: 1)));

        // --- Transformer primitives ---
        yield return ("Softmax[axis=-1]", be => Host(be, be.Softmax(T(be, 1, 2, 3, 4), axis: -1)));
        yield return ("RmsNorm", be => Host(be, be.RmsNorm(T(be, 1, 2, 3, 4), T(be, 9, 4), 1e-5f)));
        yield return ("Silu", be => Host(be, be.Silu(T(be, 1, 2, 3, 4))));
        yield return ("RotaryEmbedding", be => Host(be, be.RotaryEmbedding(
            be.FromHost(Rand(1, 1 * 2 * 3 * 4), new Shape(1, 2, 3, 4), DType.F32),
            be.FromHost(Rand(2, 3 * 4), new Shape(3, 4), DType.F32),
            be.FromHost(Rand(3, 3 * 4), new Shape(3, 4), DType.F32))));

        // --- Strided / broadcast / offset views (the paths the real model exercises but the dense cases above
        //     don't: transposed RoPE inputs, sliced KV-cache offsets, the GQA size-1 broadcast batch axis). S2-2 review. ---
        yield return ("Softmax[transposed]", be => Host(be, be.Softmax(T(be, 1, 3, 4).Transpose(0, 1), axis: -1)));
        yield return ("Sum[sliced,offset]", be => Host(be, be.Sum(T(be, 1, 4, 5).Slice(0, 1, 2), axis: -1)));
        yield return ("Add[stride-0 broadcast view]", be => Host(be, be.Add(T(be, 1, 2, 3, 4), T(be, 2, 1, 1, 4).BroadcastTo(new Shape(2, 3, 4)))));
        yield return ("MatMul[batch-broadcast,GQA]", be => Host(be, be.MatMul(T(be, 1, 2, 2, 3, 4, 6), T(be, 2, 2, 2, 1, 5, 6), transposeB: true)));

        // --- Structural: concat (the on-device KV-cache growth path). Includes a transposed/strided second
        //     operand, mirroring how the value cache appends a [b,kvh,s,dh] transpose view. ---
        yield return ("Cat[axis=2]", be => Host(be, be.Cat(T(be, 1, 2, 3, 4, 5), T(be, 2, 2, 3, 6, 5), axis: 2)));
        yield return ("Cat[axis=-1]", be => Host(be, be.Cat(T(be, 3, 2, 3, 4), T(be, 4, 2, 3, 6), axis: -1)));
        yield return ("Cat[strided incoming]", be => Host(be, be.Cat(T(be, 5, 2, 3, 4, 5), T(be, 6, 2, 6, 3, 5).Transpose(1, 2), axis: 2)));

        // --- Indexing & loss ---
        yield return ("Gather", be => Host(be, be.Gather(T(be, 1, 6, 4), [0, 2, 5, 2])));
        yield return ("ScatterAddRows", be => Host(be, be.ScatterAddRows(T(be, 1, 4, 4), [0, 2, 5, 2], rowCount: 6)));
        yield return ("CrossEntropy", be => Host(be, be.CrossEntropy(T(be, 1, 4, 5), [1, 3, 0, 2], ignoreIndex: -100)));
        yield return ("CrossEntropy[ignore]", be => Host(be, be.CrossEntropy(T(be, 1, 4, 5), [1, -100, 0, 2], ignoreIndex: -100)));
        yield return ("CrossEntropyGrad", be => Host(be, be.CrossEntropyGrad(T(be, 1, 4, 5), [1, -100, 0, 2], ignoreIndex: -100)));
    }

    // --- helpers: deterministic input tensors ---

    private static Tensor T(IComputeBackend be, int seed, params int[] dims)
        => be.FromHost(Rand(seed, Product(dims)), new Shape(dims), DType.F32);

    private static Tensor TPos(IComputeBackend be, int seed, params int[] dims)
    {
        var data = Rand(seed, Product(dims));
        for (int i = 0; i < data.Length; i++) data[i] = MathF.Abs(data[i]) + 0.1f; // strictly positive (for Sqrt/Div)
        return be.FromHost(data, new Shape(dims), DType.F32);
    }

    private static float[] Rand(int seed, int n)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static int Product(int[] dims)
    {
        int p = 1;
        foreach (int d in dims) p *= d;
        return p;
    }

    private static float[] Host(IComputeBackend be, Tensor t)
    {
        var buf = new float[t.ElementCount];
        be.ToHost(t, buf);
        return buf;
    }
}
