using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using Xunit;

namespace ProjectAI.Tests;

public class MatMulTests
{
    private static float[] Host(CpuComputeBackend be, Tensor t)
    {
        var buf = new float[t.ElementCount];
        be.ToHost(t, buf);
        return buf;
    }

    /// <summary>
    /// Independent triple-loop reference GEMM. Crucially it accumulates in <c>double</c>, so it is a
    /// genuine numerical oracle for the float32 backend (not a bit-identical mirror of its loop order).
    /// </summary>
    private static double[] Naive(float[] a, float[] b, int m, int k, int n, bool transposeB)
    {
        var o = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double acc = 0d;
                for (int p = 0; p < k; p++)
                    acc += (double)a[i * k + p] * (transposeB ? b[j * k + p] : b[p * n + j]);
                o[i * n + j] = acc;
            }
        return o;
    }

    /// <summary>Compares a float32 result against a double reference with a k-aware relative tolerance.</summary>
    private static void AssertClose(double[] expected, float[] actual, double rtol = 1e-4, double atol = 1e-5)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double e = expected[i];
            double diff = Math.Abs(e - actual[i]);
            Assert.True(diff <= atol + rtol * Math.Abs(e),
                $"index {i}: expected {e}, got {actual[i]} (|diff|={diff}, tol={atol + rtol * Math.Abs(e)})");
        }
    }

    private static float[] Block(float[] arr, int index, int size) => arr[(index * size)..((index + 1) * size)];

    private static float[] RandomFlat(Random rng, int count)
    {
        var v = new float[count];
        for (int i = 0; i < count; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    [Fact]
    public void MatMul_KnownSmallProduct()
    {
        using var be = new CpuComputeBackend();
        var a = be.FromHost([1, 2, 3, 4, 5, 6], new Shape(2, 3), DType.F32); // [[1,2,3],[4,5,6]]
        var b = be.FromHost([7, 8, 9, 10, 11, 12], new Shape(3, 2), DType.F32); // [[7,8],[9,10],[11,12]]

        var c = be.MatMul(a, b);

        Assert.Equal(new Shape(2, 2), c.Shape);
        AssertClose([58, 64, 139, 154], Host(be, c));
    }

    [Fact]
    public void MatMul_Identity_IsNoOp()
    {
        using var be = new CpuComputeBackend();
        var a = be.FromHost([1, 2, 3, 4, 5, 6], new Shape(2, 3), DType.F32);
        var id = be.FromHost([1, 0, 0, 0, 1, 0, 0, 0, 1], new Shape(3, 3), DType.F32);

        AssertClose([1, 2, 3, 4, 5, 6], Host(be, be.MatMul(a, id)));
    }

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(2, 3, 4)]
    [InlineData(7, 1, 9)]
    [InlineData(16, 16, 16)]
    [InlineData(13, 31, 17)]
    [InlineData(40, 256, 24)] // upper end of the S0-3 acceptance window (k≈256, deepest accumulation)
    [InlineData(96, 96, 96)]
    public void MatMul_RandomSizes_MatchDoubleReference(int m, int k, int n)
    {
        using var be = new CpuComputeBackend();
        var rng = new Random(m * 1000 + k * 31 + n);
        var a = RandomFlat(rng, m * k);
        var b = RandomFlat(rng, k * n);

        var c = be.MatMul(be.FromHost(a, new Shape(m, k), DType.F32), be.FromHost(b, new Shape(k, n), DType.F32));

        Assert.Equal(new Shape(m, n), c.Shape);
        AssertClose(Naive(a, b, m, k, n, transposeB: false), Host(be, c));
    }

    [Fact]
    public void MatMul_TransposeB_MatchesReference()
    {
        using var be = new CpuComputeBackend();
        var rng = new Random(7);
        int m = 4, k = 5, n = 3;
        var a = RandomFlat(rng, m * k);
        var bT = RandomFlat(rng, n * k); // b stored as [n, k]

        var c = be.MatMul(be.FromHost(a, new Shape(m, k), DType.F32),
                          be.FromHost(bT, new Shape(n, k), DType.F32), transposeB: true);

        Assert.Equal(new Shape(m, n), c.Shape);
        AssertClose(Naive(a, bT, m, k, n, transposeB: true), Host(be, c));
    }

    [Fact]
    public void MatMul_Batched_MatchesPerBatchReference()
    {
        using var be = new CpuComputeBackend();
        var rng = new Random(99);
        int batch = 2, m = 3, k = 4, n = 5;
        var a = RandomFlat(rng, batch * m * k);
        var b = RandomFlat(rng, batch * k * n);

        var c = be.MatMul(be.FromHost(a, new Shape(batch, m, k), DType.F32),
                          be.FromHost(b, new Shape(batch, k, n), DType.F32));

        Assert.Equal(new Shape(batch, m, n), c.Shape);
        var got = Host(be, c);
        for (int bi = 0; bi < batch; bi++)
            AssertClose(Naive(Block(a, bi, m * k), Block(b, bi, k * n), m, k, n, false), got[(bi * m * n)..((bi + 1) * m * n)]);
    }

    [Fact]
    public void MatMul_Batched_TransposeB()
    {
        using var be = new CpuComputeBackend();
        var rng = new Random(41);
        int batch = 3, m = 2, k = 4, n = 3;
        var a = RandomFlat(rng, batch * m * k);
        var bT = RandomFlat(rng, batch * n * k); // [batch, n, k]

        var c = be.MatMul(be.FromHost(a, new Shape(batch, m, k), DType.F32),
                          be.FromHost(bT, new Shape(batch, n, k), DType.F32), transposeB: true);

        Assert.Equal(new Shape(batch, m, n), c.Shape);
        var got = Host(be, c);
        for (int bi = 0; bi < batch; bi++)
            AssertClose(Naive(Block(a, bi, m * k), Block(bT, bi, n * k), m, k, n, true), got[(bi * m * n)..((bi + 1) * m * n)]);
    }

    [Fact]
    public void MatMul_BroadcastsBatchlessRightOperand()
    {
        using var be = new CpuComputeBackend();
        var rng = new Random(123);
        int batch = 2, m = 3, k = 4, n = 5;
        var a = RandomFlat(rng, batch * m * k);
        var b = RandomFlat(rng, k * n); // no batch dim → broadcast across batches

        var c = be.MatMul(be.FromHost(a, new Shape(batch, m, k), DType.F32), be.FromHost(b, new Shape(k, n), DType.F32));

        Assert.Equal(new Shape(batch, m, n), c.Shape);
        var got = Host(be, c);
        for (int bi = 0; bi < batch; bi++)
            AssertClose(Naive(Block(a, bi, m * k), b, m, k, n, false), got[(bi * m * n)..((bi + 1) * m * n)]);
    }

    [Fact]
    public void MatMul_BroadcastsBatchlessLeftOperand()
    {
        using var be = new CpuComputeBackend();
        var rng = new Random(321);
        int batch = 2, m = 3, k = 4, n = 5;
        var a = RandomFlat(rng, m * k); // no batch dim
        var b = RandomFlat(rng, batch * k * n);

        var c = be.MatMul(be.FromHost(a, new Shape(m, k), DType.F32), be.FromHost(b, new Shape(batch, k, n), DType.F32));

        Assert.Equal(new Shape(batch, m, n), c.Shape);
        var got = Host(be, c);
        for (int bi = 0; bi < batch; bi++)
            AssertClose(Naive(a, Block(b, bi, k * n), m, k, n, false), got[(bi * m * n)..((bi + 1) * m * n)]);
    }

    [Fact]
    public void MatMul_MutualBatchBroadcast_4D()
    {
        using var be = new CpuComputeBackend();
        var rng = new Random(555);
        int m = 2, k = 3, n = 4; // a:[2,1,m,k]  b:[1,3,k,n] → out:[2,3,m,n]
        var a = RandomFlat(rng, 2 * 1 * m * k);
        var b = RandomFlat(rng, 1 * 3 * k * n);

        var c = be.MatMul(be.FromHost(a, new Shape(2, 1, m, k), DType.F32),
                          be.FromHost(b, new Shape(1, 3, k, n), DType.F32));

        Assert.Equal(new Shape(2, 3, m, n), c.Shape);
        var got = Host(be, c);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
            {
                int outBatch = i * 3 + j;
                var expected = Naive(Block(a, i, m * k), Block(b, j, k * n), m, k, n, false);
                AssertClose(expected, got[(outBatch * m * n)..((outBatch + 1) * m * n)]);
            }
    }

    [Fact]
    public void MatMul_HandlesNonContiguousLeftInput()
    {
        using var be = new CpuComputeBackend();
        var raw = new float[12];
        for (int i = 0; i < 12; i++) raw[i] = i + 1;
        var a = be.FromHost(raw, new Shape(4, 3), DType.F32).Transpose(0, 1); // [3,4], strided
        Assert.False(a.IsContiguous);

        var bRaw = new[] { 1f, 0, 2, 0, 1, 0, 0, 0 }; // [4,2]
        var b = be.FromHost(bRaw, new Shape(4, 2), DType.F32);

        var c = be.MatMul(a, b);

        Assert.Equal(new Shape(3, 2), c.Shape);
        AssertClose(Naive(Host(be, a), bRaw, 3, 4, 2, false), Host(be, c));
    }

    [Fact]
    public void MatMul_HandlesNonContiguousRightInput()
    {
        using var be = new CpuComputeBackend();
        var rng = new Random(17);
        int m = 3, k = 4, n = 2;
        var a = RandomFlat(rng, m * k);
        var bRaw = RandomFlat(rng, n * k);
        var b = be.FromHost(bRaw, new Shape(n, k), DType.F32).Transpose(0, 1); // [k,n], strided
        Assert.False(b.IsContiguous);

        var c = be.MatMul(be.FromHost(a, new Shape(m, k), DType.F32), b);

        Assert.Equal(new Shape(m, n), c.Shape);
        AssertClose(Naive(a, Host(be, b), m, k, n, false), Host(be, c));
    }

    [Fact]
    public void MatMul_HandlesNonContiguousBatchedInput()
    {
        using var be = new CpuComputeBackend();
        var rng = new Random(202);
        int batch = 2, m = 3, k = 4, n = 5;
        // Build [batch, k, m] then transpose the last two axes → [batch, m, k], strided.
        var a0 = RandomFlat(rng, batch * k * m);
        var a = be.FromHost(a0, new Shape(batch, k, m), DType.F32).Transpose(1, 2);
        Assert.False(a.IsContiguous);
        var b = RandomFlat(rng, batch * k * n);

        var c = be.MatMul(a, be.FromHost(b, new Shape(batch, k, n), DType.F32));

        Assert.Equal(new Shape(batch, m, n), c.Shape);
        var aMat = Host(be, a); // logical [batch, m, k]
        var got = Host(be, c);
        for (int bi = 0; bi < batch; bi++)
            AssertClose(Naive(Block(aMat, bi, m * k), Block(b, bi, k * n), m, k, n, false), got[(bi * m * n)..((bi + 1) * m * n)]);
    }
}
