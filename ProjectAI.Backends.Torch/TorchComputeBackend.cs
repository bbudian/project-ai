using ProjectAI.Core;

namespace ProjectAI.Backends.Torch;

/// <summary>
/// libtorch-backed backend via TorchSharp: CUDA on Windows/RTX 4090, MPS on Apple Silicon.
/// This is the batteries-included baseline. Wire up TorchSharp 0.107.0 plus a platform libtorch
/// bundle at Stage 2 (see docs/BUILD_PLAN.md); until then every member is a contract placeholder.
/// </summary>
public sealed class TorchComputeBackend(Device device) : IComputeBackend
{
    public string Name => "torchsharp";
    public Device Device { get; } = device;

    public Tensor Allocate(Shape shape, DType dtype) => throw new NotImplementedException("Stage 2 — ticket S2-2.");
    public Tensor FromHost(ReadOnlySpan<float> data, Shape shape, DType dtype) => throw new NotImplementedException("Stage 2.");
    public void ToHost(Tensor source, Span<float> destination) => throw new NotImplementedException("Stage 2.");
    public void Copy(Tensor source, Tensor destination) => throw new NotImplementedException("Stage 2.");
    public Tensor Add(Tensor a, Tensor b) => throw new NotImplementedException("Stage 2.");
    public Tensor Mul(Tensor a, Tensor b) => throw new NotImplementedException("Stage 2.");
    public Tensor AddScalar(Tensor a, float scalar) => throw new NotImplementedException("Stage 2.");
    public Tensor MulScalar(Tensor a, float scalar) => throw new NotImplementedException("Stage 2.");
    public Tensor MatMul(Tensor a, Tensor b, bool transposeB = false) => throw new NotImplementedException("Stage 2.");
    public Tensor Softmax(Tensor x, int axis) => throw new NotImplementedException("Stage 2.");
    public Tensor RmsNorm(Tensor x, Tensor weight, float eps) => throw new NotImplementedException("Stage 2.");
    public Tensor Silu(Tensor x) => throw new NotImplementedException("Stage 2.");
    public Tensor RotaryEmbedding(Tensor x, Tensor cos, Tensor sin) => throw new NotImplementedException("Stage 2.");
    public void Dispose() { }
}
