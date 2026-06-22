using ProjectAI.Core;

namespace ProjectAI.Backends.Vulkan;

/// <summary>
/// Handwritten Vulkan compute backend via Silk.NET. This is where Ben's shader expertise applies
/// directly: GLSL/SPIR-V compute kernels for matmul, attention, norms, and elementwise ops.
/// On macOS the same Vulkan calls reach Metal through MoltenVK. Built out in Stage 2
/// (see docs/BUILD_PLAN.md); members are contract placeholders until then.
/// </summary>
public sealed class VulkanComputeBackend(Device device) : IComputeBackend
{
    public string Name => "vulkan";
    public Device Device { get; } = device;

    public Tensor Allocate(Shape shape, DType dtype) => throw new NotImplementedException("Stage 2 — ticket S2-5.");
    public Tensor FromHost(ReadOnlySpan<float> data, Shape shape, DType dtype) => throw new NotImplementedException("Stage 2.");
    public void ToHost(Tensor source, Span<float> destination) => throw new NotImplementedException("Stage 2.");
    public void Copy(Tensor source, Tensor destination) => throw new NotImplementedException("Stage 2.");
    public Tensor Add(Tensor a, Tensor b) => throw new NotImplementedException("Stage 2.");
    public Tensor Mul(Tensor a, Tensor b) => throw new NotImplementedException("Stage 2.");
    public Tensor Sub(Tensor a, Tensor b) => throw new NotImplementedException("Stage 2.");
    public Tensor Div(Tensor a, Tensor b) => throw new NotImplementedException("Stage 2.");
    public Tensor AddScalar(Tensor a, float scalar) => throw new NotImplementedException("Stage 2.");
    public Tensor MulScalar(Tensor a, float scalar) => throw new NotImplementedException("Stage 2.");
    public Tensor Sqrt(Tensor x) => throw new NotImplementedException("Stage 2.");
    public Tensor Sigmoid(Tensor x) => throw new NotImplementedException("Stage 2.");
    public Tensor MatMul(Tensor a, Tensor b, bool transposeB = false) => throw new NotImplementedException("Stage 2 — ticket S2-5.");
    public Tensor Sum(Tensor x, int axis, bool keepDims = false) => throw new NotImplementedException("Stage 2 — ticket S2-5.");
    public Tensor Mean(Tensor x, int axis, bool keepDims = false) => throw new NotImplementedException("Stage 2 — ticket S2-5.");
    public Tensor Max(Tensor x, int axis, bool keepDims = false) => throw new NotImplementedException("Stage 2 — ticket S2-5.");
    public Tensor Softmax(Tensor x, int axis) => throw new NotImplementedException("Stage 2.");
    public Tensor RmsNorm(Tensor x, Tensor weight, float eps) => throw new NotImplementedException("Stage 2.");
    public Tensor Silu(Tensor x) => throw new NotImplementedException("Stage 2.");
    public Tensor RotaryEmbedding(Tensor x, Tensor cos, Tensor sin) => throw new NotImplementedException("Stage 2.");
    public Tensor Gather(Tensor table, int[] ids) => throw new NotImplementedException("Stage 2.");
    public Tensor ScatterAddRows(Tensor rows, int[] ids, int rowCount) => throw new NotImplementedException("Stage 2.");
    public Tensor CrossEntropy(Tensor logits, int[] targets, int ignoreIndex) => throw new NotImplementedException("Stage 2.");
    public Tensor CrossEntropyGrad(Tensor logits, int[] targets, int ignoreIndex) => throw new NotImplementedException("Stage 2.");
    public void Dispose() { }
}
