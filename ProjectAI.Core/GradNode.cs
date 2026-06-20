namespace ProjectAI.Core;

/// <summary>
/// A node in the reverse-mode autograd tape. Records the input tensors that fed an op and a closure
/// that, given the gradient w.r.t. the op's output, returns the gradient w.r.t. each input.
/// </summary>
public sealed class GradNode(string opName, IReadOnlyList<Tensor> inputs, Func<Tensor, IReadOnlyList<Tensor>> backward)
{
    public string OpName { get; } = opName;
    public IReadOnlyList<Tensor> Inputs { get; } = inputs;

    /// <summary>Maps the upstream gradient into gradients for each entry of <see cref="Inputs"/>.</summary>
    public Func<Tensor, IReadOnlyList<Tensor>> Backward { get; } = backward;
}
