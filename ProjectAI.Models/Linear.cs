using ProjectAI.Core;

namespace ProjectAI.Models;

/// <summary>
/// A bias-free linear projection y = x · Wᵀ (Llama-style). The weight is stored <c>[outDim, inDim]</c>
/// and applied with the fused transpose, so a row of W is one output feature. Operates on the last axis
/// of any rank ≥ 2 input by flattening the leading dims.
/// </summary>
public sealed class Linear : Module
{
    private readonly Tensor _weight;
    private readonly int _inDim;
    private readonly int _outDim;

    public Linear(ParameterContext ctx, int inDim, int outDim, IInitializer? init = null) : base(ctx)
    {
        _inDim = inDim;
        _outDim = outDim;
        _weight = Param("weight", new Shape(outDim, inDim), init ?? Init.Normal(0f, 0.02f));
    }

    public override Tensor Forward(Tensor input, ForwardContext ctx)
    {
        if (input.Shape[input.Shape.Rank - 1] != _inDim)
            throw new ArgumentException($"Linear expected last dim {_inDim}, got {input.Shape}.");

        if (input.Shape.Rank == 2)
            return Ag.MatMul(input, _weight, transposeB: true); // [rows, outDim]

        // Flatten leading dims → matmul → restore leading dims. Reshape needs a contiguous tensor, so a
        // strided input (e.g. a transposed attention output) is densified first; Ag.Contiguous is a no-op
        // when already contiguous.
        int rows = (int)(input.ElementCount / _inDim);
        var lead = input.Shape.Dimensions[..^1].ToArray();
        var flat = Ag.Reshape(Ag.Contiguous(input), rows, _inDim);
        var y = Ag.MatMul(flat, _weight, transposeB: true); // [rows, outDim]
        var outDims = new int[lead.Length + 1];
        Array.Copy(lead, outDims, lead.Length);
        outDims[^1] = _outDim;
        return Ag.Reshape(y, outDims);
    }
}
