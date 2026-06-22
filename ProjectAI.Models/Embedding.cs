using ProjectAI.Core;

namespace ProjectAI.Models;

/// <summary>
/// Token embedding table [vocabSize, dim]. Maps integer token ids (carried as a float tensor) to their
/// embedding rows. The weight is exposed via <see cref="Weight"/> so an LM head can tie to it.
/// </summary>
public sealed class Embedding : Module
{
    private readonly Tensor _weight;

    public int VocabSize { get; }
    public int Dim { get; }
    /// <summary>The [vocabSize, dim] embedding matrix (shareable with a tied output projection).</summary>
    public Tensor Weight => _weight;

    public Embedding(ParameterContext ctx, int vocabSize, int dim, IInitializer? init = null) : base(ctx)
    {
        VocabSize = vocabSize;
        Dim = dim;
        _weight = Param("weight", new Shape(vocabSize, dim), init ?? Init.Normal(0f, 0.02f));
    }

    public override Tensor Forward(Tensor input, ForwardContext ctx)
    {
        var ids = ToIds(input);
        var rows = Ag.Embedding(_weight, ids); // [N, dim]
        if (input.Shape.Rank == 1) return rows; // [seq, dim]

        var outDims = new int[input.Shape.Rank + 1];
        for (int i = 0; i < input.Shape.Rank; i++) outDims[i] = input.Shape[i];
        outDims[^1] = Dim;
        return Ag.Reshape(rows, outDims); // [.., seq, dim]
    }

    /// <summary>Reads token ids (stored as floats) into an int array.</summary>
    private int[] ToIds(Tensor input)
    {
        var host = new float[input.ElementCount];
        Backend.ToHost(input, host);
        var ids = new int[host.Length];
        for (int i = 0; i < host.Length; i++) ids[i] = (int)MathF.Round(host[i]);
        return ids;
    }
}
