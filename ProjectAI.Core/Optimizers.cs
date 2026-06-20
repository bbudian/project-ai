namespace ProjectAI.Core;

/// <summary>A parameter optimizer. Operates on the tensors yielded by <see cref="Module.Parameters"/>.</summary>
public interface IOptimizer
{
    void Step();
    void ZeroGrad();
}

/// <summary>
/// Decoupled-weight-decay Adam (AdamW). First/second moment state is allocated lazily per parameter.
/// The update math runs against the active <see cref="IComputeBackend"/> and is implemented in Stage 0.
/// </summary>
public sealed class AdamW : IOptimizer
{
    private readonly IReadOnlyList<Tensor> _parameters;

    public AdamW(
        IReadOnlyList<Tensor> parameters,
        IComputeBackend backend,
        float learningRate = 1e-3f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float epsilon = 1e-8f,
        float weightDecay = 0.01f)
    {
        _parameters = parameters;
        Backend = backend;
        LearningRate = learningRate;
        Beta1 = beta1;
        Beta2 = beta2;
        Epsilon = epsilon;
        WeightDecay = weightDecay;
    }

    public IComputeBackend Backend { get; }
    public float LearningRate { get; set; }
    public float Beta1 { get; }
    public float Beta2 { get; }
    public float Epsilon { get; }
    public float WeightDecay { get; }

    public void Step() => throw new NotImplementedException("AdamW update step — ticket S0-5.");

    public void ZeroGrad()
    {
        foreach (var p in _parameters) p.Grad = null;
    }
}
