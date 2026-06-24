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

    /// <summary>Optional schedule mapping the (1-based) step number to a learning rate; overrides <see cref="LearningRate"/>.</summary>
    public Func<int, float>? LearningRateSchedule { get; set; }

    // Per-parameter first/second moment estimates and timestep, allocated lazily on first use. The
    // timestep is per-parameter (not global) so a parameter whose first gradient arrives late still
    // gets the correct first-step bias correction (t=1), matching reference AdamW.
    private readonly Dictionary<Tensor, (Tensor M, Tensor V, int T)> _state = new(ReferenceEqualityComparer.Instance);
    private int _step;

    /// <summary>Number of <see cref="Step"/> calls so far.</summary>
    public int StepCount => _step;

    /// <summary>A per-parameter moment snapshot (first/second moment + timestep), for checkpoint save/restore.</summary>
    public readonly record struct MomentState(Tensor M, Tensor V, int T);

    /// <summary>Reads the moment state for a parameter; false if the parameter has not been stepped yet.</summary>
    public bool TryGetState(Tensor parameter, out MomentState state)
    {
        if (_state.TryGetValue(parameter, out var s) && s.M is not null)
        {
            state = new MomentState(s.M, s.V, s.T);
            return true;
        }
        state = default;
        return false;
    }

    /// <summary>Restores the moment state for a parameter (used when resuming from a checkpoint).</summary>
    public void LoadState(Tensor parameter, MomentState state) => _state[parameter] = (state.M, state.V, state.T);

    /// <summary>Restores the global step counter (so a resumed run continues the LR schedule).</summary>
    public void SetStepCount(int step) => _step = step;

    /// <summary>
    /// One AdamW update: bias-corrected first/second moments with <em>decoupled</em> weight decay
    /// (the decay term acts on the parameter, not through the moment estimates). Runs entirely on the
    /// backend; the update itself must not build an autograd graph, so it is wrapped in <c>no_grad</c>.
    /// </summary>
    public void Step()
    {
        _step++;
        float lr = LearningRateSchedule?.Invoke(_step) ?? LearningRate;

        using var _ = GradMode.NoGrad();
        foreach (var p in _parameters)
        {
            if (p.Grad is null) continue;
            var g = p.Grad;

            _state.TryGetValue(p, out var s);
            s.M ??= Backend.Allocate(p.Shape, p.DType); // zero-initialized
            s.V ??= Backend.Allocate(p.Shape, p.DType);
            int t = s.T + 1;                            // per-parameter timestep

            // m = β1·m + (1-β1)·g ;  v = β2·v + (1-β2)·g²
            var m = Backend.Add(Backend.MulScalar(s.M, Beta1), Backend.MulScalar(g, 1f - Beta1));
            var v = Backend.Add(Backend.MulScalar(s.V, Beta2), Backend.MulScalar(Backend.Mul(g, g), 1f - Beta2));
            _state[p] = (m, v, t);

            float biasCorrection1 = 1f - MathF.Pow(Beta1, t);
            float biasCorrection2 = 1f - MathF.Pow(Beta2, t);
            var mHat = Backend.MulScalar(m, 1f / biasCorrection1);
            var vHat = Backend.MulScalar(v, 1f / biasCorrection2);
            var adaptive = Backend.Div(mHat, Backend.AddScalar(Backend.Sqrt(vHat), Epsilon));

            // θ ← θ - lr·(m̂/(√v̂+ε)) - lr·λ·θ   (decoupled weight decay)
            var delta = Backend.Add(Backend.MulScalar(adaptive, lr), Backend.MulScalar(p, lr * WeightDecay));
            Backend.Copy(Backend.Sub(p, delta), p);
        }
    }

    public void ZeroGrad()
    {
        foreach (var p in _parameters) p.Grad = null;
    }
}
