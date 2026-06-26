namespace ProjectAI.Core;

/// <summary>
/// What a module needs to exist: the backend (to allocate parameters), the <see cref="Autograd"/> facade
/// (the only route to the tape), and the seeded <see cref="IRng"/> (reproducible init). Bundled so a
/// module constructor takes one argument and so the same context flows to child modules.
/// </summary>
public sealed record ParameterContext(IComputeBackend Backend, Autograd Ag, IRng Rng)
{
    /// <summary>
    /// Precision of the model's parameters and constants (RoPE tables, masks). Default F32; set to BF16/F16 to
    /// build a half-precision model that fits ~2x more on the GPU (ticket S3-1). Activations follow the params'
    /// dtype, since the forward ops infer their output dtype from their inputs.
    /// </summary>
    public DType ComputeDType { get; init; } = DType.F32;

    /// <summary>Builds a context over a backend with a fresh autograd facade and a seeded RNG (F32 params).</summary>
    public static ParameterContext Create(IComputeBackend backend, ulong seed) =>
        new(backend, new Autograd(backend), new PcgRng(seed));

    /// <summary>Builds a context with a chosen parameter precision (e.g. BF16 for half-precision inference).</summary>
    public static ParameterContext Create(IComputeBackend backend, ulong seed, DType computeDType) =>
        new(backend, new Autograd(backend), new PcgRng(seed)) { ComputeDType = computeDType };
}

/// <summary>
/// Base class for differentiable building blocks (analogous to torch.nn.Module). Owns named parameters
/// and child modules so an optimizer can enumerate a model's full parameter set from its root. Modules
/// build their forward through <see cref="Ag"/> (the autograd facade) so the tape is recorded; the
/// backend is used only to allocate parameters and constants.
/// </summary>
public abstract class Module
{
    // Dictionaries give O(1) lookup / duplicate-name detection; the parallel key lists pin a documented,
    // deterministic enumeration order (registration order) that Parameters()/NamedParameters() rely on so
    // checkpoint IO and AdamW state stay portable across builds — never depend on Dictionary iteration order.
    private readonly Dictionary<string, Tensor> _parameters = new();
    private readonly List<string> _parameterOrder = new();
    private readonly Dictionary<string, Module> _children = new();
    private readonly List<string> _childOrder = new();

    protected Module(ParameterContext ctx) => Ctx = ctx;

    protected ParameterContext Ctx { get; }
    /// <summary>The autograd facade — the only sanctioned way for a module to build differentiable ops.</summary>
    protected Autograd Ag => Ctx.Ag;
    /// <summary>The backend — for allocating parameters/constants only (never for differentiable ops).</summary>
    protected IComputeBackend Backend => Ctx.Backend;

    /// <summary>
    /// Allocates, initializes (under no-grad), and registers a parameter in one call. This is the sole
    /// birthplace of a parameter tensor, so each parameter object is created exactly once — which the
    /// optimizer relies on (AdamW keys its moment state on the parameter reference).
    /// <para>Initialization draws from the context's shared <see cref="IRng"/> sequentially, so a fixed
    /// seed reproduces identical weights and sibling parameters are distinct — but the values depend on
    /// <em>construction order</em>. Reordering how a model's modules/parameters are built is a breaking
    /// change for a given seed.</para>
    /// </summary>
    protected Tensor Param(string name, Shape shape, IInitializer init)
    {
        var p = Backend.Allocate(shape, Ctx.ComputeDType); // zero-initialized, in the model's precision (S3-1)
        p.RequiresGrad = true;
        using (GradMode.NoGrad()) init.Fill(p, Backend, Ctx.Rng);
        if (!_parameters.ContainsKey(name)) _parameterOrder.Add(name);
        _parameters[name] = p;
        return p;
    }

    /// <summary>Registers an externally-created parameter (e.g. a tied or loaded weight).</summary>
    protected void RegisterParameter(string name, Tensor parameter)
    {
        if (!_parameters.ContainsKey(name)) _parameterOrder.Add(name);
        _parameters[name] = parameter;
    }

    /// <summary>Registers a child module and returns it for fluent assignment.</summary>
    protected T RegisterModule<T>(string name, T module) where T : Module
    {
        if (!_children.ContainsKey(name)) _childOrder.Add(name);
        _children[name] = module;
        return module;
    }

    /// <summary>
    /// This module's parameters followed by those of all descendant modules.
    /// <para>Order is <em>stable and deterministic</em>: this module's parameters in registration order,
    /// then each child module (in registration order) recursively. It is a pure function of how the model
    /// is constructed — independent of <see cref="Dictionary{TKey,TValue}"/> iteration — so the optimizer's
    /// reference-keyed state and the checkpoint's positional layout stay aligned across processes/builds.</para>
    /// </summary>
    public IEnumerable<Tensor> Parameters()
    {
        foreach (var name in _parameterOrder) yield return _parameters[name];
        foreach (var childName in _childOrder)
            foreach (var p in _children[childName].Parameters())
                yield return p;
    }

    /// <summary>
    /// Parameters with dotted names (e.g. "block.0.ffn.gate.weight"); used by checkpoint IO (S1-4).
    /// <para>Emitted in the same stable registration order as <see cref="Parameters"/> (this module's
    /// parameters, then each child recursively), so a checkpoint written by one build loads positionally
    /// into a freshly-constructed model of the same config regardless of dictionary iteration order.</para>
    /// </summary>
    public IEnumerable<(string Name, Tensor Param)> NamedParameters()
    {
        foreach (var name in _parameterOrder) yield return (name, _parameters[name]);
        foreach (var childName in _childOrder)
            foreach (var (name, p) in _children[childName].NamedParameters())
                yield return ($"{childName}.{name}", p);
    }

    /// <summary>Runs the forward computation, threading the cross-cutting <paramref name="ctx"/>.</summary>
    public abstract Tensor Forward(Tensor input, ForwardContext ctx);

    /// <summary>Convenience for leaf modules / callers that need no mask/positions/cache.</summary>
    public Tensor Forward(Tensor input) => Forward(input, ForwardContext.Inference());
}
