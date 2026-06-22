namespace ProjectAI.Core;

/// <summary>
/// What a module needs to exist: the backend (to allocate parameters), the <see cref="Autograd"/> facade
/// (the only route to the tape), and the seeded <see cref="IRng"/> (reproducible init). Bundled so a
/// module constructor takes one argument and so the same context flows to child modules.
/// </summary>
public sealed record ParameterContext(IComputeBackend Backend, Autograd Ag, IRng Rng)
{
    /// <summary>Builds a context over a backend with a fresh autograd facade and a seeded RNG.</summary>
    public static ParameterContext Create(IComputeBackend backend, ulong seed) =>
        new(backend, new Autograd(backend), new PcgRng(seed));
}

/// <summary>
/// Base class for differentiable building blocks (analogous to torch.nn.Module). Owns named parameters
/// and child modules so an optimizer can enumerate a model's full parameter set from its root. Modules
/// build their forward through <see cref="Ag"/> (the autograd facade) so the tape is recorded; the
/// backend is used only to allocate parameters and constants.
/// </summary>
public abstract class Module
{
    private readonly Dictionary<string, Tensor> _parameters = new();
    private readonly Dictionary<string, Module> _children = new();

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
        var p = Backend.Allocate(shape, DType.F32); // zero-initialized
        p.RequiresGrad = true;
        using (GradMode.NoGrad()) init.Fill(p, Backend, Ctx.Rng);
        _parameters[name] = p;
        return p;
    }

    /// <summary>Registers an externally-created parameter (e.g. a tied or loaded weight).</summary>
    protected void RegisterParameter(string name, Tensor parameter) => _parameters[name] = parameter;

    /// <summary>Registers a child module and returns it for fluent assignment.</summary>
    protected T RegisterModule<T>(string name, T module) where T : Module
    {
        _children[name] = module;
        return module;
    }

    /// <summary>This module's parameters followed by those of all descendant modules.</summary>
    public IEnumerable<Tensor> Parameters()
    {
        foreach (var p in _parameters.Values) yield return p;
        foreach (var child in _children.Values)
            foreach (var p in child.Parameters())
                yield return p;
    }

    /// <summary>Parameters with dotted names (e.g. "block.0.ffn.gate.weight"); used by checkpoint IO (S1-4).</summary>
    public IEnumerable<(string Name, Tensor Param)> NamedParameters()
    {
        foreach (var (name, p) in _parameters) yield return (name, p);
        foreach (var (childName, child) in _children)
            foreach (var (name, p) in child.NamedParameters())
                yield return ($"{childName}.{name}", p);
    }

    /// <summary>Runs the forward computation, threading the cross-cutting <paramref name="ctx"/>.</summary>
    public abstract Tensor Forward(Tensor input, ForwardContext ctx);

    /// <summary>Convenience for leaf modules / callers that need no mask/positions/cache.</summary>
    public Tensor Forward(Tensor input) => Forward(input, ForwardContext.Inference());
}
