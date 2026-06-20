namespace ProjectAI.Core;

/// <summary>
/// Base class for differentiable building blocks (analogous to torch.nn.Module).
/// Owns named parameters and composes child modules so an optimizer can enumerate the full
/// parameter set of a model from its root.
/// </summary>
public abstract class Module
{
    private readonly Dictionary<string, Tensor> _parameters = new();
    private readonly Dictionary<string, Module> _children = new();

    protected void RegisterParameter(string name, Tensor parameter) => _parameters[name] = parameter;
    protected void RegisterModule(string name, Module module) => _children[name] = module;

    /// <summary>Enumerates this module's parameters followed by those of all descendant modules.</summary>
    public IEnumerable<Tensor> Parameters()
    {
        foreach (var p in _parameters.Values) yield return p;
        foreach (var child in _children.Values)
            foreach (var p in child.Parameters())
                yield return p;
    }

    /// <summary>Runs the forward computation for this module.</summary>
    public abstract Tensor Forward(Tensor input);
}
