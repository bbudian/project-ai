using System.Diagnostics.CodeAnalysis;
using ProjectAI.Core;

namespace ProjectAI.Formats;

/// <summary>A mapping from parameter name to tensor, as loaded from a checkpoint file.</summary>
public sealed class StateDict
{
    private readonly Dictionary<string, Tensor> _tensors = new();

    public Tensor this[string name]
    {
        get => _tensors[name];
        set => _tensors[name] = value;
    }

    public IReadOnlyDictionary<string, Tensor> Tensors => _tensors;
    public int Count => _tensors.Count;

    public bool TryGet(string name, [MaybeNullWhen(false)] out Tensor tensor) =>
        _tensors.TryGetValue(name, out tensor);
}

/// <summary>Reads model weights from a file into device tensors via the supplied backend.</summary>
public interface IWeightLoader
{
    StateDict Load(string path, IComputeBackend backend);
}

/// <summary>HuggingFace .safetensors: JSON header + raw little-endian tensor bytes (zero-copy where possible).</summary>
public interface ISafetensorsLoader : IWeightLoader;

/// <summary>GGUF (llama.cpp) checkpoints, including quantized weight blocks.</summary>
public interface IGgufLoader : IWeightLoader;

public sealed class SafetensorsLoader : ISafetensorsLoader
{
    public StateDict Load(string path, IComputeBackend backend) => throw new NotImplementedException("ticket S1-4.");
}

public sealed class GgufLoader : IGgufLoader
{
    public StateDict Load(string path, IComputeBackend backend) => throw new NotImplementedException("ticket S1-5.");
}
