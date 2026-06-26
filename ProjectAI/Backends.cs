using ProjectAI.Backends.Cpu;
using ProjectAI.Backends.Torch;
using ProjectAI.Core;

// The compute-backend catalog + factory. This is the one place that names concrete backends — the composition
// root is allowed to (nothing below the IComputeBackend seam is). It maps friendly ids (cpu, torch:cuda, …) to
// backends, reports which are actually available on this machine so the UI offers only those, and resolves the
// --backend / --device CLI flags to an id.
internal static class Backends
{
    private sealed record Entry(string Id, string Label, DeviceKind Kind, bool IsTorch);

    private static readonly Entry[] Catalog =
    [
        new("cpu",         "CPU (reference)", DeviceKind.Cpu,   IsTorch: false),
        new("torch:cpu",   "CPU (libtorch)",  DeviceKind.Cpu,   IsTorch: true),
        new("torch:cuda",  "GPU (CUDA)",      DeviceKind.Cuda,  IsTorch: true),
        new("torch:metal", "GPU (Metal)",     DeviceKind.Metal, IsTorch: true),
    ];

    public const string DefaultId = "cpu";

    /// <summary>Creates the backend for an id. Throws if its native runtime/device isn't available on this machine.</summary>
    public static IComputeBackend Create(string id)
    {
        var entry = Find(id);
        if (!entry.IsTorch) return new CpuComputeBackend();
        var backend = new TorchComputeBackend(new Device(entry.Kind));
        backend.ToHost(backend.Allocate(new Shape(1), DType.F32), new float[1]); // force native init → fail fast & clearly
        return backend;
    }

    /// <summary>The full catalog tagged with per-machine availability (cpu always; torch options gated on libtorch + the device).</summary>
    public static IReadOnlyList<BackendStatus> Available()
    {
        var list = new List<BackendStatus>(Catalog.Length);
        foreach (var e in Catalog)
        {
            bool available;
            string? reason;
            if (!e.IsTorch) { available = true; reason = null; }
            else available = TorchComputeBackend.IsAvailable(e.Kind, out reason);
            list.Add(new BackendStatus(e.Id, e.Label, available, reason));
        }
        return list;
    }

    public static bool IsKnown(string id) => Catalog.Any(e => e.Id == id);

    /// <summary>Maps the --backend / --device flags to a catalog id (cpu, torch:cpu, torch:cuda, torch:metal).</summary>
    public static string ResolveId(string? backend, string? device)
    {
        backend = (backend ?? "cpu").ToLowerInvariant();
        device = device?.ToLowerInvariant();
        return backend switch
        {
            "cpu" when device is null or "cpu" => "cpu",
            "cpu" => throw new ArgumentException($"the cpu backend only runs on --device cpu (got '{device}')"),
            "torch" => device switch
            {
                null or "cpu" => "torch:cpu",
                "cuda" => "torch:cuda",
                "metal" or "mps" => "torch:metal",
                _ => throw new ArgumentException($"unknown --device '{device}' (use cpu|cuda|metal)"),
            },
            _ => throw new ArgumentException($"unknown --backend '{backend}' (use cpu|torch)"),
        };
    }

    public static string LabelOf(string id) => Catalog.FirstOrDefault(e => e.Id == id)?.Label ?? id;

    private static Entry Find(string id) => Catalog.FirstOrDefault(e => e.Id == id)
        ?? throw new ArgumentException($"unknown backend id '{id}' (known: {string.Join(", ", Catalog.Select(e => e.Id))})");
}

internal sealed record BackendStatus(string Id, string Label, bool Available, string? Reason);
