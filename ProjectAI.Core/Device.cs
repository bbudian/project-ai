namespace ProjectAI.Core;

public enum DeviceKind { Cpu, Cuda, Metal, Vulkan }

/// <summary>Identifies a compute device: a kind plus an ordinal index (e.g. cuda:0).</summary>
public readonly record struct Device(DeviceKind Kind, int Index = 0)
{
    public static Device Cpu => new(DeviceKind.Cpu);
    public override string ToString() => $"{Kind}:{Index}";
}
