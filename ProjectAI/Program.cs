using ProjectAI.Backends.Cpu;
using ProjectAI.Core;

Console.WriteLine("ProjectAI — local, hand-written AI runtime (.NET 10)");
Console.WriteLine();

// Composition root. Stage 2 adds TorchComputeBackend (CUDA/MPS) and VulkanComputeBackend
// behind this same interface — selected here by config/flags without touching call sites.
using IComputeBackend backend = new CpuComputeBackend();
Console.WriteLine($"Active backend: {backend.Name} on {backend.Device}");
Console.WriteLine();

var command = args.Length > 0 ? args[0] : "help";
switch (command)
{
    case "generate":
        Console.WriteLine("[generate] text generation — implemented in Stage 1. See docs/BUILD_PLAN.md.");
        break;
    case "train":
        Console.WriteLine("[train] training loop — implemented in Stage 1. See docs/BUILD_PLAN.md.");
        break;
    case "convert":
        Console.WriteLine("[convert] safetensors / GGUF import — implemented in Stage 1. See docs/BUILD_PLAN.md.");
        break;
    default:
        Console.WriteLine("Usage: projectai <generate|train|convert>");
        break;
}
