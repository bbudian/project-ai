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
    case "demo":
        RunStage0Demo(backend);
        break;
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
        Console.WriteLine("Usage: projectai <demo|generate|train|convert>");
        Console.WriteLine("  demo   Stage 0 milestone: fit y = Wx + b with our own autograd + AdamW.");
        break;
}

// Stage 0 milestone (BUILD_PLAN.md §6): train a tiny linear model to near-zero loss using only our
// Tensor + autograd + AdamW on the CPU backend — the first time the engine visibly *learns*.
static void RunStage0Demo(IComputeBackend be)
{
    var ag = new Autograd(be);
    var rng = new Random(0);
    const int n = 128, din = 3, dout = 1, steps = 200;
    float[] wTrue = [2f, -3f, 0.5f];
    const float bTrue = 1.5f;

    // Synthetic, noise-free data: Y = X · wTrue + bTrue.
    var xBuf = new float[n * din];
    var yBuf = new float[n * dout];
    for (int i = 0; i < n; i++)
    {
        float dot = 0f;
        for (int j = 0; j < din; j++)
        {
            float v = (float)(rng.NextDouble() * 2 - 1);
            xBuf[i * din + j] = v;
            dot += v * wTrue[j];
        }
        yBuf[i] = dot + bTrue;
    }

    var x = be.FromHost(xBuf, new Shape(n, din), DType.F32);
    var y = be.FromHost(yBuf, new Shape(n, dout), DType.F32);
    var w = be.FromHost(new float[din * dout], new Shape(din, dout), DType.F32);
    var bias = be.FromHost(new float[dout], new Shape(dout), DType.F32);
    w.RequiresGrad = true;
    bias.RequiresGrad = true;

    var optimizer = new AdamW([w, bias], be, learningRate: 0.1f, weightDecay: 0f);

    Console.WriteLine($"Stage 0 demo — fitting y = Wx + b  (n={n}, in={din}) with hand-written autograd + AdamW");
    Console.WriteLine();
    var lossHost = new float[1];
    for (int step = 1; step <= steps; step++)
    {
        optimizer.ZeroGrad();
        var pred = ag.Add(ag.MatMul(x, w), bias);   // [n, dout]
        var diff = ag.Sub(pred, y);
        var loss = ag.Mean(ag.Mul(diff, diff));      // mean squared error
        loss.Backward();
        optimizer.Step();

        if (step == 1 || step % 20 == 0)
        {
            be.ToHost(loss, lossHost);
            Console.WriteLine($"  step {step,3}   loss {lossHost[0]:F6}");
        }
    }

    var wOut = new float[din];
    var bOut = new float[dout];
    be.ToHost(w, wOut);
    be.ToHost(bias, bOut);
    Console.WriteLine();
    Console.WriteLine($"learned W ≈ [{string.Join(", ", wOut.Select(v => v.ToString("F3")))}]   (true [2, -3, 0.5])");
    Console.WriteLine($"learned b ≈ {bOut[0]:F3}   (true 1.5)");
}
