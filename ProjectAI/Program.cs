using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using ProjectAI.Models;
using ProjectAI.Tokenizers;

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
    case "train":
        RunLlmDemo(backend, args.Length > 1 ? string.Join(' ', args[1..]) : "roses are ");
        break;
    case "generate":
        Console.WriteLine("[generate] standalone generation needs checkpoint load/save (ticket S1-11).");
        Console.WriteLine("For now, `train [prompt]` trains a tiny model and generates in one run.");
        break;
    case "convert":
        Console.WriteLine("[convert] safetensors / GGUF import — ticket S1-4/S1-5. See docs/BUILD_PLAN.md.");
        break;
    default:
        Console.WriteLine("Usage: projectai <demo|train|generate|convert>");
        Console.WriteLine("  demo          Stage 0 milestone: fit y = Wx + b with our own autograd + AdamW.");
        Console.WriteLine("  train [prompt] Train a tiny LLaMA on a built-in corpus, then generate from [prompt].");
        break;
}

// Stage 1 milestone: train a tiny Llama-style transformer from scratch on a small corpus, then generate.
// Everything below — embedding, RoPE, GQA attention, SwiGLU, RMSNorm, cross-entropy, AdamW — is hand-written.
static void RunLlmDemo(IComputeBackend be, string prompt)
{
    // A few short, independent sentences — each learned from position 0, so any of them can be continued
    // from its start. (RoPE encodes ABSOLUTE position, so a tiny over-fit model memorizes position→token;
    // training each sentence from position 0 is what lets an arbitrary in-corpus prompt continue correctly.)
    string[] lines = ["roses are red", "violets are blue", "sugar is sweet", "and so are you"];
    var tokenizer = new BpeTokenizer([]); // byte-level (vocab = 259)
    int vocab = tokenizer.VocabSize, pad = tokenizer.PadId, eos = tokenizer.EosId;

    // Tokenize each (append EOS), pad into a [sentences, T] batch; the loss ignores the padding.
    var seqs = lines.Select(line => { var e = tokenizer.Encode(line).ToList(); e.Add(eos); return e; }).ToList();
    int n = seqs.Count, T = seqs.Max(s => s.Count) - 1;
    var inputBuf = new float[n * T];
    var targetBuf = new float[n * T];
    for (int i = 0; i < n; i++)
    {
        var s = seqs[i];
        for (int t = 0; t < T; t++)
        {
            inputBuf[i * T + t] = t < s.Count ? s[t] : pad;
            targetBuf[i * T + t] = (t + 1) < s.Count ? s[t + 1] : pad; // pad targets are ignored
        }
    }
    var input = be.FromHost(inputBuf, new Shape(n, T), DType.F32);
    var target = be.FromHost(targetBuf, new Shape(n, T), DType.F32);

    var config = new ModelConfig
    {
        VocabSize = vocab, EmbeddingDim = 64, LayerCount = 2, HeadCount = 4, KvHeadCount = 2,
        FeedForwardHiddenDim = 256, MaxSequenceLength = 128,
    };
    var ctx = ParameterContext.Create(be, 0);
    var model = new LlamaModel(ctx, config);
    var optimizer = new AdamW(model.Parameters().ToList(), be, learningRate: 0.003f, weightDecay: 0f);

    long paramCount = model.Parameters().Sum(p => p.ElementCount);
    Console.WriteLine($"Training a tiny LLaMA ({config.LayerCount} layers, dModel {config.EmbeddingDim}, ~{paramCount:N0} params)");
    Console.WriteLine($"on {n} short sentences, byte-level. (CPU reference backend — correctness, not speed.)");
    Console.WriteLine();

    var lossHost = new float[1];
    const int steps = 400;
    for (int step = 1; step <= steps; step++)
    {
        optimizer.ZeroGrad();
        var loss = Loss.CrossEntropy(ctx.Ag, model.Forward(input), target, ignoreIndex: pad);
        loss.Backward();
        optimizer.Step();
        if (step == 1 || step % 40 == 0)
        {
            be.ToHost(loss, lossHost);
            Console.WriteLine($"  step {step,3}   loss {lossHost[0]:F4}");
        }
    }

    // Greedy generation: re-run the model on the growing sequence (no KV cache yet); stop at EOS.
    Console.WriteLine();
    Console.WriteLine($"Prompt:    \"{prompt}\"");
    var generated = new List<int>(tokenizer.Encode(prompt));
    if (generated.Count == 0) generated.Add((int)' ');
    if (generated.Count >= config.MaxSequenceLength)
        generated = generated.Skip(generated.Count - (config.MaxSequenceLength - 1)).ToList();
    using (GradMode.NoGrad())
    {
        for (int t = 0; t < 48 && generated.Count < config.MaxSequenceLength; t++)
        {
            var logits = model.Forward(be.FromHost(ToFloats(generated, 0, generated.Count), new Shape(1, generated.Count), DType.F32));
            var host = new float[logits.ElementCount];
            be.ToHost(logits, host);
            int rowBase = (generated.Count - 1) * vocab, next = 0;
            for (int v = 1; v < vocab; v++) if (host[rowBase + v] > host[rowBase + next]) next = v;
            if (next == eos) break;
            generated.Add(next);
        }
    }
    Console.WriteLine($"Generated: \"{tokenizer.Decode(generated)}\"");
}

static float[] ToFloats(IReadOnlyList<int> ids, int start, int count)
{
    var f = new float[count];
    for (int i = 0; i < count; i++) f[i] = ids[start + i];
    return f;
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
