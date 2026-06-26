using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using ProjectAI.Backends.Cpu;
using ProjectAI.Backends.Torch;
using ProjectAI.Core;
using ProjectAI.Formats.Datasets;
using ProjectAI.Models;
using ProjectAI.Tokenizers;
using ProjectAI.Training;

Console.WriteLine("ProjectAI — local, hand-written AI runtime (.NET 10)");
Console.WriteLine();

// Composition root. The default backend is chosen ONCE here (--backend cpu|torch [--device cpu|cuda|metal]);
// every call site below depends only on IComputeBackend, so the GPU path requires no change above this seam.
// `serve` additionally lets the client switch backend per request (see ComputeRegistry), seeded with this one.
string backendId = ResolveBackendId(args);
using IComputeBackend backend = CreateBackend(backendId);
Console.WriteLine($"Active backend: {backend.Name} on {backend.Device}");
Console.WriteLine();

var command = args.Length > 0 ? args[0] : "help";
switch (command)
{
    case "demo":
        RunStage0Demo(backend);
        break;
    case "train":
    {
        var opts = ParseCliOptions(args.Length > 1 ? args[1..] : []);
        if (opts.Data is not null) RunTrainFile(backend, opts);
        else RunLlmDemo(backend, opts.Prompt, opts.Sampler, opts.Name);
        break;
    }
    case "generate":
    {
        var opts = ParseCliOptions(args.Length > 1 ? args[1..] : []);
        if (opts.Load is null)
        {
            Console.Error.WriteLine("error: generate requires --load <checkpoint> (train first to create one)");
            Environment.Exit(2);
        }
        RunGenerate(backend, opts.Load, opts.Prompt, opts.Sampler);
        break;
    }
    case "tokenize":
    {
        var opts = ParseCliOptions(args.Length > 1 ? args[1..] : []);
        string ckpt = opts.Load ?? Path.Combine("checkpoints", opts.Name + ".ckpt");
        RunTokenize(opts.Prompt, ckpt);
        break;
    }
    case "serve":
    {
        var opts = ParseCliOptions(args.Length > 1 ? args[1..] : []);
        string modelsDir;
        string defaultModel;
        if (opts.Load is not null) // --load <file>: that model is the default, its folder is the picker source
        {
            modelsDir = Path.GetDirectoryName(opts.Load) is { Length: > 0 } dir ? dir : ".";
            defaultModel = Path.GetFileNameWithoutExtension(opts.Load);
        }
        else
        {
            modelsDir = opts.Models ?? "checkpoints";
            defaultModel = "model";
        }
        Server.Run(backend, backendId, modelsDir, defaultModel, opts.Port);
        break;
    }
    case "convert":
    {
        var opts = ParseCliOptions(args.Length > 1 ? args[1..] : []);
        RunConvert(backend, opts.Prompt, opts.Name, opts.Bf16);
        break;
    }
    case "convert-dataset":
    {
        var opts = ParseCliOptions(args.Length > 1 ? args[1..] : []);
        RunConvertDataset(opts);
        break;
    }
    default:
        Console.WriteLine("Usage: projectai <demo|train|generate|tokenize|serve|convert>");
        Console.WriteLine("  demo            Stage 0 milestone: fit y = Wx + b with our own autograd + AdamW.");
        Console.WriteLine("  train [prompt] [--name M] [--data <file>] [--steps N] [--batch B] [--seqlen S] [--lr X] [--temp T] [--topk K] [--topp P] [--seed S]");
        Console.WriteLine("                  No --data: train a tiny LLaMA on a built-in corpus. With --data <file>: train on");
        Console.WriteLine("                  your own text. Saves checkpoints/<name>.ckpt (--name, default 'model') for the picker.");
        Console.WriteLine("  generate --load <checkpoint> [prompt] [--temp T] [--topk K] [--topp P] [--seed S]");
        Console.WriteLine("                  Reload a saved checkpoint (architecture read from the file) and generate.");
        Console.WriteLine("  tokenize <text> [--load <checkpoint>] [--name M]");
        Console.WriteLine("                  Show how <text> splits into tokens (ids + pieces) for a model's tokenizer.");
        Console.WriteLine("  serve [--models <dir>] [--load <checkpoint>] [--port N]");
        Console.WriteLine("                  Serve an HTTP API (POST /generate, GET /health, GET /models) over a directory of");
        Console.WriteLine("                  trained .ckpt models (default ./checkpoints) so the UI client can pick a model.");
        Console.WriteLine("  convert <hf-model-dir> [--name M]");
        Console.WriteLine("                  Convert a HuggingFace Llama checkpoint (config.json + .safetensors) to a .ckpt.");
        Console.WriteLine("  convert-dataset --input <text-file> [--output <dir>] [--seqlen S] [--format text]");
        Console.WriteLine("                  Pre-tokenize a corpus into a packed token-id dataset (mmap'd at train time),");
        Console.WriteLine("                  then train on it with: train --data <dir>.");
        Console.WriteLine();
        Console.WriteLine("  Global:  --backend cpu|torch   Pick the compute backend (default cpu).");
        Console.WriteLine("           --device cpu|cuda|metal Device for the torch backend (default cpu; cuda needs a libtorch bundle).");
        Console.WriteLine("                  Default decoding is greedy; any sampling flag switches to temperature/top-k/top-p.");
        break;
}

// Resolves the default compute backend id from --backend / --device (scanned from the raw args before command
// dispatch), then creates it — probing the native runtime so a missing libtorch bundle or an absent GPU fails fast
// with clear guidance rather than deep inside training. `serve` reuses the id to seed the per-request registry.
static string ResolveBackendId(string[] cliArgs)
{
    try { return Backends.ResolveId(ArgValue(cliArgs, "--backend"), ArgValue(cliArgs, "--device")); }
    catch (ArgumentException ex) { return Die<string>(ex.Message); }
}

static IComputeBackend CreateBackend(string id)
{
    try { return Backends.Create(id); }
    catch (Exception ex)
    {
        return Die<IComputeBackend>(
            $"could not start the '{id}' backend ({ex.Message}). For a GPU/torch backend install a libtorch bundle " +
            "(TorchSharp-cuda-windows for an RTX 4090, or libtorch-cpu) — see ProjectAI.Backends.Torch.csproj.");
    }
}

static string? ArgValue(string[] a, string flag)
{
    for (int i = 0; i < a.Length - 1; i++)
        if (a[i] == flag) return a[i + 1];
    return null;
}

[DoesNotReturn]
static T Die<T>(string message)
{
    Console.Error.WriteLine($"error: {message}");
    Environment.Exit(2);
    throw new InvalidOperationException("unreachable");
}

// Parses the prompt, decoding flags, and (for train) data/hyperparameter flags. No sampling flag → greedy;
// otherwise a seeded temperature/top-k/top-p sampler (S1-9). Non-flag tokens join into the prompt; a bad flag,
// missing value, unknown flag, or out-of-range value fails fast with a usage message rather than being swallowed.
static CliOptions ParseCliOptions(string[] rest)
{
    float temperature = 1.0f;
    int topK = 0;
    float topP = 1.0f;
    ulong seed = 0;
    bool sample = false;
    string? load = null, data = null, models = null;
    string name = "model";
    int steps = 400, batch = 8, seqLen = 64, port = 8080;
    float lr = 3e-3f;
    bool bf16 = false;
    string? input = null, output = null;
    string format = "text";
    var promptParts = new List<string>();

    for (int i = 0; i < rest.Length; i++)
    {
        string tok = rest[i];
        switch (tok)
        {
            case "--temp": temperature = ParseFloatFlag(rest, ref i, tok); sample = true; break;
            case "--topk": topK = ParseIntFlag(rest, ref i, tok); sample = true; break;
            case "--topp": topP = ParseFloatFlag(rest, ref i, tok); sample = true; break;
            case "--seed": seed = (ulong)ParseIntFlag(rest, ref i, tok); break;
            case "--load": load = ParseStringFlag(rest, ref i, tok); break;
            case "--data": data = ParseStringFlag(rest, ref i, tok); break;
            case "--models": models = ParseStringFlag(rest, ref i, tok); break;
            case "--name": name = ParseStringFlag(rest, ref i, tok); break;
            case "--steps": steps = ParseIntFlag(rest, ref i, tok); break;
            case "--batch": batch = ParseIntFlag(rest, ref i, tok); break;
            case "--seqlen": seqLen = ParseIntFlag(rest, ref i, tok); break;
            case "--lr": lr = ParseFloatFlag(rest, ref i, tok); break;
            case "--port": port = ParseIntFlag(rest, ref i, tok); break;
            case "--bf16": bf16 = true; break; // convert: load + run the model in BF16 (half memory, S3-1)
            case "--input": input = ParseStringFlag(rest, ref i, tok); break;   // convert-dataset: source corpus
            case "--output": output = ParseStringFlag(rest, ref i, tok); break; // convert-dataset: dataset dir
            case "--format": format = ParseStringFlag(rest, ref i, tok); break; // convert-dataset: text|parquet|jsonl
            // Consumed at the composition root (CreateBackend); accepted here so they aren't flagged as unknown.
            case "--backend": _ = ParseStringFlag(rest, ref i, tok); break;
            case "--device": _ = ParseStringFlag(rest, ref i, tok); break;
            default:
                if (tok.StartsWith("--", StringComparison.Ordinal)) Fail($"unknown flag '{tok}'");
                promptParts.Add(tok);
                break;
        }
    }

    if (temperature < 0f) Fail("--temp must be >= 0");
    if (topK < 0) Fail("--topk must be >= 0");
    if (topP <= 0f || topP > 1f) Fail("--topp must be in (0, 1]");
    if (steps < 1) Fail("--steps must be >= 1");
    if (batch < 1) Fail("--batch must be >= 1");
    if (seqLen < 1 || seqLen > 8192) Fail("--seqlen must be in [1, 8192]"); // cap RoPE-table / memory blow-up
    if (lr <= 0f) Fail("--lr must be > 0");
    if (port is < 1 or > 65535) Fail("--port must be in [1, 65535]");
    if (name.Length == 0 || name.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 || name != Path.GetFileName(name))
        Fail("--name must be a simple file name (no path separators)");

    string prompt = promptParts.Count > 0 ? string.Join(' ', promptParts) : "roses are ";
    ISampler sampler = sample
        ? new TopKTopPSampler(new PcgRng(seed), temperature, topK, topP)
        : new GreedySampler();
    return new CliOptions(prompt, sampler, load, data, models, name, steps, batch, seqLen, lr, port, bf16, input, output, format);

    static float ParseFloatFlag(string[] a, ref int i, string flag)
    {
        if (i + 1 >= a.Length) Fail($"{flag} expects a value");
        // Reject non-finite ("nan"/"infinity") too: NaN evades every range comparison below and would
        // silently fall back to greedy/unfiltered decoding, the opposite of what the sampling flag asked for.
        if (!float.TryParse(a[++i], NumberStyles.Float, CultureInfo.InvariantCulture, out float v) || !float.IsFinite(v))
            Fail($"{flag} expects a finite number, got '{a[i]}'");
        return v;
    }

    static int ParseIntFlag(string[] a, ref int i, string flag)
    {
        if (i + 1 >= a.Length) Fail($"{flag} expects a value");
        if (!int.TryParse(a[++i], NumberStyles.Integer, CultureInfo.InvariantCulture, out int v))
            Fail($"{flag} expects an integer, got '{a[i]}'");
        return v;
    }

    static string ParseStringFlag(string[] a, ref int i, string flag)
    {
        if (i + 1 >= a.Length) Fail($"{flag} expects a value");
        return a[++i];
    }

    [DoesNotReturn]
    static void Fail(string message)
    {
        Console.Error.WriteLine($"error: {message}");
        Console.Error.WriteLine("usage: projectai train [prompt] [--temp T] [--topk K] [--topp P] [--seed S]");
        Environment.Exit(2);
    }
}

// Stage 1 milestone: train a tiny Llama-style transformer from scratch on a small corpus, then generate.
// Everything below — embedding, RoPE, GQA attention, SwiGLU, RMSNorm, cross-entropy, AdamW — is hand-written.
static void RunLlmDemo(IComputeBackend be, string prompt, ISampler sampler, string name)
{
    // A few short, independent sentences — each learned from position 0, so any of them can be continued
    // from its start. (RoPE encodes ABSOLUTE position, so a tiny over-fit model memorizes position→token;
    // training each sentence from position 0 is what lets an arbitrary in-corpus prompt continue correctly.)
    string[] lines = ["roses are red", "violets are blue", "sugar is sweet", "and so are you"];
    var tokenizer = new BpeTokenizer([]); // byte-level (vocab = 259)
    int pad = tokenizer.PadId, eos = tokenizer.EosId;

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

    var config = DemoConfig();
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

    // Persist the trained model (weights + config + tokenizer) so `generate --load` can reproduce it.
    SaveAndAnnounce(be, model, config, tokenizer, steps, name);
    Generate(be, model, tokenizer, prompt, sampler, config);
}

// Trains on a user-provided text file via the real Trainer (batched, warmup+cosine LR), saves a config-bearing
// checkpoint, then generates from the prompt.
static void RunTrainFile(IComputeBackend be, CliOptions opts)
{
    // A packed dataset (a directory with a manifest, or the manifest itself) trains via the mmap'd PackedBinDataset
    // and reads its sequence length / vocab / tokenizer from the manifest; otherwise --data is a raw text file.
    if (DatasetManifest.TryResolvePath(opts.Data!) is not null)
    {
        RunTrainPacked(be, opts);
        return;
    }
    if (!File.Exists(opts.Data!))
    {
        Console.Error.WriteLine($"error: data file '{opts.Data}' not found");
        Environment.Exit(2);
    }
    string text = File.ReadAllText(opts.Data!);
    var tokenizer = new BpeTokenizer([]); // byte-level
    var config = DemoConfig() with { MaxSequenceLength = Math.Max(DemoConfig().MaxSequenceLength, opts.SeqLen) };

    TextDataset dataset;
    try { dataset = new TextDataset(text, tokenizer, opts.SeqLen); }
    catch (ArgumentException ex) { Console.Error.WriteLine($"error: {ex.Message}"); Environment.Exit(2); return; }

    var model = new LlamaModel(ParameterContext.Create(be, 0), config);
    long paramCount = model.Parameters().Sum(p => p.ElementCount);
    Console.WriteLine($"Training a tiny LLaMA ({config.LayerCount} layers, dModel {config.EmbeddingDim}, ~{paramCount:N0} params)");
    Console.WriteLine($"on '{opts.Data}' — {dataset.Count} blocks of {opts.SeqLen} tokens, batch {opts.Batch}, {opts.Steps} steps.");
    Console.WriteLine();

    var report = new Trainer(be).Train(model, dataset, new TrainingConfig
    {
        BatchSize = opts.Batch, SequenceLength = opts.SeqLen, LearningRate = opts.Lr, MaxSteps = opts.Steps,
        WarmupSteps = Math.Min(100, opts.Steps / 10), Seed = 0, CheckpointDirectory = "",
        OnStep = (step, loss) => { if (step == 1 || step % 20 == 0) Console.WriteLine($"  step {step,4}   loss {loss:F4}"); },
    });
    Console.WriteLine($"  loss {report.FirstLoss:F3} → {report.LastLoss:F3}  ({report.FinalStep} steps)");

    SaveAndAnnounce(be, model, config, tokenizer, report.FinalStep, opts.Name);
    Generate(be, model, tokenizer, opts.Prompt, opts.Sampler, config);
}

// Offline: pre-tokenize a local text corpus into a packed token-id binary + manifest that `train --data <dir>`
// memory-maps. The first stage of the HuggingFace data path (P0): raw text in, packed dataset out, byte-level BPE
// (the same tokenizer `train` uses, so the ids line up). Parquet/JSONL and hub auto-download are later phases.
static void RunConvertDataset(CliOptions opts)
{
    if (string.IsNullOrWhiteSpace(opts.Input) || !File.Exists(opts.Input))
    {
        Console.Error.WriteLine($"error: convert-dataset needs --input <text-file>; got '{opts.Input}'");
        Console.Error.WriteLine("usage: projectai convert-dataset --input <text-file> [--output <dir>] [--seqlen S] [--format text]");
        Environment.Exit(2);
        return;
    }
    if (!string.Equals(opts.Format, "text", StringComparison.OrdinalIgnoreCase))
    {
        Console.Error.WriteLine($"error: --format '{opts.Format}' is not supported yet (P0 reads raw text; Parquet/JSONL land in P1).");
        Environment.Exit(2);
        return;
    }

    string outputDir = opts.Output ?? Path.Combine("datasets", Path.GetFileNameWithoutExtension(opts.Input));
    var reader = new RawTextDatasetReader();
    var tokenizer = new BpeTokenizer([]); // byte-level — matches `train`/RunTrainFile so the ids are interchangeable

    Console.WriteLine($"Packing '{opts.Input}' → '{outputDir}'  (byte-level BPE, sequence length {opts.SeqLen}) …");
    DatasetManifest manifest;
    try
    {
        manifest = DatasetPacker.Pack(
            reader.ReadDocuments(opts.Input), tokenizer, opts.SeqLen, outputDir,
            appendEosBetweenDocuments: true, sourceId: opts.Input);
    }
    catch (Exception ex) when (ex is InvalidOperationException or ArgumentException)
    {
        Console.Error.WriteLine($"error: {ex.Message}");
        Environment.Exit(2);
        return;
    }

    Console.WriteLine($"  {manifest.TotalTokens:N0} tokens → {manifest.BlockCount:N0} blocks of {manifest.SequenceLength + 1} ids ({manifest.Dtype}); {manifest.DroppedTokens} dropped.");
    Console.WriteLine($"Wrote {Path.Combine(outputDir, DatasetManifest.FileName)}");
    Console.WriteLine($"Train on it:  train --data {outputDir} --steps {opts.Steps} --batch {opts.Batch}");
}

// Trains on a packed dataset directory (manifest + .bin). Sequence length, vocab size, and the tokenizer are read
// FROM the manifest — never CLI defaults — so a dataset packed at one sequence length can't be silently trained at
// another, and the checkpoint embeds the exact tokenizer the ids were produced with (so `generate` decodes right).
static void RunTrainPacked(IComputeBackend be, CliOptions opts)
{
    using var dataset = PackedBinDataset.Open(opts.Data!);
    var m = dataset.Manifest;
    int seqLen = m.SequenceLength;
    if (opts.SeqLen != seqLen)
        Console.WriteLine($"note: using sequence length {seqLen} from the dataset manifest (ignoring --seqlen {opts.SeqLen}).");

    ITokenizer tokenizer = TokenizerCodec.Deserialize(m.TokenizerKind, m.TokenizerState);
    var config = DemoConfig() with
    {
        VocabSize = m.VocabSize,
        MaxSequenceLength = Math.Max(DemoConfig().MaxSequenceLength, seqLen),
    };

    var model = new LlamaModel(ParameterContext.Create(be, 0), config);
    long paramCount = model.Parameters().Sum(p => p.ElementCount);
    Console.WriteLine($"Training a tiny LLaMA ({config.LayerCount} layers, dModel {config.EmbeddingDim}, ~{paramCount:N0} params)");
    Console.WriteLine($"on packed dataset '{opts.Data}' — {dataset.Count} blocks of {seqLen} tokens (vocab {m.VocabSize}, {m.TokenizerKind}), batch {opts.Batch}, {opts.Steps} steps.");
    Console.WriteLine();

    var report = new Trainer(be).Train(model, dataset, new TrainingConfig
    {
        BatchSize = opts.Batch, SequenceLength = seqLen, LearningRate = opts.Lr, MaxSteps = opts.Steps,
        WarmupSteps = Math.Min(100, opts.Steps / 10), Seed = 0, CheckpointDirectory = "",
        OnStep = (step, loss) => { if (step == 1 || step % 20 == 0) Console.WriteLine($"  step {step,4}   loss {loss:F4}"); },
    });
    Console.WriteLine($"  loss {report.FirstLoss:F3} → {report.LastLoss:F3}  ({report.FinalStep} steps)");

    SaveAndAnnounce(be, model, config, tokenizer, report.FinalStep, opts.Name);
    Generate(be, model, tokenizer, opts.Prompt, opts.Sampler, config);
}

static void SaveAndAnnounce(IComputeBackend be, LlamaModel model, ModelConfig config, ITokenizer tokenizer, int step, string name)
{
    string checkpoint = Path.Combine("checkpoints", name + ".ckpt");
    Directory.CreateDirectory("checkpoints");
    Checkpointing.SaveModel(checkpoint, model, config, tokenizer, step, optimizer: null, be);
    Console.WriteLine();
    Console.WriteLine($"Saved checkpoint → {checkpoint}   (serve it with: serve   →   pick '{name}' in the client)");
}

// Converts a HuggingFace Llama checkpoint (config.json + .safetensors) into our model and saves it. Weights
// only — the model's own tokenizer is a separate step, so text isn't meaningful until that lands.
static void RunConvert(IComputeBackend be, string source, string name, bool bf16)
{
    if (string.IsNullOrWhiteSpace(source) || (!File.Exists(source) && !Directory.Exists(source)))
    {
        Console.Error.WriteLine($"error: convert needs a HuggingFace model directory or .safetensors file; got '{source}'");
        Console.Error.WriteLine("usage: projectai convert <model-dir> [--name <out>] [--bf16]");
        Environment.Exit(2);
    }
    if (bf16 && be.Name != "torchsharp")
    {
        Console.Error.WriteLine("error: --bf16 needs the Torch backend (the CPU reference is F32-only); add --backend torch --device cuda.");
        Environment.Exit(2);
    }

    var computeDType = bf16 ? DType.BF16 : DType.F32;
    Console.WriteLine($"Converting '{source}' … ({(bf16 ? "BF16 — half memory" : "F32")})");
    LlamaModel model;
    ModelConfig config;
    try
    {
        (model, config) = Converter.Load(source, be, computeDType);
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"error: {ex.Message}");
        Environment.Exit(2);
        return;
    }

    long paramCount = model.Parameters().Sum(p => p.ElementCount);
    Console.WriteLine($"Loaded {config.LayerCount} layers, dModel {config.EmbeddingDim}, {config.HeadCount} heads (KV {config.KvHeadCount}), vocab {config.VocabSize} — ~{paramCount:N0} params.");

    // Use the model's own tokenizer (tokenizer.json) so generation is meaningful; fall back to a placeholder.
    ITokenizer tokenizer = Converter.TryLoadTokenizer(source) ?? (ITokenizer)new BpeTokenizer([]);
    bool realTokenizer = tokenizer is HfTokenizer;

    Directory.CreateDirectory("checkpoints");
    string checkpoint = Path.Combine("checkpoints", name + ".ckpt");
    Checkpointing.SaveModel(checkpoint, model, config, tokenizer, step: 0, optimizer: null, be);
    Console.WriteLine($"Saved → {checkpoint}");
    if (realTokenizer)
        Console.WriteLine($"Loaded the model's tokenizer (vocab {tokenizer.VocabSize}). Serve it, then pick '{name}' in the client.");
    else
    {
        Console.WriteLine();
        Console.WriteLine("NOTE: no tokenizer.json found — saved with a placeholder byte-level tokenizer, so generated");
        Console.WriteLine("text won't be meaningful. Put the model's tokenizer.json next to its weights and re-run convert.");
    }
}

// Loads a saved checkpoint — rebuilding the model from the config stored in the file — and generates.
static void RunGenerate(IComputeBackend be, string checkpoint, string prompt, ISampler sampler)
{
    if (!File.Exists(checkpoint))
    {
        Console.Error.WriteLine($"error: checkpoint '{checkpoint}' not found (run `train` first)");
        Environment.Exit(2);
    }
    var (model, config, tokenizer, step) = Checkpointing.LoadModel(checkpoint, be);
    Console.WriteLine($"Loaded checkpoint '{checkpoint}' (step {step}, {config.LayerCount} layers, dModel {config.EmbeddingDim}).");
    Generate(be, model, tokenizer, prompt, sampler, config);
}

// Prints a prompt and its continuation, decoding via the shared inference core (used by the CLI demos).
static void Generate(IComputeBackend be, LlamaModel model, ITokenizer tokenizer, string prompt, ISampler sampler, ModelConfig config)
{
    Console.WriteLine();
    Console.WriteLine($"Prompt:    \"{prompt}\"");
    var gen = Inference.GenerateText(be, model, tokenizer, config, prompt, sampler, maxTokens: 48);
    Console.WriteLine($"Generated: \"{gen.FullText}\"");
}

// Shows how a string splits into tokens for a model's tokenizer: each id + its decoded piece, plus the round-trip.
// Loads ONLY the tokenizer from the checkpoint (no weights), so it's instant even for multi-GB models.
static void RunTokenize(string text, string checkpoint)
{
    if (!File.Exists(checkpoint))
    {
        Console.Error.WriteLine($"error: checkpoint '{checkpoint}' not found (use --load <ckpt>, or --name <model> for checkpoints/<model>.ckpt)");
        Environment.Exit(2);
    }
    ITokenizer tokenizer;
    try { tokenizer = Checkpointing.LoadTokenizer(checkpoint); }
    catch (Exception ex) { Console.Error.WriteLine($"error: {ex.Message}"); Environment.Exit(2); return; }

    var ids = tokenizer.Encode(text);
    Console.WriteLine($"Tokenizer: {checkpoint}  (vocab {tokenizer.VocabSize})");
    Console.WriteLine($"Text:      \"{text}\"");
    Console.WriteLine($"Tokens:    {ids.Count}");
    foreach (int id in ids)
    {
        string piece = tokenizer.Decode([id]);
        Console.WriteLine($"  [{id,6}]  {(piece.Length == 0 ? "·(special/control)" : $"\"{EscapeWs(piece)}\"")}");
    }
    string round = tokenizer.Decode(ids);
    Console.WriteLine($"Decoded:   \"{round}\"  ({(round == text ? "round-trips ✓" : "differs ✗")})");
}

static string EscapeWs(string s) =>
    s.Replace("\\", "\\\\").Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");

// The fixed architecture shared by `train` (which saves it) and `generate` (which restores it).
static ModelConfig DemoConfig() => new()
{
    VocabSize = 259, EmbeddingDim = 64, LayerCount = 2, HeadCount = 4, KvHeadCount = 2,
    FeedForwardHiddenDim = 256, MaxSequenceLength = 128,
};

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

// Parsed CLI options shared across commands: the prompt, decoder, optional checkpoint to load, the training
// data file + hyperparameters (train --data), and the serve port.
internal sealed record CliOptions(
    string Prompt, ISampler Sampler, string? Load, string? Data, string? Models, string Name, int Steps, int Batch, int SeqLen, float Lr, int Port, bool Bf16,
    string? Input, string? Output, string Format);
