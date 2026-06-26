using System.Diagnostics;
using ProjectAI.Backends.Cpu;
using ProjectAI.Backends.Torch;
using ProjectAI.Core;
using ProjectAI.Models;
using ProjectAI.Tokenizers;
using ProjectAI.Training;

// ProjectAI.Trainer — a focused, launchable trainer for your own models. Point it at a text file, pick a size,
// and it trains on the GPU (CUDA) if one is available (else CPU), saves a checkpoint, shows a sample of what it
// learned, and prints how to chat with it. Drag a .txt onto train.cmd, or:
//   dotnet run --project ProjectAI.Trainer -- mytext.txt --size small

Console.WriteLine("ProjectAI Trainer — train your own model");
Console.WriteLine();

var opts = TrainerOptions.Parse(args);

// Resolve the input text file — prompt for one if none was given (e.g. a bare double-click).
string? dataFile = opts.DataFile;
if (string.IsNullOrWhiteSpace(dataFile))
{
    Console.Write("Path to a text file to train on: ");
    dataFile = Console.ReadLine()?.Trim().Trim('"');
}
if (string.IsNullOrWhiteSpace(dataFile) || !File.Exists(dataFile))
{
    Console.Error.WriteLine($"error: text file not found: '{dataFile}'");
    Console.Error.WriteLine("usage: projectai-train <textfile> [--name N] [--size tiny|small|medium|large]");
    Console.Error.WriteLine("       [--steps N] [--batch N] [--seqlen N] [--lr X] [--device auto|cpu|cuda] [--prompt \"...\"]");
    return 2;
}

string size = opts.Size.Trim().ToLowerInvariant();
ModelConfig preset;
try { preset = ModelPresets.Get(size); }
catch (ArgumentException ex) { Console.Error.WriteLine($"error: {ex.Message}"); return 2; }

string name = opts.Name is { Length: > 0 } n ? n : Sanitize(Path.GetFileNameWithoutExtension(dataFile));
if (string.IsNullOrWhiteSpace(name)) name = "mymodel";

// Memory-aware defaults (shared with the server): bigger models use a smaller batch so a single step's activation
// graph fits typical VRAM (deterministic per-step memory is ticket S2-3). Override with --batch / --seqlen.
var (autoBatch, autoSeqLen) = ModelPresets.DefaultTraining(size);
int batch = opts.Batch > 0 ? opts.Batch : autoBatch;
int seqLen = opts.SeqLen > 0 ? opts.SeqLen : autoSeqLen;
// Gradient checkpointing (S3-2) trades a recompute for activation memory; on by default for 'large' (which won't
// otherwise fit a modest GPU) and opt-in elsewhere via --checkpoint.
bool useCheckpointing = opts.Checkpoint || size == "large";

string text = File.ReadAllText(dataFile);
var tokenizer = new BpeTokenizer([]); // byte-level: trains on any text with zero setup
var config = preset with { MaxSequenceLength = Math.Max(preset.MaxSequenceLength, seqLen) };

TextDataset dataset;
try { dataset = new TextDataset(text, tokenizer, seqLen); }
catch (ArgumentException ex) { Console.Error.WriteLine($"error: {ex.Message} (use a longer text file or a smaller --seqlen)"); return 2; }
if (dataset.Count == 0)
{
    Console.Error.WriteLine("error: the text is too short for even one training block — use a longer file or a smaller --seqlen.");
    return 2;
}

using IComputeBackend backend = CreateBackend(opts.Device);
Console.WriteLine($"Backend:  {backend.Name} on {backend.Device}");

var model = new LlamaModel(ParameterContext.Create(backend, 0), config);
long paramCount = model.Parameters().Sum(p => p.ElementCount);
Console.WriteLine($"Model:    '{size}' — {ModelPresets.Describe(size)} — ~{paramCount:N0} params{(useCheckpointing ? "  (gradient checkpointing on)" : "")}");
Console.WriteLine($"Data:     {text.Length:N0} chars → {dataset.Count} blocks of {seqLen} tokens   (batch {batch}, {opts.Steps} steps)");
Console.WriteLine();

var sw = Stopwatch.StartNew();
TrainingReport report;
try
{
    report = new Trainer(backend).Train(model, dataset, new TrainingConfig
    {
        BatchSize = batch,
        SequenceLength = seqLen,
        LearningRate = opts.Lr,
        MaxSteps = opts.Steps,
        WarmupSteps = Math.Clamp(opts.Steps / 10, 1, Math.Max(1, opts.Steps - 1)),
        GradientCheckpointing = useCheckpointing,
        CheckpointDirectory = "", // the trainer writes the final, self-describing checkpoint itself (below)
        OnStep = (step, loss) =>
        {
            if (step == 1 || step % 20 == 0 || step == opts.Steps)
                Console.WriteLine($"  step {step,5}/{opts.Steps}   loss {loss,8:F4}");
            // Free detached native tensors between steps. CUDA pressure doesn't trigger the .NET GC, so without
            // this the per-step graph (S2-3) can accumulate across steps on the GPU until it OOMs.
            if (step % 10 == 0) { GC.Collect(); GC.WaitForPendingFinalizers(); }
        },
    });
}
catch (Exception ex) when (IsOutOfMemory(ex))
{
    Console.Error.WriteLine();
    Console.Error.WriteLine($"error: ran out of GPU memory training the '{size}' model at batch {batch}, seqlen {seqLen}.");
    Console.Error.WriteLine("  Try one of: a smaller --batch (e.g. --batch 1), a smaller --seqlen, --checkpoint (gradient checkpointing), a smaller --size, or --device cpu.");
    return 3;
}
sw.Stop();
double secs = Math.Max(0.001, sw.Elapsed.TotalSeconds);
Console.WriteLine();
Console.WriteLine($"Done:     loss {report.FirstLoss:F3} → {report.LastLoss:F3}  in {secs:F1}s  ({report.FinalStep / secs:F1} steps/s)");

Directory.CreateDirectory("checkpoints");
string checkpoint = Path.Combine("checkpoints", name + ".ckpt");
Checkpointing.SaveModel(checkpoint, model, config, tokenizer, report.FinalStep, optimizer: null, backend);
Console.WriteLine($"Saved →   {checkpoint}");

// A quick greedy sample so you can see what it learned right away.
string promptText = opts.Prompt is { Length: > 0 } p ? p : new string(text.Replace('\r', ' ').Replace('\n', ' ').Take(24).ToArray());
string sample = Inference.GenerateText(backend, model, tokenizer, config, promptText, new GreedySampler(), maxTokens: 80).FullText;
Console.WriteLine();
Console.WriteLine($"Sample (prompt \"{promptText.Trim()}\"):");
Console.WriteLine($"  {sample.Replace("\n", "\n  ")}");
Console.WriteLine();
Console.WriteLine($"Chat with it:  dotnet run --project ProjectAI -- serve --load {checkpoint}");
Console.WriteLine("               …then open the Godot client and pick this model.");
return 0;

// Auto-selects the GPU when asked (or available), else CPU. Forces native init so a bad device fails fast.
static IComputeBackend CreateBackend(string device)
{
    switch (device.Trim().ToLowerInvariant())
    {
        case "cpu":
            return new CpuComputeBackend();
        case "cuda":
        case "gpu":
            return ForceInit(new TorchComputeBackend(new Device(DeviceKind.Cuda)));
        default: // auto
            if (TorchComputeBackend.IsAvailable(DeviceKind.Cuda, out _))
            {
                Console.WriteLine("GPU (CUDA) detected — training on the GPU.");
                return ForceInit(new TorchComputeBackend(new Device(DeviceKind.Cuda)));
            }
            Console.WriteLine("No CUDA GPU detected — training on CPU (slower; add the CUDA bundle for GPU speed).");
            return new CpuComputeBackend();
    }
}

static IComputeBackend ForceInit(IComputeBackend backend)
{
    backend.ToHost(backend.Allocate(new Shape(1), DType.F32), new float[1]); // probe the device now → clear error if it can't start
    return backend;
}

// True if this exception (or an inner one) is a CUDA/host out-of-memory, so we can fail with guidance not a stack trace.
static bool IsOutOfMemory(Exception ex)
{
    for (Exception? e = ex; e is not null; e = e.InnerException)
        if (e.Message.Contains("out of memory", StringComparison.OrdinalIgnoreCase)) return true;
    return false;
}

static string Sanitize(string s) => new([.. s.Where(c => char.IsLetterOrDigit(c) || c is '-' or '_')]);

sealed record TrainerOptions(string? DataFile, string? Name, string Size, int Steps, int Batch, int SeqLen, float Lr, string Device, string? Prompt, bool Checkpoint)
{
    public static TrainerOptions Parse(string[] args)
    {
        string? file = null, name = null, prompt = null;
        string size = "small", device = "auto";
        int steps = 300, batch = 0, seqLen = 0; // 0 = use the memory-aware per-size default
        float lr = 3e-4f;
        bool checkpoint = false;
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            string? next = i + 1 < args.Length ? args[i + 1] : null;
            switch (a)
            {
                case "--name": name = next; i++; break;
                case "--size": if (next is not null) { size = next; i++; } break;
                case "--steps": if (next is not null) { steps = ParseInt(next, steps); i++; } break;
                case "--batch": if (next is not null) { batch = ParseInt(next, batch); i++; } break;
                case "--seqlen": if (next is not null) { seqLen = ParseInt(next, seqLen); i++; } break;
                case "--lr": if (next is not null) { lr = ParseFloat(next, lr); i++; } break;
                case "--device": if (next is not null) { device = next; i++; } break;
                case "--prompt": prompt = next; i++; break;
                case "--checkpoint": checkpoint = true; break; // flag, no value
                default:
                    if (!a.StartsWith("--", StringComparison.Ordinal) && file is null) file = a; // first positional = the text file
                    break;
            }
        }
        return new TrainerOptions(file, name, size, Math.Max(1, steps), batch, seqLen, lr, device, prompt, checkpoint);
    }

    private static int ParseInt(string? s, int fallback) => int.TryParse(s, out int v) ? v : fallback;
    private static float ParseFloat(string? s, float fallback) => float.TryParse(s, out float v) ? v : fallback;
}
