using ProjectAI.Backends.Cpu;
using ProjectAI.Backends.Torch;
using ProjectAI.Core;
using ProjectAI.Models;
using ProjectAI.Tokenizers;
using ProjectAI.Training;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>
/// Runs the S2-1 conformance op coverage against the TorchSharp backend (ticket S2-2), asserting it matches the
/// CPU oracle within tolerance. The native libtorch runtime is an opt-in per-machine package, so when it isn't
/// installed this test SKIPS rather than fails — CPU-only checkouts stay green, and the moment libtorch (CPU or
/// CUDA) is present it validates the GPU backend with no extra code.
/// </summary>
public class TorchConformanceTests
{
    [Fact]
    public void TorchSharpBackendMatchesTheCpuOracle()
    {
        if (!TorchRuntimeAvailable(out string reason))
        {
            Assert.Skip($"libtorch runtime not installed — install a libtorch bundle to validate the Torch backend ({reason}).");
            return;
        }

        using var oracle = new CpuComputeBackend();
        using var torch = new TorchComputeBackend(Device.Cpu);
        foreach (var (name, run) in BackendConformanceTests.Cases())
            BackendConformanceTests.AssertMatchesOracle(torch, oracle, name, run);
    }

    // Exercises the S2-3 per-step memory scoping on the REAL Torch backend: the trainer scopes each micro-batch's
    // forward+backward (keeping only the accumulated grads) and AdamW scopes its update (keeping only the moments).
    // If any still-needed tensor were freed early, training would diverge or NaN — so a clean loss drop with grad
    // accumulation (cross-scope grad accumulation) is the regression guard the CPU tests can't provide (CPU scopes
    // are no-ops). Skips when libtorch isn't installed.
    [Fact]
    public void TorchBackendTrainsUnderPerStepScoping()
    {
        if (!TorchRuntimeAvailable(out string reason))
        {
            Assert.Skip($"libtorch runtime not installed — install a libtorch bundle to validate Torch training ({reason}).");
            return;
        }

        using var backend = new TorchComputeBackend(Device.Cpu);
        var tokenizer = new BpeTokenizer([]);
        string text = string.Concat(Enumerable.Repeat("the quick brown fox jumps over the lazy dog. ", 30));
        const int seqLen = 24;
        var config = ModelPresets.Get("tiny") with { MaxSequenceLength = Math.Max(ModelPresets.Get("tiny").MaxSequenceLength, seqLen) };
        var dataset = new TextDataset(text, tokenizer, seqLen);
        var model = new LlamaModel(ParameterContext.Create(backend, 0), config);

        var report = new Trainer(backend).Train(model, dataset, new TrainingConfig
        {
            BatchSize = 4,
            SequenceLength = seqLen,
            LearningRate = 3e-3f,
            MaxSteps = 40,
            GradientAccumulationSteps = 2, // accumulate grads across two scoped micro-batches per step
            WarmupSteps = 4,
            CheckpointDirectory = "",
        });

        Assert.True(float.IsFinite(report.LastLoss), "loss became non-finite — scoping likely freed a needed tensor");
        Assert.True(report.LastLoss < report.FirstLoss * 0.9f,
            $"Torch training under per-step scopes did not converge: {report.FirstLoss:F3} → {report.LastLoss:F3}");
    }

    // Validates gradient checkpointing (S3-2) on the REAL Torch backend, where the recompute's BeginScope /
    // KeepAlive / Release actually free native tensors — the CPU gradient-check can't (CPU scope is a no-op). If
    // a block parameter's grad were freed too early (e.g. block.Parameters() missed it), it would diverge here.
    [Fact]
    public void GradientCheckpointingMatchesTheStandardPathOnTorch()
    {
        if (!TorchRuntimeAvailable(out string reason))
        {
            Assert.Skip($"libtorch runtime not installed — install a libtorch bundle to validate Torch checkpointing ({reason}).");
            return;
        }

        using var backend = new TorchComputeBackend(Device.Cpu);
        var config = new ModelConfig
        {
            VocabSize = 16, EmbeddingDim = 32, LayerCount = 3,
            HeadCount = 4, KvHeadCount = 2, FeedForwardHiddenDim = 64, MaxSequenceLength = 16,
        };
        var model = new LlamaModel(ParameterContext.Create(backend, seed: 0), config);

        const int batch = 2, seq = 6;
        var inputData = new float[batch * seq];
        var targets = new int[batch * seq];
        for (int i = 0; i < inputData.Length; i++) { inputData[i] = i % config.VocabSize; targets[i] = (i * 7 + 3) % config.VocabSize; }

        Dictionary<string, float[]> Grads(bool checkpoint)
        {
            foreach (var (_, p) in model.NamedParameters()) p.Grad = null;
            model.GradientCheckpointing = checkpoint;
            var ag = new Autograd(backend);
            var logits = model.Forward(backend.FromHost(inputData, new Shape(batch, seq), DType.F32));
            ag.CrossEntropy(ag.Reshape(logits, targets.Length, config.VocabSize), targets, ignoreIndex: -100).Backward();
            var d = new Dictionary<string, float[]>();
            foreach (var (name, p) in model.NamedParameters())
            {
                var g = new float[p.ElementCount];
                if (p.Grad is not null) backend.ToHost(p.Grad, g);
                d[name] = g;
            }
            return d;
        }

        var baseline = Grads(checkpoint: false);
        var checkpointed = Grads(checkpoint: true);
        foreach (var (name, expected) in baseline)
            for (int i = 0; i < expected.Length; i++)
                Assert.True(MathF.Abs(expected[i] - checkpointed[name][i]) <= 1e-3f + 1e-3f * MathF.Abs(expected[i]),
                    $"Torch checkpointed gradient mismatch for '{name}'[{i}]: {expected[i]} vs {checkpointed[name][i]}");
    }

    // Exercises the FULL production path on libtorch: the trainer's per-micro-batch BeginScope around a
    // CHECKPOINTED forward+backward (a recompute scope nested inside it), the two-level grad hand-off
    // (recompute → step → detached over the full param set), and AdamW's deterministic Release of superseded
    // moments/grads — with grad accumulation. If any of those frees a still-needed tensor, training diverges or
    // NaNs here (the bit-exact grad test runs Backward with no enclosing scope, so it can't reach this path).
    [Fact]
    public void TorchTrainerConvergesWithGradientCheckpointing()
    {
        if (!TorchRuntimeAvailable(out string reason))
        {
            Assert.Skip($"libtorch runtime not installed — install a libtorch bundle to validate Torch checkpointed training ({reason}).");
            return;
        }

        using var backend = new TorchComputeBackend(Device.Cpu);
        var tokenizer = new BpeTokenizer([]);
        string text = string.Concat(Enumerable.Repeat("the quick brown fox jumps over the lazy dog. ", 30));
        const int seqLen = 24;
        var config = ModelPresets.Get("tiny") with { MaxSequenceLength = Math.Max(ModelPresets.Get("tiny").MaxSequenceLength, seqLen) };
        var dataset = new TextDataset(text, tokenizer, seqLen);
        var model = new LlamaModel(ParameterContext.Create(backend, seed: 0), config);

        var report = new Trainer(backend).Train(model, dataset, new TrainingConfig
        {
            BatchSize = 4,
            SequenceLength = seqLen,
            LearningRate = 3e-3f,
            MaxSteps = 40,
            GradientAccumulationSteps = 2, // + checkpointing stresses the nested-scope grad hand-off + Release
            GradientCheckpointing = true,
            WarmupSteps = 4,
            CheckpointDirectory = "",
        });

        Assert.True(float.IsFinite(report.LastLoss), "loss became non-finite — a checkpointed grad was likely freed too early");
        Assert.True(report.LastLoss < report.FirstLoss * 0.9f,
            $"Torch training under gradient checkpointing did not converge: {report.FirstLoss:F3} → {report.LastLoss:F3}");
    }

    // Half-precision inference (S3-1): a model built with ComputeDType=BF16 runs its whole forward in bf16 (params,
    // RoPE tables, mask, activations) and produces logits close to the F32 model — proving the dtype threads end to
    // end. bf16 is Torch-only (the CPU oracle is F32), so this is a Torch test.
    [Fact]
    public void BF16InferenceTracksF32OnTorch()
    {
        if (!TorchRuntimeAvailable(out string reason))
        {
            Assert.Skip($"libtorch runtime not installed — install a libtorch bundle to validate BF16 inference ({reason}).");
            return;
        }

        using var backend = new TorchComputeBackend(Device.Cpu);
        var config = new ModelConfig
        {
            VocabSize = 32, EmbeddingDim = 64, LayerCount = 2,
            HeadCount = 4, KvHeadCount = 2, FeedForwardHiddenDim = 128, MaxSequenceLength = 16,
        };
        const int batch = 1, seq = 8;
        var inputData = new float[batch * seq];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = i % config.VocabSize;

        float[] Logits(DType dtype)
        {
            var model = new LlamaModel(ParameterContext.Create(backend, seed: 0, computeDType: dtype), config);
            using (GradMode.NoGrad())
            {
                var logits = model.Forward(backend.FromHost(inputData, new Shape(batch, seq), DType.F32));
                var host = new float[logits.ElementCount];
                backend.ToHost(logits, host);
                return host;
            }
        }

        var f32 = Logits(DType.F32);
        var bf16 = Logits(DType.BF16);

        Assert.Equal(f32.Length, bf16.Length);
        double maxAbs = 1e-6;
        foreach (var v in f32) maxAbs = Math.Max(maxAbs, Math.Abs(v));
        for (int i = 0; i < f32.Length; i++)
        {
            Assert.True(float.IsFinite(bf16[i]), $"bf16 logit[{i}] is not finite");
            // bf16 has ~3 significant digits; same seed → bf16 weights are the f32 weights rounded, so logits track within a few %.
            Assert.True(Math.Abs(f32[i] - bf16[i]) <= 0.06 * maxAbs + 0.05,
                $"bf16 logit[{i}] diverged from f32: {f32[i]} vs {bf16[i]} (scale {maxAbs:F3})");
        }
    }

    // A BF16 model must survive a checkpoint save→load: the precision is recorded in metadata and the weights
    // reload at BF16 (the on-disk payload is widened to F32, then narrowed back — exact for bf16). Proves
    // `convert --bf16` → serve keeps half-precision instead of silently expanding to F32 (S3-1).
    [Fact]
    public void BF16ModelSurvivesACheckpointRoundTripOnTorch()
    {
        if (!TorchRuntimeAvailable(out string reason))
        {
            Assert.Skip($"libtorch runtime not installed — install a libtorch bundle to validate BF16 checkpoints ({reason}).");
            return;
        }

        using var backend = new TorchComputeBackend(Device.Cpu);
        var config = new ModelConfig
        {
            VocabSize = 32, EmbeddingDim = 64, LayerCount = 2,
            HeadCount = 4, KvHeadCount = 2, FeedForwardHiddenDim = 128, MaxSequenceLength = 16,
        };
        var tokenizer = new BpeTokenizer([]);
        const int batch = 1, seq = 8;
        var inputData = new float[batch * seq];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = i % config.VocabSize;

        float[] Logits(LlamaModel m)
        {
            using (GradMode.NoGrad())
            {
                var l = m.Forward(backend.FromHost(inputData, new Shape(batch, seq), DType.F32));
                var h = new float[l.ElementCount];
                backend.ToHost(l, h);
                return h;
            }
        }

        string path = Path.Combine(Path.GetTempPath(), $"pai-bf16-{Guid.NewGuid():N}.ckpt");
        try
        {
            var model = new LlamaModel(ParameterContext.Create(backend, seed: 0, computeDType: DType.BF16), config);
            var before = Logits(model);
            Checkpointing.SaveModel(path, model, config, tokenizer, step: 1, optimizer: null, backend);

            var (reloaded, _, _, _) = Checkpointing.LoadModel(path, backend);
            Assert.Equal(DType.BF16, reloaded.Parameters().First().DType); // reloaded model is still BF16
            var after = Logits(reloaded);
            for (int i = 0; i < before.Length; i++)
                Assert.True(MathF.Abs(before[i] - after[i]) <= 1e-3f, $"bf16 checkpoint logit[{i}] changed: {before[i]} vs {after[i]}");
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    // Probe by forcing a real native call (Allocate + ToHost). A missing libtorch bundle throws a load error here.
    private static bool TorchRuntimeAvailable(out string reason)
    {
        try
        {
            using var be = new TorchComputeBackend(Device.Cpu);
            var t = be.Allocate(new Shape(1), DType.F32);
            be.ToHost(t, new float[1]);
            reason = "";
            return true;
        }
        catch (Exception ex)
        {
            reason = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }
}
