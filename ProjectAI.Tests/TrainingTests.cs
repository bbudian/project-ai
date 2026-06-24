using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using ProjectAI.Formats;
using ProjectAI.Models;
using ProjectAI.Tokenizers;
using ProjectAI.Training;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>Tests for the S1-10 training loop, dataset packing, gradient accumulation, and checkpointing.</summary>
public class TrainingTests
{
    private const string Corpus =
        "the quick brown fox jumps over the lazy dog. " +
        "a stitch in time saves nine. the early bird catches the worm. " +
        "all that glitters is not gold. practice makes perfect every single day.";

    private static ModelConfig SmallConfig() => new()
    {
        VocabSize = 259, EmbeddingDim = 32, LayerCount = 2, HeadCount = 4, KvHeadCount = 2,
        FeedForwardHiddenDim = 64, MaxSequenceLength = 64,
    };

    private static float[] Host(CpuComputeBackend be, Tensor t)
    {
        var buf = new float[t.ElementCount];
        be.ToHost(t, buf);
        return buf;
    }

    [Fact]
    public void TextDataset_PacksBlocksOfSequenceLengthPlusOne()
    {
        var tok = new BpeTokenizer([]);
        int ids = tok.Encode(Corpus).Count;
        var ds = new TextDataset(Corpus, tok, sequenceLength: 16);

        Assert.Equal(ids / 17, ds.Count);
        Assert.All(Enumerable.Range(0, ds.Count), i => Assert.Equal(17, ds.GetSequence(i).Length));
        // Block b must be the contiguous slice starting at b*17 of the tokenized corpus.
        var all = tok.Encode(Corpus);
        Assert.Equal(all[0], ds.GetSequence(0).Span[0]);
        Assert.Equal(all[17], ds.GetSequence(1).Span[0]);
    }

    [Fact]
    public void TextDataset_ThrowsWhenCorpusTooShort()
    {
        var tok = new BpeTokenizer([]);
        Assert.Throws<ArgumentException>(() => new TextDataset("hi", tok, sequenceLength: 64));
    }

    [Fact]
    public void Trainer_LossDecreasesOverTraining()
    {
        using var be = new CpuComputeBackend();
        var model = new LlamaModel(ParameterContext.Create(be, 7), SmallConfig());
        var dataset = new TextDataset(Corpus, new BpeTokenizer([]), sequenceLength: 16);
        var dir = NewTempDir();
        try
        {
            var report = new Trainer(be).Train(model, dataset, new TrainingConfig
            {
                BatchSize = 4, SequenceLength = 16, LearningRate = 3e-3f, MaxSteps = 120,
                WarmupSteps = 10, Seed = 1, CheckpointDirectory = dir,
            });

            float firstAvg = report.StepLosses.Take(10).Average();
            float lastAvg = report.StepLosses.TakeLast(10).Average();
            Assert.True(lastAvg < firstAvg * 0.7f, $"loss should fall clearly: first10 {firstAvg:F3}, last10 {lastAvg:F3}");
            Assert.Equal(120, report.FinalStep);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void Checkpoint_Reload_ReproducesIdenticalLogits()
    {
        using var be = new CpuComputeBackend();
        var dir = NewTempDir();
        try
        {
            // Train model A and let the trainer write the final checkpoint.
            var config = new TrainingConfig
            {
                BatchSize = 4, SequenceLength = 16, LearningRate = 3e-3f, MaxSteps = 25,
                WarmupSteps = 5, Seed = 3, CheckpointDirectory = dir,
            };
            var modelA = new LlamaModel(ParameterContext.Create(be, 7), SmallConfig());
            new Trainer(be).Train(modelA, new TextDataset(Corpus, new BpeTokenizer([]), 16), config);

            var probe = be.FromHost([1, 2, 3, 4, 5, 6, 7, 8], new Shape(1, 8), DType.F32);
            float[] logitsA = Host(be, modelA.Forward(probe));

            // A fresh model with a DIFFERENT init seed, then restore the checkpoint into it.
            var modelB = new LlamaModel(ParameterContext.Create(be, 99), SmallConfig());
            float[] logitsBefore = Host(be, modelB.Forward(probe));
            int step = Trainer.Restore(Path.Combine(dir, "step-25.ckpt"), modelB, optimizer: null, be);
            float[] logitsAfter = Host(be, modelB.Forward(probe));

            Assert.Equal(25, step);
            Assert.NotEqual(logitsA, logitsBefore);          // different weights before restore
            Assert.Equal(logitsA.Length, logitsAfter.Length);
            for (int i = 0; i < logitsA.Length; i++)
                Assert.Equal(logitsA[i], logitsAfter[i]);    // bit-identical after restore
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void Checkpoint_Restore_RecoversStepAndOptimizerMoments()
    {
        using var be = new CpuComputeBackend();
        var dir = NewTempDir();
        try
        {
            var model = new LlamaModel(ParameterContext.Create(be, 7), SmallConfig());
            new Trainer(be).Train(model, new TextDataset(Corpus, new BpeTokenizer([]), 16), new TrainingConfig
            {
                BatchSize = 4, SequenceLength = 16, LearningRate = 3e-3f, MaxSteps = 30,
                WarmupSteps = 5, Seed = 2, CheckpointDirectory = dir,
            });

            var resumed = new LlamaModel(ParameterContext.Create(be, 99), SmallConfig());
            var optimizer = new AdamW(resumed.Parameters().ToList(), be);
            int step = Trainer.Restore(Path.Combine(dir, "step-30.ckpt"), resumed, optimizer, be);

            Assert.Equal(30, step);
            Assert.Equal(30, optimizer.StepCount);
            // Every parameter should have its optimizer moments restored.
            Assert.All(resumed.Parameters(), p => Assert.True(optimizer.TryGetState(p, out _)));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void GradientAccumulation_OverMicroBatches_MatchesFullBatch()
    {
        using var be = new CpuComputeBackend();
        // Two identically-initialized models (same seed → same weights) so their grads are comparable.
        var full = new LlamaModel(ParameterContext.Create(be, 5), SmallConfig());
        var accum = new LlamaModel(ParameterContext.Create(be, 5), SmallConfig());

        const int b = 4, s = 8;
        var rng = new Random(11);
        var inBuf = new float[b * s];
        var tgtBuf = new float[b * s];
        for (int i = 0; i < b * s; i++) { inBuf[i] = rng.Next(1, 259); tgtBuf[i] = rng.Next(1, 259); }

        // Full batch [4, s]: one forward/backward with the mean loss.
        var agF = new Autograd(be);
        var lossF = Loss.CrossEntropy(agF, full.Forward(
            be.FromHost(inBuf, new Shape(b, s), DType.F32)), be.FromHost(tgtBuf, new Shape(b, s), DType.F32));
        lossF.Backward();

        // Two micro-batches [2, s] each, loss scaled by 1/2, grads accumulated across both backwards.
        var agA = new Autograd(be);
        var half = be.FromHost([0.5f], new Shape(1), DType.F32);
        for (int m = 0; m < 2; m++)
        {
            var inM = new float[2 * s];
            var tgM = new float[2 * s];
            Array.Copy(inBuf, m * 2 * s, inM, 0, 2 * s);
            Array.Copy(tgtBuf, m * 2 * s, tgM, 0, 2 * s);
            var loss = Loss.CrossEntropy(agA, accum.Forward(
                be.FromHost(inM, new Shape(2, s), DType.F32)), be.FromHost(tgM, new Shape(2, s), DType.F32));
            agA.Mul(loss, half).Backward();
        }

        // Accumulated micro-batch grads must match the full-batch grads.
        var fullParams = full.NamedParameters().ToDictionary(n => n.Name, n => n.Param);
        foreach (var (name, pa) in accum.NamedParameters())
        {
            float[] gFull = Host(be, fullParams[name].Grad!);
            float[] gAcc = Host(be, pa.Grad!);
            for (int i = 0; i < gFull.Length; i++)
                Assert.True(MathF.Abs(gFull[i] - gAcc[i]) <= 1e-4f, $"{name}[{i}]: full {gFull[i]} vs accum {gAcc[i]}");
        }
    }

    [Fact]
    public void Checkpointing_RoundTripsConfigTokenizerAndWeights()
    {
        using var be = new CpuComputeBackend();
        var dir = NewTempDir();
        try
        {
            var config = SmallConfig();
            var tokenizer = new BpeTokenizer([]);
            var modelA = new LlamaModel(ParameterContext.Create(be, 4), config);
            var probe = be.FromHost([1, 2, 3, 4, 5], new Shape(1, 5), DType.F32);
            float[] logitsA = Host(be, modelA.Forward(probe));

            string path = Path.Combine(dir, "m.ckpt");
            Checkpointing.SaveModel(path, modelA, config, tokenizer, step: 42, optimizer: null, be);

            // generate rebuilds the model purely from the file — no hardcoded config.
            var (modelB, configB, tokB, step) = Checkpointing.LoadModel(path, be);
            Assert.Equal(42, step);
            Assert.Equal(config, configB);                    // record value equality across all fields
            Assert.Equal(tokenizer.VocabSize, tokB.VocabSize);
            float[] logitsB = Host(be, modelB.Forward(probe));
            for (int i = 0; i < logitsA.Length; i++) Assert.Equal(logitsA[i], logitsB[i]); // bit-identical
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void Checkpointing_RoundTripsHfTokenizer()
    {
        using var be = new CpuComputeBackend();
        var dir = NewTempDir();
        try
        {
            const string tokenizerJson =
                """
                { "model": { "type": "BPE", "vocab": { "a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4 },
                  "merges": ["a b", "ab c"] }, "added_tokens": [ { "id": 5, "content": "<eos>", "special": true } ] }
                """;
            var tokenizer = HfTokenizer.FromTokenizerJson(tokenizerJson, eosToken: "<eos>");
            var config = SmallConfig() with { VocabSize = tokenizer.VocabSize };
            var model = new LlamaModel(ParameterContext.Create(be, 3), config);

            string path = Path.Combine(dir, "hf.ckpt");
            Checkpointing.SaveModel(path, model, config, tokenizer, step: 1, optimizer: null, be);

            var (_, _, loaded, _) = Checkpointing.LoadModel(path, be);
            Assert.IsType<HfTokenizer>(loaded); // reconstructed as the right tokenizer kind
            Assert.Equal(tokenizer.Encode("abc"), loaded.Encode("abc"));
            Assert.Equal(tokenizer.EosId, loaded.EosId);
            Assert.Equal("abc", loaded.Decode(loaded.Encode("abc")));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void LoadModel_RejectsCheckpointWithoutMetadata()
    {
        using var be = new CpuComputeBackend();
        var dir = NewTempDir();
        try
        {
            // A weights-only checkpoint (what the trainer writes for resume) cannot be loaded for inference.
            var model = new LlamaModel(ParameterContext.Create(be, 1), SmallConfig());
            string path = Path.Combine(dir, "resume.ckpt");
            Checkpoint.Save(path, 5, model.NamedParameters().Select(p => (p.Name, p.Param)), be);
            Assert.Throws<InvalidDataException>(() => Checkpointing.LoadModel(path, be));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void Checkpoint_RejectsCorruptFile()
    {
        using var be = new CpuComputeBackend();
        var dir = NewTempDir();
        try
        {
            string path = Path.Combine(dir, "m.ckpt");
            var t = be.FromHost([1f, 2f, 3f, 4f, 5f, 6f], new Shape(2, 3), DType.F32);
            Checkpoint.Save(path, 1, [("w", t)], be);
            var bytes = File.ReadAllBytes(path);

            // Corrupted magic must be rejected with a clear InvalidDataException.
            var badMagic = (byte[])bytes.Clone();
            badMagic[0] ^= 0xFF;
            string badMagicPath = Path.Combine(dir, "bad.ckpt");
            File.WriteAllBytes(badMagicPath, badMagic);
            Assert.Throws<InvalidDataException>(() => Checkpoint.Load(badMagicPath, be));

            // A truncated payload (declared length exceeds the bytes present) must be rejected, not read short.
            string truncPath = Path.Combine(dir, "trunc.ckpt");
            File.WriteAllBytes(truncPath, bytes[..^4]);
            Assert.Throws<InvalidDataException>(() => Checkpoint.Load(truncPath, be));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void Checkpoint_RejectsTruncationInsideMetadata()
    {
        using var be = new CpuComputeBackend();
        var dir = NewTempDir();
        try
        {
            // SaveModel writes a sizable metadata region (config + tokenizer JSON). Truncating inside it must be
            // rejected cleanly, not decode garbage or throw a bare EndOfStreamException.
            string path = Path.Combine(dir, "m.ckpt");
            var model = new LlamaModel(ParameterContext.Create(be, 1), SmallConfig());
            Checkpointing.SaveModel(path, model, SmallConfig(), new BpeTokenizer([]), 1, optimizer: null, be);
            var bytes = File.ReadAllBytes(path);

            // Header is magic(8) + step(4) + metaLen(4) = 16 bytes; keep a few metadata bytes, drop the rest.
            string truncPath = Path.Combine(dir, "trunc-meta.ckpt");
            File.WriteAllBytes(truncPath, bytes[..20]);
            Assert.Throws<InvalidDataException>(() => Checkpoint.Load(truncPath, be));
            Assert.Throws<InvalidDataException>(() => Checkpointing.LoadModel(truncPath, be));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    private static string NewTempDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), "paitest-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }
}
