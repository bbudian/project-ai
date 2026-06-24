using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;
using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using ProjectAI.Models;
using ProjectAI.Training;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>Tests for the S1-11 HuggingFace → LlamaModel converter: config mapping and a weight round-trip.</summary>
public class ConverterTests
{
    private static float[] Host(CpuComputeBackend be, Tensor t)
    {
        var buf = new float[t.ElementCount];
        be.ToHost(t, buf);
        return buf;
    }

    [Fact]
    public void ConfigFromHuggingFace_MapsFields()
    {
        string json =
            """
            { "vocab_size": 320, "hidden_size": 64, "num_hidden_layers": 3, "num_attention_heads": 8,
              "num_key_value_heads": 2, "intermediate_size": 172, "max_position_embeddings": 2048,
              "rope_theta": 500000.0, "rms_norm_eps": 1e-6 }
            """;
        var c = Converter.ConfigFromHuggingFace(json);
        Assert.Equal(320, c.VocabSize);
        Assert.Equal(64, c.EmbeddingDim);
        Assert.Equal(3, c.LayerCount);
        Assert.Equal(8, c.HeadCount);
        Assert.Equal(2, c.KvHeadCount);
        Assert.Equal(172, c.FeedForwardHiddenDim);
        Assert.Equal(2048, c.MaxSequenceLength);
        Assert.Equal(500000f, c.RopeTheta);
        Assert.Equal(1e-6f, c.NormEpsilon);
    }

    [Fact]
    public void ConfigFromHuggingFace_DefaultsKvHeadsAndContext()
    {
        var c = Converter.ConfigFromHuggingFace(
            """{ "vocab_size": 16, "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 4, "intermediate_size": 16 }""");
        Assert.Equal(4, c.KvHeadCount);       // num_key_value_heads absent → multi-head
        Assert.Equal(4096, c.MaxSequenceLength); // max_position_embeddings absent → default
    }

    [Fact]
    public void Load_RoundTrips_HuggingFaceFormat()
    {
        using var be = new CpuComputeBackend();
        var dir = NewTempDir();
        try
        {
            var config = TinyConfig();
            var reference = new LlamaModel(ParameterContext.Create(be, 4), config);
            File.WriteAllText(Path.Combine(dir, "config.json"), HfConfigJson(config));
            WriteSafetensors(be, dir, reference);

            var (converted, loadedConfig) = Converter.Load(dir, be);
            Assert.Equal(config, loadedConfig);

            // Same weights copied 1:1 → the converted model must reproduce the reference forward bit-for-bit.
            var probe = be.FromHost([1, 2, 3, 4, 5], new Shape(1, 5), DType.F32);
            float[] refLogits, gotLogits;
            using (GradMode.NoGrad())
            {
                refLogits = Host(be, reference.Forward(probe));
                gotLogits = Host(be, converted.Forward(probe));
            }
            for (int i = 0; i < refLogits.Length; i++) Assert.Equal(refLogits[i], gotLogits[i]);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void Load_RejectsAttentionBias()
    {
        using var be = new CpuComputeBackend();
        var dir = NewTempDir();
        try
        {
            var config = TinyConfig() with { LayerCount = 1 };
            var reference = new LlamaModel(ParameterContext.Create(be, 4), config);
            File.WriteAllText(Path.Combine(dir, "config.json"), HfConfigJson(config));
            // Add a stray attention-bias tensor — our attention is bias-free, so convert must refuse.
            WriteSafetensors(be, dir, reference, ("model.layers.0.self_attn.q_proj.bias", [config.EmbeddingDim], new float[config.EmbeddingDim]));
            Assert.Throws<NotSupportedException>(() => Converter.Load(dir, be));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void Load_RejectsUntiedEmbeddings()
    {
        using var be = new CpuComputeBackend();
        var dir = NewTempDir();
        try
        {
            var config = TinyConfig() with { LayerCount = 1 };
            var reference = new LlamaModel(ParameterContext.Create(be, 4), config);
            string json =
                $$"""
                { "vocab_size": {{config.VocabSize}}, "hidden_size": {{config.EmbeddingDim}}, "num_hidden_layers": {{config.LayerCount}},
                  "num_attention_heads": {{config.HeadCount}}, "num_key_value_heads": {{config.KvHeadCount}},
                  "intermediate_size": {{config.FeedForwardHiddenDim}}, "max_position_embeddings": {{config.MaxSequenceLength}},
                  "tie_word_embeddings": false }
                """;
            File.WriteAllText(Path.Combine(dir, "config.json"), json);
            WriteSafetensors(be, dir, reference);
            Assert.Throws<NotSupportedException>(() => Converter.Load(dir, be)); // our LM head is tied
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void Load_RejectsQkNorm()
    {
        using var be = new CpuComputeBackend();
        var dir = NewTempDir();
        try
        {
            var config = TinyConfig() with { LayerCount = 1 };
            var reference = new LlamaModel(ParameterContext.Create(be, 4), config);
            File.WriteAllText(Path.Combine(dir, "config.json"), HfConfigJson(config));
            int headDim = config.EmbeddingDim / config.HeadCount;
            WriteSafetensors(be, dir, reference, ("model.layers.0.self_attn.q_norm.weight", [headDim], new float[headDim]));
            Assert.Throws<NotSupportedException>(() => Converter.Load(dir, be)); // QK-norm not supported
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void Load_RejectsRopeScaling()
    {
        using var be = new CpuComputeBackend();
        var dir = NewTempDir();
        try
        {
            var config = TinyConfig() with { LayerCount = 1 };
            var reference = new LlamaModel(ParameterContext.Create(be, 4), config);
            string json =
                $$"""
                { "vocab_size": {{config.VocabSize}}, "hidden_size": {{config.EmbeddingDim}}, "num_hidden_layers": {{config.LayerCount}},
                  "num_attention_heads": {{config.HeadCount}}, "num_key_value_heads": {{config.KvHeadCount}},
                  "intermediate_size": {{config.FeedForwardHiddenDim}}, "max_position_embeddings": {{config.MaxSequenceLength}},
                  "rope_scaling": { "rope_type": "llama3", "factor": 8.0 } }
                """;
            File.WriteAllText(Path.Combine(dir, "config.json"), json);
            WriteSafetensors(be, dir, reference);
            Assert.Throws<NotSupportedException>(() => Converter.Load(dir, be)); // RoPE scaling not implemented
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void Load_RejectsShippedLmHead()
    {
        using var be = new CpuComputeBackend();
        var dir = NewTempDir();
        try
        {
            var config = TinyConfig() with { LayerCount = 1 };
            var reference = new LlamaModel(ParameterContext.Create(be, 4), config);
            File.WriteAllText(Path.Combine(dir, "config.json"), HfConfigJson(config));
            WriteSafetensors(be, dir, reference, ("lm_head.weight", [config.VocabSize, config.EmbeddingDim], new float[config.VocabSize * config.EmbeddingDim]));
            Assert.Throws<NotSupportedException>(() => Converter.Load(dir, be)); // untied LM head
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    private static ModelConfig TinyConfig() => new()
    {
        VocabSize = 16, EmbeddingDim = 8, LayerCount = 2, HeadCount = 2, KvHeadCount = 1,
        FeedForwardHiddenDim = 16, MaxSequenceLength = 16,
    };

    private static string HfConfigJson(ModelConfig c) =>
        $$"""
        { "vocab_size": {{c.VocabSize}}, "hidden_size": {{c.EmbeddingDim}}, "num_hidden_layers": {{c.LayerCount}},
          "num_attention_heads": {{c.HeadCount}}, "num_key_value_heads": {{c.KvHeadCount}},
          "intermediate_size": {{c.FeedForwardHiddenDim}}, "max_position_embeddings": {{c.MaxSequenceLength}} }
        """;

    // Writes the model's weights (under HuggingFace tensor names) plus optional extra tensors as a real
    // safetensors file: u64 header length, JSON header, then tightly-packed F32 data.
    private static void WriteSafetensors(CpuComputeBackend be, string dir, LlamaModel model, params (string Name, int[] Shape, float[] Data)[] extra)
    {
        var tensors = new List<(string Name, int[] Shape, byte[] Data)>();
        foreach (var (name, p) in model.NamedParameters())
            tensors.Add((Converter.ToHuggingFaceName(name), p.Shape.Dimensions.ToArray(), MemoryMarshal.AsBytes(Host(be, p).AsSpan()).ToArray()));
        foreach (var (name, shape, data) in extra)
            tensors.Add((name, shape, MemoryMarshal.AsBytes(data.AsSpan()).ToArray()));

        var header = new StringBuilder("{");
        var chunks = new List<byte[]>();
        long offset = 0;
        for (int i = 0; i < tensors.Count; i++)
        {
            var (name, shape, bytes) = tensors[i];
            if (i > 0) header.Append(',');
            header.Append($"\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{string.Join(",", shape)}],\"data_offsets\":[{offset},{offset + bytes.Length}]}}");
            chunks.Add(bytes);
            offset += bytes.Length;
        }
        header.Append('}');

        using var ms = new MemoryStream();
        Span<byte> length = stackalloc byte[8];
        var headerBytes = Encoding.UTF8.GetBytes(header.ToString());
        BinaryPrimitives.WriteUInt64LittleEndian(length, (ulong)headerBytes.Length);
        ms.Write(length);
        ms.Write(headerBytes);
        foreach (var chunk in chunks) ms.Write(chunk);
        File.WriteAllBytes(Path.Combine(dir, "model.safetensors"), ms.ToArray());
    }

    private static string NewTempDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), "paiconv-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }
}
