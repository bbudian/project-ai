using System.Text.Json;
using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using ProjectAI.Formats;
using ProjectAI.Models;
using ProjectAI.Tokenizers;
using ProjectAI.Training;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>
/// Back-compat: checkpoints written before the tokenizer-kind tag existed stored only the tokenizer JSON
/// (no <c>TokenizerKind</c>). The only tokenizer at that time was byte-level BPE, so <see cref="Checkpointing.LoadModel"/>
/// must treat an absent/empty kind as "bpe" rather than throwing "unknown tokenizer kind ''".
/// </summary>
public class CheckpointBackCompatTests
{
    private static ModelConfig SmallConfig() => new()
    {
        VocabSize = 259, EmbeddingDim = 32, LayerCount = 1, HeadCount = 2, KvHeadCount = 1,
        FeedForwardHiddenDim = 64, MaxSequenceLength = 32,
    };

    [Fact]
    public void LoadModel_TreatsMissingTokenizerKind_AsBpe()
    {
        using var be = new CpuComputeBackend();
        var config = SmallConfig();
        var model = new LlamaModel(ParameterContext.Create(be, 1), config);

        // Simulate a pre-kind checkpoint: metadata with Config + Tokenizer but NO TokenizerKind field.
        string tokenizerJson = new BpeTokenizer([]).ToJson();
        string oldMetadata = JsonSerializer.Serialize(new { Config = config, Tokenizer = tokenizerJson });

        string path = Path.Combine(Path.GetTempPath(), $"backcompat-{Guid.NewGuid():N}.ckpt");
        try
        {
            Checkpoint.Save(path, step: 5, model.NamedParameters().Select(np => (np.Name, np.Param)), be, oldMetadata);

            var (_, loadedConfig, tokenizer, step) = Checkpointing.LoadModel(path, be);

            Assert.IsType<BpeTokenizer>(tokenizer);
            Assert.Equal(config, loadedConfig);
            Assert.Equal(5, step);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void LoadModel_StillRejects_UnknownTokenizerKind()
    {
        using var be = new CpuComputeBackend();
        var config = SmallConfig();
        var model = new LlamaModel(ParameterContext.Create(be, 1), config);

        string badMetadata = JsonSerializer.Serialize(new { Config = config, TokenizerKind = "sentencepiece", Tokenizer = "{}" });
        string path = Path.Combine(Path.GetTempPath(), $"badkind-{Guid.NewGuid():N}.ckpt");
        try
        {
            Checkpoint.Save(path, step: 1, model.NamedParameters().Select(np => (np.Name, np.Param)), be, badMetadata);
            var ex = Assert.Throws<InvalidDataException>(() => Checkpointing.LoadModel(path, be));
            Assert.Contains("sentencepiece", ex.Message);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
