using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using ProjectAI.Models;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>
/// Pins the deterministic, registration-order contract of <see cref="Module.NamedParameters"/> /
/// <see cref="Module.Parameters"/> (S1-4 follow-up). Checkpoint IO and AdamW's reference-keyed state both
/// enumerate a live model, so the order must be a stable function of construction — not Dictionary iteration.
/// </summary>
public class ModuleOrderingTests
{
    private static ModelConfig OneLayerConfig() => new()
    {
        VocabSize = 8, EmbeddingDim = 16, LayerCount = 1, HeadCount = 2, KvHeadCount = 1,
        FeedForwardHiddenDim = 32, MaxSequenceLength = 16,
    };

    [Fact]
    public void NamedParameters_EmitsExpectedRegistrationOrder()
    {
        using var be = new CpuComputeBackend();
        var model = new LlamaModel(ParameterContext.Create(be, 1), OneLayerConfig());

        string[] expected =
        [
            "embedding.weight",
            "block.0.attn_norm.weight",
            "block.0.attn.wq.weight",
            "block.0.attn.wk.weight",
            "block.0.attn.wv.weight",
            "block.0.attn.wo.weight",
            "block.0.ffn_norm.weight",
            "block.0.ffn.gate.weight",
            "block.0.ffn.up.weight",
            "block.0.ffn.down.weight",
            "final_norm.weight",
        ];

        var actual = model.NamedParameters().Select(np => np.Name).ToArray();
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void NamedParameters_IsDeterministicAcrossConstructions()
    {
        using var be = new CpuComputeBackend();
        // Different seeds change the weight *values* but must not change the name *order*.
        var a = new LlamaModel(ParameterContext.Create(be, 1), OneLayerConfig() with { LayerCount = 3 });
        var b = new LlamaModel(ParameterContext.Create(be, 99), OneLayerConfig() with { LayerCount = 3 });

        Assert.Equal(
            a.NamedParameters().Select(np => np.Name),
            b.NamedParameters().Select(np => np.Name));
    }

    [Fact]
    public void Parameters_AlignsPositionallyWithNamedParameters()
    {
        using var be = new CpuComputeBackend();
        var model = new LlamaModel(ParameterContext.Create(be, 7), OneLayerConfig() with { LayerCount = 2 });

        var named = model.NamedParameters().ToArray();
        var bare = model.Parameters().ToArray();

        // Same count, and the SAME tensor object at each index — this is the invariant AdamW (reference-keyed
        // moment state) and positional checkpoint IO depend on.
        Assert.Equal(named.Length, bare.Length);
        for (int i = 0; i < bare.Length; i++)
            Assert.Same(named[i].Param, bare[i]);
    }

    [Fact]
    public void Enumeration_IsRepeatable_SameObjectsEachCall()
    {
        using var be = new CpuComputeBackend();
        var model = new LlamaModel(ParameterContext.Create(be, 3), OneLayerConfig());

        var first = model.NamedParameters().ToArray();
        var second = model.NamedParameters().ToArray();

        Assert.Equal(first.Length, second.Length);
        for (int i = 0; i < first.Length; i++)
        {
            Assert.Equal(first[i].Name, second[i].Name);
            Assert.Same(first[i].Param, second[i].Param); // identity stable across calls (AdamW relies on this)
        }
    }
}
