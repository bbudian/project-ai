using System.Text.Json;
using ProjectAI.Core;
using ProjectAI.Formats;
using ProjectAI.Models;
using ProjectAI.Tokenizers;

namespace ProjectAI.Training;

/// <summary>
/// Saves and loads a self-describing model checkpoint: the trained weights together with the
/// <see cref="ModelConfig"/> and tokenizer needed to rebuild the model for inference. This is what lets
/// <c>generate</c> reload a model without being told its architecture — the config is read from the file,
/// not hardcoded — and turns an architecture mismatch into a clear error instead of a silently-wrong load.
/// </summary>
public static class Checkpointing
{
    // TokenizerKind tags which tokenizer the metadata holds so it reconstructs polymorphically.
    private sealed record Meta(ModelConfig Config, string TokenizerKind, string Tokenizer);

    /// <summary>Writes weights + optimizer moments (if supplied) plus the config/tokenizer metadata for inference reload.</summary>
    public static void SaveModel(
        string path, LlamaModel model, ModelConfig config, ITokenizer tokenizer, int step,
        AdamW? optimizer, IComputeBackend backend)
    {
        var (kind, json) = tokenizer switch
        {
            HfTokenizer hf => ("hf", hf.ToStateJson()),
            BpeTokenizer bpe => ("bpe", bpe.ToJson()),
            _ => throw new NotSupportedException($"can't persist tokenizer of type {tokenizer.GetType().Name}."),
        };
        string metadata = JsonSerializer.Serialize(new Meta(config, kind, json));
        Checkpoint.Save(path, step, EnumerateState(model, optimizer), backend, metadata);
    }

    /// <summary>
    /// Rebuilds a model from a checkpoint's stored config, restores its weights, and reconstructs the tokenizer.
    /// Returns the model, its config, the tokenizer, and the saved step. Throws if the file carries no metadata
    /// (e.g. a resume-only checkpoint written by the trainer) or fails config validation.
    /// </summary>
    public static (LlamaModel Model, ModelConfig Config, ITokenizer Tokenizer, int Step) LoadModel(string path, IComputeBackend backend)
    {
        var (step, metadataJson, dict) = Checkpoint.Load(path, backend);
        if (string.IsNullOrEmpty(metadataJson))
            throw new InvalidDataException($"checkpoint '{path}' has no model metadata; it cannot be loaded for inference (was it saved by `train`?).");

        var meta = JsonSerializer.Deserialize<Meta>(metadataJson)
            ?? throw new InvalidDataException($"checkpoint '{path}' has unreadable model metadata.");
        meta.Config.Validate();

        ITokenizer tokenizer = meta.TokenizerKind switch
        {
            "hf" => HfTokenizer.FromState(meta.Tokenizer),
            "bpe" => BpeTokenizer.FromJson(meta.Tokenizer),
            _ => throw new InvalidDataException($"checkpoint '{path}' has unknown tokenizer kind '{meta.TokenizerKind}'."),
        };
        var model = new LlamaModel(ParameterContext.Create(backend, 0), meta.Config);
        ApplyWeights(dict, model, backend);
        return (model, meta.Config, tokenizer, step);
    }

    // Model parameters by name, each followed by its optimizer moments under "opt.m::"/"opt.v::" keys.
    internal static IEnumerable<(string Name, Tensor Tensor)> EnumerateState(LlamaModel model, AdamW? optimizer)
    {
        foreach (var (name, p) in model.NamedParameters())
        {
            yield return (name, p);
            if (optimizer is not null && optimizer.TryGetState(p, out var s))
            {
                yield return ($"opt.m::{name}", s.M);
                yield return ($"opt.v::{name}", s.V);
            }
        }
    }

    // Copies saved weights into the model's parameters in place (shape-checked), so forwards reproduce the saved logits.
    internal static void ApplyWeights(StateDict dict, LlamaModel model, IComputeBackend backend)
    {
        foreach (var (name, p) in model.NamedParameters())
        {
            if (!dict.TryGet(name, out var saved))
                throw new InvalidDataException($"checkpoint is missing parameter '{name}'.");
            if (!saved.Shape.Equals(p.Shape))
                throw new InvalidDataException($"checkpoint parameter '{name}' shape {saved.Shape} != model {p.Shape}.");
            backend.Copy(saved, p);
        }
    }

    // Restores AdamW moments and the timestep. Exact when every parameter is updated every step (dense LM
    // training); a model with conditionally-updated parameters would need per-parameter timesteps persisted.
    internal static void ApplyOptimizer(StateDict dict, LlamaModel model, AdamW optimizer, int step)
    {
        foreach (var (name, p) in model.NamedParameters())
            if (dict.TryGet($"opt.m::{name}", out var m) && dict.TryGet($"opt.v::{name}", out var v))
                optimizer.LoadState(p, new AdamW.MomentState(m, v, step));
        optimizer.SetStepCount(step);
    }
}
