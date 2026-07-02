using System.Text.Json;
using ProjectAI.Core;
using ProjectAI.Formats;
using ProjectAI.Models;
using ProjectAI.Tokenizers;

namespace ProjectAI.Training;

/// <summary>
/// Converts a HuggingFace Llama-style checkpoint (a <c>config.json</c> + one or more <c>.safetensors</c> files)
/// into this runtime's <see cref="LlamaModel"/> — the S1-11 <c>convert</c> path. It maps the HF config fields to
/// <see cref="ModelConfig"/> and remaps HF tensor names onto our parameter names; the weights copy directly
/// because our RoPE uses the same rotate-half convention HF does (no permutation needed).
/// <para>
/// Constraints (the only shape this runtime's model has): Llama architecture, bias-free attention, a SiLU/SwiGLU
/// FFN, RMSNorm, and an embedding tied to the LM head. The model's TOKENIZER is a separate concern — without
/// loading it, generated text will not be meaningful (the vocabularies won't match).
/// </para>
/// </summary>
public static class Converter
{
    private static readonly SafetensorsLoader Loader = new();

    /// <summary>
    /// Loads an HF model directory (or a single .safetensors with an adjacent config.json) into a model at the given
    /// precision. Pass <paramref name="computeDType"/> = BF16 to load + run at half memory (S3-1) so a larger model fits.
    /// </summary>
    public static (LlamaModel Model, ModelConfig Config) Load(string path, IComputeBackend backend, DType computeDType = DType.F32)
    {
        string directory = Directory.Exists(path) ? path : Path.GetDirectoryName(Path.GetFullPath(path))!;
        string configPath = Path.Combine(directory, "config.json");
        if (!File.Exists(configPath))
            throw new FileNotFoundException($"no config.json found in '{directory}'.");

        string configJson = File.ReadAllText(configPath);
        var config = ConfigFromHuggingFace(configJson);
        config.Validate();

        var weights = LoadWeights(path, directory, backend, computeDType);
        CheckCompatibility(configJson, weights);

        // SkipInit: CopyWeights overwrites every parameter (and throws on a missing one), so the eager Gaussian
        // init would be pure waste — on a billion-parameter convert it is the dominant cost.
        var model = new LlamaModel(ParameterContext.Create(backend, 0, computeDType) with { SkipInit = true }, config);
        CopyWeights(weights, model, backend);
        return (model, config);
    }

    /// <summary>Loads the model's <c>tokenizer.json</c> (byte-level BPE) if present, else null. EOS/BOS ids come from <c>config.json</c>.</summary>
    public static HfTokenizer? TryLoadTokenizer(string path)
    {
        string directory = Directory.Exists(path) ? path : Path.GetDirectoryName(Path.GetFullPath(path))!;
        string tokenizerPath = Path.Combine(directory, "tokenizer.json");
        if (!File.Exists(tokenizerPath)) return null;

        int eosId = -1, bosId = -1;
        string configPath = Path.Combine(directory, "config.json");
        if (File.Exists(configPath))
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(configPath));
            var root = doc.RootElement;
            if (root.TryGetProperty("eos_token_id", out var e) && e.ValueKind == JsonValueKind.Number && e.TryGetInt32(out int ev)) eosId = ev;
            if (root.TryGetProperty("bos_token_id", out var b) && b.ValueKind == JsonValueKind.Number && b.TryGetInt32(out int bv)) bosId = bv;
        }
        return HfTokenizer.FromTokenizerJson(File.ReadAllText(tokenizerPath), eosTokenId: eosId, bosTokenId: bosId);
    }

    /// <summary>Maps an HF Llama <c>config.json</c> to a <see cref="ModelConfig"/>.</summary>
    public static ModelConfig ConfigFromHuggingFace(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        int Require(string key) =>
            root.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number && v.TryGetInt32(out int n)
                ? n
                : throw new InvalidDataException($"config.json is missing the integer '{key}'.");
        int OptInt(string key, int fallback) =>
            root.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number && v.TryGetInt32(out int n) ? n : fallback;
        float OptFloat(string key, float fallback) =>
            root.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetSingle() : fallback;

        int heads = Require("num_attention_heads");
        return new ModelConfig
        {
            VocabSize = Require("vocab_size"),
            EmbeddingDim = Require("hidden_size"),
            LayerCount = Require("num_hidden_layers"),
            HeadCount = heads,
            KvHeadCount = OptInt("num_key_value_heads", heads), // absent ⇒ multi-head (no GQA)
            FeedForwardHiddenDim = Require("intermediate_size"),
            MaxSequenceLength = OptInt("max_position_embeddings", 4096),
            RopeTheta = OptFloat("rope_theta", 10000f),
            NormEpsilon = OptFloat("rms_norm_eps", 1e-5f),
        };
    }

    /// <summary>Our parameter name → the HuggingFace Llama tensor name.</summary>
    public static string ToHuggingFaceName(string parameterName)
    {
        switch (parameterName)
        {
            case "embedding.weight": return "model.embed_tokens.weight";
            case "final_norm.weight": return "model.norm.weight";
        }

        const string prefix = "block.";
        if (!parameterName.StartsWith(prefix, StringComparison.Ordinal))
            throw new InvalidDataException($"no HuggingFace mapping for parameter '{parameterName}'.");
        int dot = parameterName.IndexOf('.', prefix.Length);
        string index = parameterName[prefix.Length..dot];
        string rest = parameterName[(dot + 1)..];
        string hf = rest switch
        {
            "attn_norm.weight" => "input_layernorm.weight",
            "attn.wq.weight" => "self_attn.q_proj.weight",
            "attn.wk.weight" => "self_attn.k_proj.weight",
            "attn.wv.weight" => "self_attn.v_proj.weight",
            "attn.wo.weight" => "self_attn.o_proj.weight",
            "ffn_norm.weight" => "post_attention_layernorm.weight",
            "ffn.gate.weight" => "mlp.gate_proj.weight",
            "ffn.up.weight" => "mlp.up_proj.weight",
            "ffn.down.weight" => "mlp.down_proj.weight",
            _ => throw new InvalidDataException($"no HuggingFace mapping for block parameter '{rest}'."),
        };
        return $"model.layers.{index}.{hf}";
    }

    private static StateDict LoadWeights(string path, string directory, IComputeBackend backend, DType targetDType)
    {
        if (!Directory.Exists(path) && path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
            return Loader.Load(path, backend, targetDType);

        string single = Path.Combine(directory, "model.safetensors");
        if (File.Exists(single)) return Loader.Load(single, backend, targetDType);

        var shards = Directory.GetFiles(directory, "*.safetensors");
        if (shards.Length == 0) throw new FileNotFoundException($"no .safetensors files in '{directory}'.");

        var merged = new StateDict();
        foreach (var shard in shards.OrderBy(s => s, StringComparer.Ordinal))
            foreach (var (name, tensor) in Loader.Load(shard, backend, targetDType).Tensors)
                merged[name] = tensor;
        return merged;
    }

    // Fails loudly on architectures this runtime's fixed model shape can't represent — these would otherwise
    // "convert" but compute silently-wrong results.
    private static void CheckCompatibility(string configJson, StateDict weights)
    {
        using var doc = JsonDocument.Parse(configJson);
        var root = doc.RootElement;

        if (root.TryGetProperty("hidden_act", out var act) && act.ValueKind == JsonValueKind.String &&
            !string.Equals(act.GetString(), "silu", StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException($"this runtime's FFN is SwiGLU/SiLU; the model uses hidden_act='{act.GetString()}'.");

        // RoPE scaling (llama3/linear/dynamic) rescales the frequency table; we only build plain RoPE, so a
        // scaled model would mis-position tokens (Llama-3.1/3.2/3.3 all ship this). Fail until it's implemented.
        if (root.TryGetProperty("rope_scaling", out var rope) && rope.ValueKind != JsonValueKind.Null)
        {
            string type = rope.TryGetProperty("rope_type", out var rt) && rt.ValueKind == JsonValueKind.String ? rt.GetString()! : "unknown";
            throw new NotSupportedException($"RoPE scaling (rope_scaling type='{type}') is not implemented; this runtime would compute wrong positions.");
        }

        if (root.TryGetProperty("tie_word_embeddings", out var tie) && tie.ValueKind == JsonValueKind.False)
            throw new NotSupportedException("the model has untied embeddings (tie_word_embeddings=false); this runtime ties the LM head to the embedding.");

        foreach (var name in weights.Tensors.Keys)
        {
            if (name.Contains("self_attn", StringComparison.Ordinal) && name.EndsWith(".bias", StringComparison.Ordinal))
                throw new NotSupportedException($"this runtime's attention is bias-free, but the checkpoint has '{name}'.");
            if (name.Contains("q_norm", StringComparison.Ordinal) || name.Contains("k_norm", StringComparison.Ordinal))
                throw new NotSupportedException($"QK-normalization is not supported, but the checkpoint has '{name}'.");
            // A shipped lm_head.weight means untied output embeddings (regardless of the config flag); the
            // pull-based copy below would never read it and silently use the input embedding instead.
            if (name == "lm_head.weight")
                throw new NotSupportedException("the checkpoint ships a separate lm_head.weight (untied); this runtime ties the LM head to the embedding.");
        }
    }

    private static void CopyWeights(StateDict weights, LlamaModel model, IComputeBackend backend)
    {
        foreach (var (name, param) in model.NamedParameters())
        {
            string hf = ToHuggingFaceName(name);
            if (!weights.TryGet(hf, out var source))
                throw new InvalidDataException($"checkpoint is missing '{hf}' (needed for '{name}').");
            if (!source.Shape.Equals(param.Shape))
                throw new InvalidDataException($"'{hf}' shape {source.Shape} != model parameter '{name}' shape {param.Shape}.");
            backend.Copy(source, param);
        }
    }
}
