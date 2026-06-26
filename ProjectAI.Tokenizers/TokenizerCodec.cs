namespace ProjectAI.Tokenizers;

/// <summary>
/// Serializes an <see cref="ITokenizer"/> to a <c>(kind, state-json)</c> pair and back, so a dataset manifest or
/// checkpoint can record exactly which tokenizer produced a token stream and reconstruct it on load. The kind tag
/// (<c>"bpe"</c>/<c>"hf"</c>) is what makes reconstruction polymorphic; an absent/empty kind means byte-level BPE
/// (the only tokenizer that existed before the tag), so older artifacts still load. This is the single home for
/// that mapping — the same switch the checkpoint loader uses, lifted so the data path can share it.
/// </summary>
public static class TokenizerCodec
{
    /// <summary>Returns the persistence tag and serialized state for a tokenizer.</summary>
    public static (string Kind, string State) Serialize(ITokenizer tokenizer) => tokenizer switch
    {
        HfTokenizer hf => ("hf", hf.ToStateJson()),
        BpeTokenizer bpe => ("bpe", bpe.ToJson()),
        _ => throw new NotSupportedException($"can't persist tokenizer of type {tokenizer.GetType().Name}."),
    };

    /// <summary>Rebuilds a tokenizer from its kind tag and serialized state (absent/empty kind ⇒ byte-level BPE).</summary>
    public static ITokenizer Deserialize(string? kind, string state) => kind switch
    {
        "hf" => HfTokenizer.FromState(state),
        "bpe" or null or "" => BpeTokenizer.FromJson(state),
        _ => throw new NotSupportedException($"unknown tokenizer kind '{kind}'."),
    };
}
