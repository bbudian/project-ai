namespace ProjectAI.Tokenizers;

/// <summary>Converts between text and integer token ids.</summary>
public interface ITokenizer
{
    int VocabSize { get; }
    IReadOnlyList<int> Encode(string text, bool addBos = false, bool addEos = false);
    string Decode(IReadOnlyList<int> ids);
}

/// <summary>Byte-Pair-Encoding tokenizer. Trained from a corpus or loaded from a vocab/merges file.</summary>
public interface IBpeTokenizer : ITokenizer
{
    int BosId { get; }
    int EosId { get; }
}

/// <summary>Byte-level BPE over UTF-8, GPT-2/Llama style. Implemented in Stage 1 (ticket S1-1).</summary>
public sealed class BpeTokenizer : IBpeTokenizer
{
    public int VocabSize => throw new NotImplementedException("ticket S1-1.");
    public int BosId => throw new NotImplementedException("ticket S1-1.");
    public int EosId => throw new NotImplementedException("ticket S1-1.");
    public IReadOnlyList<int> Encode(string text, bool addBos = false, bool addEos = false) =>
        throw new NotImplementedException("ticket S1-1.");
    public string Decode(IReadOnlyList<int> ids) => throw new NotImplementedException("ticket S1-1.");
}
