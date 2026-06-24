using ProjectAI.Tokenizers;

namespace ProjectAI.Training;

/// <summary>
/// Packs a corpus into fixed-length next-token training blocks (ticket S1-10). The text is tokenized once,
/// then chopped into contiguous, non-overlapping blocks of <c>sequenceLength + 1</c> ids; the trainer reads
/// each block as input = block[..^1], target = block[1..]. A trailing remainder shorter than a full block is
/// dropped. This is the standard "packed" LM dataset: every token (except the dropped tail) is both a target
/// once and a context for later tokens.
/// </summary>
public sealed class TextDataset : IDataset
{
    private readonly int[][] _blocks;

    /// <summary>The model context length; each emitted sequence has <c>SequenceLength + 1</c> ids.</summary>
    public int SequenceLength { get; }

    public TextDataset(string text, ITokenizer tokenizer, int sequenceLength)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (sequenceLength < 1) throw new ArgumentOutOfRangeException(nameof(sequenceLength), "must be >= 1.");
        SequenceLength = sequenceLength;

        var ids = tokenizer.Encode(text);
        int blockLen = sequenceLength + 1;
        int blockCount = ids.Count / blockLen;
        if (blockCount == 0)
            throw new ArgumentException(
                $"corpus has {ids.Count} tokens but needs at least {blockLen} for one block of sequence length {sequenceLength}.",
                nameof(text));

        _blocks = new int[blockCount][];
        for (int b = 0; b < blockCount; b++)
        {
            var block = new int[blockLen];
            for (int i = 0; i < blockLen; i++) block[i] = ids[b * blockLen + i];
            _blocks[b] = block;
        }
    }

    public int Count => _blocks.Length;

    public ReadOnlyMemory<int> GetSequence(int index) => _blocks[index];
}
