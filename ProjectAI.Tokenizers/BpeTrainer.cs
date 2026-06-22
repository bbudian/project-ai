namespace ProjectAI.Tokenizers;

/// <summary>Trains a byte-level BPE tokenizer from a text corpus (ticket S1-1).</summary>
public static class BpeTrainer
{
    /// <summary>
    /// Learns BPE merges from <paramref name="corpus"/> until the vocabulary reaches
    /// <paramref name="vocabSize"/> (256 byte tokens + 3 specials + merges, so the merge budget is
    /// <c>vocabSize - 259</c>). Deterministic: the most frequent adjacent pair is merged each round, with
    /// ties broken by the lexicographically smallest (a, b) pair, so a fixed corpus reproduces the merges.
    /// </summary>
    public static BpeTokenizer Train(IEnumerable<string> corpus, int vocabSize)
    {
        int mergeBudget = Math.Max(0, vocabSize - BpeTokenizer.FirstMergeId);

        // Pre-tokenize, then collapse to unique chunks with counts; each chunk becomes a byte-id list.
        var chunkCounts = new Dictionary<string, int>();
        foreach (var text in corpus)
            foreach (var chunk in BpeTokenizer.Pretokenize(text))
                chunkCounts[chunk] = chunkCounts.GetValueOrDefault(chunk) + 1;

        var words = new List<(List<int> Tokens, int Count)>();
        foreach (var (chunk, count) in chunkCounts)
        {
            var bytes = BpeTokenizer.EncodeUtf8(chunk);
            if (bytes.Length == 0) continue;
            var tokens = new List<int>(bytes.Length);
            foreach (var b in bytes) tokens.Add(b);
            words.Add((tokens, count));
        }

        var merges = new List<(int A, int B)>();
        int nextId = BpeTokenizer.FirstMergeId;
        for (int step = 0; step < mergeBudget; step++)
        {
            var pairCounts = new Dictionary<(int, int), int>();
            foreach (var (tokens, count) in words)
                for (int i = 0; i < tokens.Count - 1; i++)
                {
                    var pair = (tokens[i], tokens[i + 1]);
                    pairCounts[pair] = pairCounts.GetValueOrDefault(pair) + count;
                }
            if (pairCounts.Count == 0) break; // nothing left to merge

            // Pick highest count; deterministic tie-break by smallest pair.
            var best = default((int, int));
            int bestCount = -1;
            foreach (var kv in pairCounts)
                if (kv.Value > bestCount || (kv.Value == bestCount && ComparePair(kv.Key, best) < 0))
                {
                    best = kv.Key;
                    bestCount = kv.Value;
                }

            merges.Add(best);
            int newId = nextId++;
            foreach (var (tokens, _) in words) ApplyMerge(tokens, best, newId);
        }

        return new BpeTokenizer(merges);
    }

    private static int ComparePair((int A, int B) x, (int A, int B) y)
        => x.A != y.A ? x.A.CompareTo(y.A) : x.B.CompareTo(y.B);

    /// <summary>Replaces every non-overlapping left-to-right occurrence of <paramref name="pair"/> in place.</summary>
    private static void ApplyMerge(List<int> tokens, (int A, int B) pair, int newId)
    {
        for (int i = 0; i < tokens.Count - 1;)
        {
            if (tokens[i] == pair.A && tokens[i + 1] == pair.B)
            {
                tokens[i] = newId;
                tokens.RemoveAt(i + 1);
            }
            else
            {
                i++;
            }
        }
    }
}
