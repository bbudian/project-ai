using ProjectAI.Tokenizers;
using Xunit;

namespace ProjectAI.Tests;

public class TokenizerTests
{
    private static readonly string[] FuzzCorpus =
    [
        "Hello, world!",
        "the quick brown fox jumps over the lazy dog",
        "héllo café naïve façade",      // accents / multi-byte Latin
        "日本語のテキスト",               // CJK
        "emoji 🚀🔥😀 and symbols ©®™",   // emoji / symbols (multi-byte)
        "  leading and  trailing   ",     // whitespace runs
        "tabs\tand\nnewlines\r\n",
        "numbers 1234 and 56.78",
        "",                               // empty
        "a",                              // single char
    ];

    [Fact]
    public void Decode_Encode_RoundTrips_ByteLevel_NoMerges()
    {
        var tok = new BpeTokenizer([]); // pure byte vocab
        foreach (var s in FuzzCorpus)
            Assert.Equal(s, tok.Decode(tok.Encode(s)));
    }

    [Fact]
    public void Decode_Encode_RoundTrips_WithTrainedMerges()
    {
        var tok = BpeTrainer.Train(FuzzCorpus, vocabSize: 400);
        foreach (var s in FuzzCorpus)
            Assert.Equal(s, tok.Decode(tok.Encode(s)));
    }

    [Fact]
    public void VocabSize_IsBytesPlusSpecialsPlusMerges()
    {
        Assert.Equal(259, new BpeTokenizer([]).VocabSize);          // 256 bytes + PAD/BOS/EOS
        Assert.Equal(259 + 2, new BpeTokenizer([(97, 97), (259, 259)]).VocabSize);
    }

    [Fact]
    public void SpecialTokenIds_AreDistinct_AndWithinVocab()
    {
        var tok = new BpeTokenizer([]);
        Assert.Equal(256, tok.PadId);
        Assert.Equal(257, tok.BosId);
        Assert.Equal(258, tok.EosId);
        Assert.True(tok.BosId < tok.VocabSize && tok.EosId < tok.VocabSize && tok.PadId < tok.VocabSize);
    }

    [Fact]
    public void Encode_AddsBosEos_AndDecodeIgnoresThem()
    {
        var tok = new BpeTokenizer([]);
        var ids = tok.Encode("hi", addBos: true, addEos: true);
        Assert.Equal(tok.BosId, ids[0]);
        Assert.Equal(tok.EosId, ids[^1]);
        Assert.Equal("hi", tok.Decode(ids));            // specials carry no bytes
        Assert.Equal("hi", tok.Decode(tok.Encode("hi"))); // and without specials
    }

    [Fact]
    public void Train_ReproducesKnownMerges_OnFixedCorpus()
    {
        // "aaaa" → bytes [97,97,97,97]. Round 1 merges (97,97)→259, round 2 merges (259,259)→260.
        var tok = BpeTrainer.Train(["aaaa"], vocabSize: 261); // 2 merges
        Assert.Equal(261, tok.VocabSize);
        Assert.Equal([260], tok.Encode("aaaa"));
        Assert.Equal("aaaa", tok.Decode([260]));
    }

    [Fact]
    public void Train_IsDeterministic()
    {
        var a = BpeTrainer.Train(FuzzCorpus, vocabSize: 350);
        var b = BpeTrainer.Train(FuzzCorpus, vocabSize: 350);
        // Identical merges ⇒ identical encodings for every sample.
        foreach (var s in FuzzCorpus)
            Assert.Equal(a.Encode(s), b.Encode(s));
        Assert.Equal(a.VocabSize, b.VocabSize);
    }

    [Fact]
    public void EncodedIds_AreWithinVocab()
    {
        var tok = BpeTrainer.Train(FuzzCorpus, vocabSize: 400);
        foreach (var s in FuzzCorpus)
            foreach (var id in tok.Encode(s, addBos: true, addEos: true))
                Assert.InRange(id, 0, tok.VocabSize - 1);
    }

    [Fact]
    public void JsonRoundTrip_PreservesEncoding()
    {
        var trained = BpeTrainer.Train(FuzzCorpus, vocabSize: 400);
        var reloaded = BpeTokenizer.FromJson(trained.ToJson());

        Assert.Equal(trained.VocabSize, reloaded.VocabSize);
        foreach (var s in FuzzCorpus)
        {
            Assert.Equal(trained.Encode(s), reloaded.Encode(s));
            Assert.Equal(s, reloaded.Decode(reloaded.Encode(s)));
        }
    }

    [Fact]
    public void Encode_RejectsIllFormedUtf16()
    {
        var tok = new BpeTokenizer([]);
        // Lone/unpaired surrogates are not valid text — fail loudly rather than silently corrupt to U+FFFD.
        Assert.Throws<System.Text.EncoderFallbackException>(() => tok.Encode("a\uD800b"));
        Assert.Throws<System.Text.EncoderFallbackException>(() => tok.Encode("\uDC00"));
        // A valid surrogate pair (emoji) is well-formed and round-trips fine.
        Assert.Equal("😀", tok.Decode(tok.Encode("😀")));
    }

    [Fact]
    public void Train_TieBreak_PrefersSmallestPair_Deterministically()
    {
        // "ab" and "cd" each occur once → pairs (97,98) and (99,100) tie. Smallest-pair tie-break picks
        // (97,98) first, so with a 1-merge budget only "ab" collapses to a single token — regardless of order.
        foreach (var corpus in new[] { new[] { "ab", "cd" }, ["cd", "ab"] })
        {
            var tok = BpeTrainer.Train(corpus, vocabSize: 260); // 1 merge
            Assert.Equal([259], tok.Encode("ab"));
            Assert.Equal([99, 100], tok.Encode("cd"));
        }
    }

    [Fact]
    public void Train_AggregatesPairCountsAcrossStrings()
    {
        // "ab" recurs across three separate corpus entries (count 3) and beats "cd" (count 1) only if
        // counts are aggregated across the whole IEnumerable.
        var tok = BpeTrainer.Train(["ab", "ab", "ab", "cd"], vocabSize: 260); // 1 merge
        Assert.Equal([259], tok.Encode("ab"));
        Assert.Equal([99, 100], tok.Encode("cd"));
    }

    [Fact]
    public void Constructor_RejectsForwardReferencingMerge()
    {
        // Token id 259 doesn't exist until the first merge creates it, so (259,97) as merge 0 is invalid.
        Assert.Throws<ArgumentException>(() => new BpeTokenizer([(259, 97)]));
    }

    [Fact]
    public void FromJson_RejectsMalformedMergePair()
    {
        Assert.Throws<FormatException>(() => BpeTokenizer.FromJson("{\"Merges\":[[1,2,3]]}"));
    }

    [Fact]
    public void MergesActuallyShortenEncoding()
    {
        const string text = "the quick brown fox the quick brown fox";
        var bytes = new BpeTokenizer([]).Encode(text).Count;
        var trained = BpeTrainer.Train([text], vocabSize: 320).Encode(text).Count;
        Assert.True(trained < bytes, $"trained encoding ({trained}) should be shorter than byte encoding ({bytes})");
    }
}
