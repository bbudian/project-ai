using ProjectAI.Tokenizers;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>Tests for the HF byte-level BPE tokenizer loader (validated synthetically against a hand-built tokenizer.json).</summary>
public class HfTokenizerTests
{
    // A tiny byte-level BPE tokenizer: single-char tokens a/b/c, two merges (a+b→ab, ab+c→abc), plus an EOS.
    private const string TokenizerJson =
        """
        {
          "model": {
            "type": "BPE",
            "vocab": { "a": 0, "b": 1, "c": 2, "d": 3, "ab": 4, "abc": 5 },
            "merges": ["a b", "ab c"]
          },
          "added_tokens": [ { "id": 6, "content": "<eos>", "special": true } ]
        }
        """;

    [Fact]
    public void EncodesByMergeRank()
    {
        var tok = HfTokenizer.FromTokenizerJson(TokenizerJson, eosToken: "<eos>");
        Assert.Equal(new[] { 5 }, tok.Encode("abc"));    // a+b→ab, then ab+c→abc
        Assert.Equal(new[] { 4 }, tok.Encode("ab"));     // a+b→ab
        Assert.Equal(new[] { 0, 3 }, tok.Encode("ad"));  // 'a','d' — no (a,d) merge
        Assert.Equal(new[] { 3, 4 }, tok.Encode("dab")); // 'd', then a+b→ab
    }

    [Fact]
    public void Encode_SplitsAddedTokensAsSingleIds()
    {
        var tok = HfTokenizer.FromTokenizerJson(TokenizerJson, eosToken: "<eos>");
        // "<eos>" must be emitted as its single id (6), not shredded through byte-level BPE.
        Assert.Equal(new[] { 6 }, tok.Encode("<eos>"));
        Assert.Equal(new[] { 5, 6, 0 }, tok.Encode("abc<eos>a")); // abc→5, <eos>→6, a→0
    }

    [Fact]
    public void RoundTripsThroughDecode()
    {
        var tok = HfTokenizer.FromTokenizerJson(TokenizerJson, eosToken: "<eos>");
        foreach (var text in new[] { "abc", "ab", "ad", "abcabc", "dcba" })
            Assert.Equal(text, tok.Decode(tok.Encode(text)));
    }

    [Fact]
    public void VocabSizeSpansSpecialTokens_AndEosResolves()
    {
        var tok = HfTokenizer.FromTokenizerJson(TokenizerJson, eosToken: "<eos>");
        Assert.Equal(7, tok.VocabSize); // ids 0..6 (5 vocab + ab/abc + the eos at 6)
        Assert.Equal(6, tok.EosId);
    }

    [Fact]
    public void DecodeSkipsSpecialTokens()
    {
        var tok = HfTokenizer.FromTokenizerJson(TokenizerJson, eosToken: "<eos>");
        Assert.Equal("abc", tok.Decode([5, 6])); // 6 = <eos>, skipped
    }

    [Fact]
    public void AcceptsArrayStyleMerges()
    {
        // Newer tokenizer.json stores merges as ["a","b"] pairs rather than "a b" strings.
        const string json =
            """
            { "model": { "type": "BPE", "vocab": { "a": 0, "b": 1, "ab": 2 }, "merges": [["a","b"]] } }
            """;
        var tok = HfTokenizer.FromTokenizerJson(json);
        Assert.Equal(new[] { 2 }, tok.Encode("ab"));
    }
}
