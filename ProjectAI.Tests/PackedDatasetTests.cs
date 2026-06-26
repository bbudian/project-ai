using ProjectAI.Tokenizers;
using ProjectAI.Training;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>
/// Tests the packed-binary dataset path (P0 of the HuggingFace data integration): pack a corpus to a token-id
/// <c>.bin</c> + manifest, mmap it back via <see cref="PackedBinDataset"/>, and assert it reproduces exactly what
/// the in-memory <see cref="TextDataset"/> produces — same block math, same ids — plus manifest round-trip and the
/// defensive guards.
/// </summary>
public class PackedDatasetTests
{
    private const string Corpus =
        "the quick brown fox jumps over the lazy dog. " +
        "a stitch in time saves nine. the early bird catches the worm. " +
        "all that glitters is not gold. practice makes perfect every single day.";

    private static string NewTempDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), "projectai-ds-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    [Fact]
    public void Pack_ThenOpen_MatchesTextDatasetBlocks()
    {
        var tok = new BpeTokenizer([]);
        const int seqLen = 16;
        string dir = NewTempDir();
        try
        {
            // appendEos:false + a single document ⇒ the packed stream equals tokenizer.Encode(Corpus), so blocks
            // must match TextDataset exactly (same non-overlapping seqLen+1 chunks, trailing remainder dropped).
            var manifest = DatasetPacker.Pack([Corpus], tok, seqLen, dir, appendEosBetweenDocuments: false);
            var expected = new TextDataset(Corpus, tok, seqLen);

            using var ds = PackedBinDataset.Open(dir);

            Assert.Equal(expected.Count, ds.Count);
            Assert.Equal(expected.Count, (int)manifest.BlockCount);
            Assert.Equal("u16", manifest.Dtype);          // byte-level vocab 259 fits u16
            Assert.Equal(tok.VocabSize, manifest.VocabSize);
            Assert.Equal("bpe", manifest.TokenizerKind);
            Assert.Equal(seqLen, ds.SequenceLength);

            for (int b = 0; b < expected.Count; b++)
            {
                var exp = expected.GetSequence(b).Span;
                var got = ds.GetSequence(b).Span;
                Assert.Equal(seqLen + 1, got.Length);
                for (int i = 0; i < exp.Length; i++) Assert.Equal(exp[i], got[i]);
            }
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Pack_WritesShardAndManifestWithTokenizerHash()
    {
        var tok = new BpeTokenizer([]);
        string dir = NewTempDir();
        try
        {
            var manifest = DatasetPacker.Pack([Corpus], tok, 16, dir, appendEosBetweenDocuments: false);

            Assert.StartsWith("sha256:", manifest.TokenizerHash);
            Assert.Single(manifest.Shards);
            Assert.Equal(DatasetPacker.DefaultBinName, manifest.Shards[0]);
            Assert.True(File.Exists(Path.Combine(dir, DatasetManifest.FileName)));
            Assert.True(File.Exists(Path.Combine(dir, DatasetPacker.DefaultBinName)));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Manifest_RoundTripsThroughJson()
    {
        var tok = new BpeTokenizer([]);
        string dir = NewTempDir();
        try
        {
            var manifest = DatasetPacker.Pack([Corpus], tok, 16, dir, appendEosBetweenDocuments: false);
            var back = DatasetManifest.FromJson(manifest.ToJson());

            Assert.Equal(manifest.SequenceLength, back.SequenceLength);
            Assert.Equal(manifest.BlockCount, back.BlockCount);
            Assert.Equal(manifest.Dtype, back.Dtype);
            Assert.Equal(manifest.VocabSize, back.VocabSize);
            Assert.Equal(manifest.TokenizerHash, back.TokenizerHash);
            Assert.Equal(manifest.Shards[0], back.Shards[0]);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void AppendEos_AddsBoundaryToken()
    {
        var tok = new BpeTokenizer([]);
        string dir = NewTempDir();
        try
        {
            // With one document, appendEos adds exactly one extra token to the stream.
            long withoutEos = DatasetPacker.Pack([Corpus], tok, 16, NewTempDirInto(dir, "a"), appendEosBetweenDocuments: false).TotalTokens;
            long withEos = DatasetPacker.Pack([Corpus], tok, 16, NewTempDirInto(dir, "b"), appendEosBetweenDocuments: true).TotalTokens;
            Assert.Equal(withoutEos + 1, withEos);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Pack_ThrowsWhenCorpusTooShortForOneBlock()
    {
        var tok = new BpeTokenizer([]);
        string dir = NewTempDir();
        try
        {
            Assert.Throws<InvalidOperationException>(() =>
                DatasetPacker.Pack(["hi"], tok, sequenceLength: 64, dir, appendEosBetweenDocuments: false));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Open_RejectsMissingManifest()
    {
        string dir = NewTempDir();
        try { Assert.ThrowsAny<Exception>(() => PackedBinDataset.Open(dir)); }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Open_RejectsTruncatedShard()
    {
        var tok = new BpeTokenizer([]);
        string dir = NewTempDir();
        try
        {
            DatasetPacker.Pack([Corpus], tok, 16, dir, appendEosBetweenDocuments: false);
            string bin = Path.Combine(dir, DatasetPacker.DefaultBinName);
            using (var fs = new FileStream(bin, FileMode.Open, FileAccess.Write)) fs.SetLength(8); // can't hold the blocks
            Assert.Throws<InvalidDataException>(() => PackedBinDataset.Open(dir));
        }
        finally { Directory.Delete(dir, true); }
    }

    private static string NewTempDirInto(string parent, string name)
    {
        string dir = Path.Combine(parent, name);
        Directory.CreateDirectory(dir);
        return dir;
    }
}
