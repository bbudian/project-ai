namespace ProjectAI.Formats.Datasets;

/// <summary>
/// Reads a dataset file into a stream of text documents — one logical training document per element. The offline
/// packer tokenizes these into a packed-binary token stream; different source formats (raw text now, Parquet/JSONL
/// next) are just different <see cref="IDatasetReader"/> implementations behind this seam, so the packer never
/// learns a new format. Implementations should stream (yield) rather than materialize a whole corpus in memory.
/// </summary>
public interface IDatasetReader
{
    /// <summary>Yields the documents in <paramref name="path"/>, lazily where possible.</summary>
    IEnumerable<string> ReadDocuments(string path);
}

/// <summary>
/// Treats a whole text file as a single document — the simplest source, and the one that matches today's
/// <c>train --data &lt;file&gt;</c> behaviour (which tokenizes the file as one string). Used to prove the
/// pack → mmap → train path before the Parquet/JSONL readers (P1) land.
/// </summary>
public sealed class RawTextDatasetReader : IDatasetReader
{
    public IEnumerable<string> ReadDocuments(string path)
    {
        yield return File.ReadAllText(path);
    }
}
