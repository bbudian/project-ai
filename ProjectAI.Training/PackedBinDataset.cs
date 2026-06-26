using System.IO.MemoryMappedFiles;

namespace ProjectAI.Training;

/// <summary>
/// An <see cref="IDataset"/> backed by a memory-mapped packed token-id binary written by <see cref="DatasetPacker"/>.
/// The corpus stays on OS-paged file backing instead of the managed heap, so it can exceed RAM and adds no gen2 GC
/// pressure; only the touched pages are resident. Each <see cref="GetSequence"/> materializes one block into a
/// managed <c>int[]</c>, so the mapped view is never aliased across a backend memory scope (the trainer frees each
/// micro-batch's graph deterministically — see <c>Backend.BeginScope</c>). Read-only and little-endian (the on-disk
/// format); <see cref="Open"/> fails loudly on a big-endian host rather than returning byte-swapped ids.
/// </summary>
public sealed class PackedBinDataset : IDataset, IDisposable
{
    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _view;
    private readonly int _blockLen;
    private readonly int _elemSize;
    private readonly bool _u16;

    /// <summary>The manifest that produced this dataset (tokenizer, dtype, provenance, block counts).</summary>
    public DatasetManifest Manifest { get; }

    /// <summary>Model context length; each emitted sequence has <c>SequenceLength + 1</c> ids.</summary>
    public int SequenceLength => Manifest.SequenceLength;

    public int Count { get; }

    private PackedBinDataset(DatasetManifest manifest, MemoryMappedFile mmf, MemoryMappedViewAccessor view)
    {
        Manifest = manifest;
        _mmf = mmf;
        _view = view;
        _blockLen = manifest.SequenceLength + 1;
        _u16 = manifest.Dtype == "u16";
        _elemSize = _u16 ? 2 : 4;
        Count = checked((int)manifest.BlockCount);
    }

    /// <summary>
    /// Opens a packed dataset from a directory containing a <see cref="DatasetManifest.FileName"/>, or from the
    /// manifest file itself. Validates the shard is large enough for the declared blocks, then memory-maps it.
    /// </summary>
    public static PackedBinDataset Open(string path)
    {
        // The shard is little-endian (BinaryWriter); reading it via the view is host-endian, so a big-endian host
        // would byte-swap every id. Fail rather than train on garbage (mirrors SafetensorsLoader's guard).
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("packed datasets are little-endian; this host is big-endian.");

        string manifestPath = DatasetManifest.TryResolvePath(path)
            ?? throw new FileNotFoundException($"no '{DatasetManifest.FileName}' found at '{path}'.");
        var manifest = DatasetManifest.FromJson(File.ReadAllText(manifestPath));

        if (manifest.Dtype is not ("u16" or "u32"))
            throw new InvalidDataException($"unsupported dataset dtype '{manifest.Dtype}'.");
        if (manifest.SequenceLength < 1)
            throw new InvalidDataException($"manifest SequenceLength must be >= 1; got {manifest.SequenceLength}.");
        if (manifest.BlockCount < 1)
            throw new InvalidDataException($"manifest BlockCount must be >= 1; got {manifest.BlockCount}.");
        if (manifest.Shards.Length != 1)
            throw new NotSupportedException(
                $"this build reads single-shard datasets; manifest lists {manifest.Shards.Length} shards (sharding is a later phase).");

        string dir = Path.GetDirectoryName(manifestPath) ?? ".";
        string binPath = Path.Combine(dir, manifest.Shards[0]);
        if (!File.Exists(binPath))
            throw new FileNotFoundException($"dataset shard '{binPath}' (from manifest) not found.");

        int blockLen = manifest.SequenceLength + 1;
        int elemSize = manifest.Dtype == "u16" ? 2 : 4;
        long needed = manifest.BlockCount * blockLen * elemSize;
        long fileLen = new FileInfo(binPath).Length;
        if (fileLen < needed)
            throw new InvalidDataException(
                $"dataset shard '{binPath}' is {fileLen} bytes but the manifest needs {needed} for {manifest.BlockCount} blocks of {blockLen} {manifest.Dtype} ids.");

        var mmf = MemoryMappedFile.CreateFromFile(binPath, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
        try
        {
            var view = mmf.CreateViewAccessor(0, needed, MemoryMappedFileAccess.Read);
            return new PackedBinDataset(manifest, mmf, view);
        }
        catch
        {
            mmf.Dispose();
            throw;
        }
    }

    public ReadOnlyMemory<int> GetSequence(int index)
    {
        if ((uint)index >= (uint)Count)
            throw new ArgumentOutOfRangeException(nameof(index), index, $"valid range is [0, {Count}).");

        long byteOffset = (long)index * _blockLen * _elemSize;
        var block = new int[_blockLen];
        if (_u16)
        {
            var raw = new ushort[_blockLen];
            _view.ReadArray(byteOffset, raw, 0, _blockLen);
            for (int i = 0; i < _blockLen; i++) block[i] = raw[i];
        }
        else
        {
            var raw = new uint[_blockLen];
            _view.ReadArray(byteOffset, raw, 0, _blockLen);
            for (int i = 0; i < _blockLen; i++) block[i] = checked((int)raw[i]);
        }
        return block;
    }

    public void Dispose()
    {
        _view.Dispose();
        _mmf.Dispose();
    }
}
