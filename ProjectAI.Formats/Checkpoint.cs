using System.Runtime.InteropServices;
using System.Text;
using ProjectAI.Core;

namespace ProjectAI.Formats;

/// <summary>
/// ProjectAI's native training checkpoint: a flat little-endian binary file of named F32 tensors, a training
/// step counter, and a free-form metadata string (e.g. JSON model config + tokenizer). It is deliberately
/// self-contained (no dependency on the external safetensors/GGUF loaders, S1-4/S1-5). Layout:
/// <code>
///   magic "PAICKPT2" (8 bytes) | int32 step | int32 metaLen | metadata (UTF-8) | int32 tensorCount
///   per tensor: int32 nameLen | name (UTF-8) | int32 rank | rank × int32 dims | int64 byteLen | F32 LE payload
/// </code>
/// Floats are written in host byte order; ProjectAI targets little-endian hosts (x64/ARM64), matching the
/// safetensors convention.
/// </summary>
public static class Checkpoint
{
    private static ReadOnlySpan<byte> Magic => "PAICKPT2"u8;

    /// <summary>Writes <paramref name="tensors"/>, the <paramref name="step"/> counter, and <paramref name="metadata"/>.</summary>
    public static void Save(string path, int step, IEnumerable<(string Name, Tensor Tensor)> tensors, IComputeBackend backend, string metadata = "")
    {
        var items = tensors.ToList();
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        writer.Write(Magic);
        writer.Write(step);
        byte[] metaBytes = Encoding.UTF8.GetBytes(metadata);
        writer.Write(metaBytes.Length);
        writer.Write(metaBytes);
        writer.Write(items.Count);

        foreach (var (name, tensor) in items)
        {
            if (tensor.DType != DType.F32)
                throw new NotSupportedException($"checkpoint supports F32 only; tensor '{name}' is {tensor.DType}.");
            if (tensor.ElementCount > int.MaxValue)
                throw new NotSupportedException($"tensor '{name}' has {tensor.ElementCount} elements, exceeding this format's 2^31 limit.");

            byte[] nameBytes = Encoding.UTF8.GetBytes(name);
            writer.Write(nameBytes.Length);
            writer.Write(nameBytes);

            writer.Write(tensor.Shape.Rank);
            for (int d = 0; d < tensor.Shape.Rank; d++) writer.Write(tensor.Shape[d]);

            var host = new float[tensor.ElementCount];
            backend.ToHost(tensor, host);
            writer.Write((long)host.Length * sizeof(float));
            writer.Write(MemoryMarshal.AsBytes(host.AsSpan()));
        }
    }

    /// <summary>Reads a checkpoint, materializing each tensor onto <paramref name="backend"/>; returns the step, metadata, and tensors.</summary>
    public static (int Step, string Metadata, StateDict Tensors) Load(string path, IComputeBackend backend)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        Span<byte> magic = stackalloc byte[8];
        if (reader.Read(magic) != magic.Length || !magic.SequenceEqual(Magic))
            throw new InvalidDataException($"'{path}' is not a ProjectAI checkpoint (bad magic).");

        int step = reader.ReadInt32();
        int metaLen = reader.ReadInt32();
        if (metaLen < 0 || metaLen > 1 << 24) throw new InvalidDataException($"'{path}' has a bad metadata length ({metaLen}).");
        string metadata = Encoding.UTF8.GetString(ReadExactly(reader, metaLen, path, "metadata"));
        int count = reader.ReadInt32();
        if (count < 0) throw new InvalidDataException($"'{path}' has a negative tensor count ({count}).");
        var dict = new StateDict();

        for (int i = 0; i < count; i++)
        {
            int nameLen = reader.ReadInt32();
            if (nameLen < 0 || nameLen > 1 << 20) throw new InvalidDataException($"'{path}' tensor {i}: bad name length {nameLen}.");
            string name = Encoding.UTF8.GetString(ReadExactly(reader, nameLen, path, $"tensor {i} name"));

            int rank = reader.ReadInt32();
            if (rank < 0 || rank > 8) throw new InvalidDataException($"checkpoint tensor '{name}' has invalid rank {rank}.");
            var dims = new int[rank];
            for (int d = 0; d < rank; d++)
            {
                dims[d] = reader.ReadInt32();
                if (dims[d] < 0) throw new InvalidDataException($"checkpoint tensor '{name}' has a negative dimension {dims[d]}.");
            }

            // The payload length must be exactly the product of the dims (as F32) — reject truncation/corruption
            // up front with a clear message instead of an opaque ReadBytes/MemoryMarshal/CopyTo failure later.
            long expected = new Shape(dims).ElementCount * sizeof(float);
            long byteLen = reader.ReadInt64();
            if (byteLen != expected || byteLen > int.MaxValue)
                throw new InvalidDataException(
                    $"checkpoint tensor '{name}': payload length {byteLen} != dims product × 4 ({expected}).");

            byte[] bytes = ReadExactly(reader, (int)byteLen, path, $"tensor '{name}' payload");
            var floats = MemoryMarshal.Cast<byte, float>(bytes).ToArray();
            dict[name] = backend.FromHost(floats, new Shape(dims), DType.F32);
        }

        return (step, metadata, dict);
    }

    // Reads exactly count bytes or throws — BinaryReader.ReadBytes silently returns a short buffer at EOF, which
    // would otherwise let a file truncated inside the metadata/name/payload regions decode garbage.
    private static byte[] ReadExactly(BinaryReader reader, int count, string path, string what)
    {
        byte[] buffer = reader.ReadBytes(count);
        if (buffer.Length != count)
            throw new InvalidDataException($"'{path}' is truncated reading {what} (expected {count} bytes, got {buffer.Length}).");
        return buffer;
    }
}
