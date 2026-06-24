using System.Buffers.Binary;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using System.Text.Json;
using ProjectAI.Core;

namespace ProjectAI.Formats;

/// <summary>A mapping from parameter name to tensor, as loaded from a checkpoint file.</summary>
public sealed class StateDict
{
    private readonly Dictionary<string, Tensor> _tensors = new();

    public Tensor this[string name]
    {
        get => _tensors[name];
        set => _tensors[name] = value;
    }

    public IReadOnlyDictionary<string, Tensor> Tensors => _tensors;
    public int Count => _tensors.Count;

    public bool TryGet(string name, [MaybeNullWhen(false)] out Tensor tensor) =>
        _tensors.TryGetValue(name, out tensor);
}

/// <summary>Reads model weights from a file into device tensors via the supplied backend.</summary>
public interface IWeightLoader
{
    StateDict Load(string path, IComputeBackend backend);
}

/// <summary>HuggingFace .safetensors: JSON header + raw little-endian tensor bytes (zero-copy where possible).</summary>
public interface ISafetensorsLoader : IWeightLoader;

/// <summary>GGUF (llama.cpp) checkpoints, including quantized weight blocks.</summary>
public interface IGgufLoader : IWeightLoader;

/// <summary>
/// Reads a HuggingFace <c>.safetensors</c> file (ticket S1-4): an 8-byte little-endian header length, a UTF-8
/// JSON header describing each tensor's dtype/shape/byte-offsets, then a tightly-packed data buffer. Every dtype
/// is materialized as F32 (the only storage this runtime has), so F32 is bit-exact and F16/BF16/F64/integers are
/// cast. The parser is defensive: it validates header/offset bounds, the data buffer must be fully tiled with no
/// gaps or overlaps, and tensors are read by seeking (one at a time) rather than loading the whole file.
/// </summary>
public sealed class SafetensorsLoader : ISafetensorsLoader
{
    // Bound the JSON header so a corrupt 8-byte length can't trigger a huge allocation before we've validated it.
    private const long MaxHeaderBytes = 100L * 1024 * 1024;

    private readonly record struct Entry(string Name, string Dtype, int[] Dims, int Elements, int ElementSize, long Begin, long End);

    public StateDict Load(string path, IComputeBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        // The payload (and the bulk MemoryMarshal reinterprets below) are little-endian; fail loudly rather than
        // returning byte-swapped weights on a big-endian host. Every platform .NET ships on in practice is LE.
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("safetensors loading requires a little-endian host.");
        using var stream = File.OpenRead(path);
        long fileLength = stream.Length;
        if (fileLength < 8) throw new InvalidDataException($"'{path}' is too small to be a safetensors file.");

        Span<byte> lengthBytes = stackalloc byte[8];
        ReadExactly(stream, lengthBytes, path, "header length");
        ulong headerLength = BinaryPrimitives.ReadUInt64LittleEndian(lengthBytes);
        if (headerLength == 0 || headerLength > (ulong)(fileLength - 8) || headerLength > MaxHeaderBytes)
            throw new InvalidDataException($"'{path}' has an invalid safetensors header length ({headerLength}).");

        var headerBytes = new byte[headerLength];
        ReadExactly(stream, headerBytes, path, "JSON header");

        long dataStart = 8L + (long)headerLength;
        long dataLength = fileLength - dataStart;

        JsonDocument document;
        try { document = JsonDocument.Parse(headerBytes); }
        catch (JsonException ex) { throw new InvalidDataException($"'{path}' has a malformed safetensors header: {ex.Message}"); }

        using (document)
        {
            if (document.RootElement.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException($"'{path}' header is not a JSON object.");

            var entries = ParseEntries(document.RootElement, dataLength, path);
            VerifyContiguous(entries, dataLength, path);

            var state = new StateDict();
            foreach (var entry in entries)
                state[entry.Name] = ReadTensor(entry, stream, dataStart, backend);
            return state;
        }
    }

    private static List<Entry> ParseEntries(JsonElement root, long dataLength, string path)
    {
        var entries = new List<Entry>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var property in root.EnumerateObject())
        {
            if (property.Name == "__metadata__") continue; // reserved string→string map, not a tensor
            if (!seen.Add(property.Name)) throw new InvalidDataException($"'{path}' has a duplicate tensor '{property.Name}'.");
            entries.Add(ParseEntry(property.Name, property.Value, dataLength));
        }
        return entries;
    }

    private static Entry ParseEntry(string name, JsonElement value, long dataLength)
    {
        if (value.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException($"tensor '{name}' is not a JSON object.");

        if (!value.TryGetProperty("dtype", out var dtypeElement) || dtypeElement.ValueKind != JsonValueKind.String)
            throw new InvalidDataException($"tensor '{name}' is missing a string 'dtype'.");
        string dtype = dtypeElement.GetString()!;
        int elementSize = DtypeByteSize(dtype)
            ?? throw new InvalidDataException($"tensor '{name}' has unsupported dtype '{dtype}'.");

        if (!value.TryGetProperty("shape", out var shapeElement) || shapeElement.ValueKind != JsonValueKind.Array)
            throw new InvalidDataException($"tensor '{name}' is missing a 'shape' array.");
        var dims = new int[shapeElement.GetArrayLength()];
        long elements = 1;
        for (int i = 0; i < dims.Length; i++)
        {
            if (shapeElement[i].ValueKind != JsonValueKind.Number || !shapeElement[i].TryGetInt64(out long dim) || dim < 0 || dim > int.MaxValue)
                throw new InvalidDataException($"tensor '{name}' has an invalid shape dimension at index {i}.");
            dims[i] = (int)dim;
            elements *= dim;
            if (elements > int.MaxValue)
                throw new InvalidDataException($"tensor '{name}' has {elements} elements, exceeding the {int.MaxValue} array limit.");
        }

        if (!value.TryGetProperty("data_offsets", out var offsets) || offsets.ValueKind != JsonValueKind.Array || offsets.GetArrayLength() != 2)
            throw new InvalidDataException($"tensor '{name}' needs a 2-element 'data_offsets'.");
        // Guard ValueKind before TryGetInt64: on a non-Number element it throws InvalidOperationException rather
        // than returning false, which would escape this parser's InvalidDataException contract.
        if (offsets[0].ValueKind != JsonValueKind.Number || offsets[1].ValueKind != JsonValueKind.Number ||
            !offsets[0].TryGetInt64(out long begin) || !offsets[1].TryGetInt64(out long end))
            throw new InvalidDataException($"tensor '{name}' has non-integer data_offsets.");
        if (begin < 0 || end < begin || end > dataLength)
            throw new InvalidDataException($"tensor '{name}' data_offsets [{begin}, {end}] are out of bounds (data length {dataLength}).");
        if (end - begin != elements * elementSize)
            throw new InvalidDataException($"tensor '{name}': byte span {end - begin} != {elements} elements × {elementSize} ({elements * elementSize}).");

        return new Entry(name, dtype, dims, (int)elements, elementSize, begin, end);
    }

    // The spec requires the data buffer to be tiled by the tensors with no gaps or overlaps.
    private static void VerifyContiguous(List<Entry> entries, long dataLength, string path)
    {
        if (entries.Count == 0)
        {
            if (dataLength != 0) throw new InvalidDataException($"'{path}' has {dataLength} data bytes but declares no tensors.");
            return;
        }
        long cursor = 0;
        // Sort by start, then end, so a zero-byte (empty) tensor sharing a real tensor's start offset is visited
        // first and doesn't read as a gap.
        foreach (var entry in entries.OrderBy(e => e.Begin).ThenBy(e => e.End))
        {
            if (entry.Begin != cursor)
                throw new InvalidDataException($"'{path}' data buffer has a gap or overlap at byte {cursor} (tensor '{entry.Name}' starts at {entry.Begin}).");
            cursor = entry.End;
        }
        if (cursor != dataLength)
            throw new InvalidDataException($"'{path}' has {dataLength - cursor} trailing data bytes covered by no tensor.");
    }

    private static Tensor ReadTensor(Entry entry, FileStream stream, long dataStart, IComputeBackend backend)
    {
        var raw = new byte[entry.End - entry.Begin];
        stream.Seek(dataStart + entry.Begin, SeekOrigin.Begin);
        ReadExactly(stream, raw, entry.Name, "tensor payload");
        return backend.FromHost(ToFloats(entry.Dtype, raw, entry.Elements), new Shape(entry.Dims), DType.F32);
    }

    private static int? DtypeByteSize(string dtype) => dtype switch
    {
        "F64" or "I64" or "U64" => 8,
        "F32" or "I32" or "U32" => 4,
        "F16" or "BF16" or "I16" or "U16" => 2,
        "I8" or "U8" or "BOOL" => 1,
        _ => null, // sub-byte (F4/F6_*), FP8 variants, and C64 are not supported by this first-pass loader
    };

    // Materializes the little-endian payload as float32. F32 is a bulk bit-for-bit reinterpret; the rest convert
    // per element. BF16→f32 is the exact upper-16-bits widening; F16 widens via System.Half.
    private static float[] ToFloats(string dtype, byte[] raw, int count)
    {
        var result = new float[count];
        switch (dtype)
        {
            case "F32":
                MemoryMarshal.Cast<byte, float>(raw).CopyTo(result);
                break;
            case "F16":
            {
                var src = MemoryMarshal.Cast<byte, ushort>(raw);
                for (int i = 0; i < count; i++) result[i] = (float)BitConverter.UInt16BitsToHalf(src[i]);
                break;
            }
            case "BF16":
            {
                var src = MemoryMarshal.Cast<byte, ushort>(raw);
                for (int i = 0; i < count; i++) result[i] = BitConverter.UInt32BitsToSingle((uint)src[i] << 16);
                break;
            }
            case "F64":
            {
                var src = MemoryMarshal.Cast<byte, double>(raw);
                for (int i = 0; i < count; i++) result[i] = (float)src[i];
                break;
            }
            case "I64": { var s = MemoryMarshal.Cast<byte, long>(raw);   for (int i = 0; i < count; i++) result[i] = s[i]; break; }
            case "U64": { var s = MemoryMarshal.Cast<byte, ulong>(raw);  for (int i = 0; i < count; i++) result[i] = s[i]; break; }
            case "I32": { var s = MemoryMarshal.Cast<byte, int>(raw);    for (int i = 0; i < count; i++) result[i] = s[i]; break; }
            case "U32": { var s = MemoryMarshal.Cast<byte, uint>(raw);   for (int i = 0; i < count; i++) result[i] = s[i]; break; }
            case "I16": { var s = MemoryMarshal.Cast<byte, short>(raw);  for (int i = 0; i < count; i++) result[i] = s[i]; break; }
            case "U16": { var s = MemoryMarshal.Cast<byte, ushort>(raw); for (int i = 0; i < count; i++) result[i] = s[i]; break; }
            case "I8":  for (int i = 0; i < count; i++) result[i] = (sbyte)raw[i]; break;
            case "U8":  for (int i = 0; i < count; i++) result[i] = raw[i]; break;
            case "BOOL": for (int i = 0; i < count; i++) result[i] = raw[i] != 0 ? 1f : 0f; break;
            default: throw new InvalidDataException($"unsupported dtype '{dtype}'."); // unreachable — DtypeByteSize gated it
        }
        return result;
    }

    private static void ReadExactly(Stream stream, Span<byte> buffer, string name, string what)
    {
        int total = 0;
        while (total < buffer.Length)
        {
            int read = stream.Read(buffer[total..]);
            if (read == 0) throw new InvalidDataException($"'{name}' is truncated reading {what}.");
            total += read;
        }
    }
}

public sealed class GgufLoader : IGgufLoader
{
    public StateDict Load(string path, IComputeBackend backend) => throw new NotImplementedException("ticket S1-5.");
}
