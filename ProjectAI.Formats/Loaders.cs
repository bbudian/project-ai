using System.Buffers.Binary;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using System.Text;
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

    public StateDict Load(string path, IComputeBackend backend) => Load(path, backend, DType.F32);

    /// <summary>
    /// Loads with an explicit target precision: F32 (default, bit-exact) or BF16/F16 to keep half-precision weights
    /// small in memory (ticket S3-1). Each tensor is widened to an F32 host buffer transiently, then materialized
    /// at <paramref name="targetDType"/> on the backend — so peak host memory is one tensor, not the whole model.
    /// </summary>
    public StateDict Load(string path, IComputeBackend backend, DType targetDType)
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
                state[entry.Name] = ReadTensor(entry, stream, dataStart, backend, targetDType);
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

    private static Tensor ReadTensor(Entry entry, FileStream stream, long dataStart, IComputeBackend backend, DType targetDType)
    {
        var raw = new byte[entry.End - entry.Begin];
        stream.Seek(dataStart + entry.Begin, SeekOrigin.Begin);
        ReadExactly(stream, raw, entry.Name, "tensor payload");
        return backend.FromHost(ToFloats(entry.Dtype, raw, entry.Elements), new Shape(entry.Dims), targetDType);
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

/// <summary>The parsed contents of a GGUF file: format version, the metadata key/value table, and the tensors.</summary>
public sealed class GgufFile
{
    public required uint Version { get; init; }
    /// <summary>Metadata values boxed as their natural CLR type (byte/short/int/long + unsigned, float/double, bool, string, or <c>object[]</c> for arrays).</summary>
    public required IReadOnlyDictionary<string, object> Metadata { get; init; }
    public required StateDict Tensors { get; init; }
}

/// <summary>
/// Reads a GGUF (llama.cpp) checkpoint (ticket S1-5): a <c>GGUF</c> magic + version, a metadata key/value table
/// (typed scalars, strings, and arrays), then per-tensor descriptors (name, dims, ggml type, data offset) and an
/// aligned tensor-data section. Float tensors (F32/F16/BF16) are materialized to F32 — the only storage this
/// runtime has. GGUF stores dimensions innermost-first (<c>ne[0]</c> contiguous), the reverse of our row-major
/// shape, so dims are reversed on load (the byte layout already matches). Quantized block types (Q4_*, Q8_*,
/// Q*_K, IQ*) are rejected with a clear error — dequantization is ticket S3-3. The parser is bounds-checked like
/// <see cref="SafetensorsLoader"/>: counts/lengths are capped and every tensor span is validated against the file.
/// </summary>
public sealed class GgufLoader : IGgufLoader
{
    private const uint Magic = 0x46554747;          // "GGUF" little-endian
    private const long MaxStringBytes = 64L * 1024 * 1024;
    private const ulong MaxCount = 100_000_000;     // bound tensor/metadata/array counts against a corrupt header

    // ggml tensor types we can materialize; everything else is a quantized block format (deferred to S3-3).
    private const uint GgmlF32 = 0, GgmlF16 = 1, GgmlBF16 = 30;

    public StateDict Load(string path, IComputeBackend backend) => LoadFile(path, backend).Tensors;

    /// <summary>Loads tensors AND metadata (the metadata carries the architecture/config + tokenizer for a future GGUF convert path).</summary>
    public GgufFile LoadFile(string path, IComputeBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        // BinaryReader reads little-endian, and the bulk F32 reinterpret below is host-endian; fail loudly on a
        // big-endian host rather than returning byte-swapped weights. Every platform .NET ships on is LE.
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("GGUF loading requires a little-endian host.");

        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        long fileLength = stream.Length;
        if (fileLength < 24) throw new InvalidDataException($"'{path}' is too small to be a GGUF file.");

        if (reader.ReadUInt32() != Magic)
            throw new InvalidDataException($"'{path}' is not a GGUF file (bad magic).");
        uint version = reader.ReadUInt32();
        if (version is not (2 or 3))
            throw new InvalidDataException($"'{path}' is GGUF v{version}; this loader supports v2 and v3 (v1 used 32-bit counts).");

        ulong tensorCount = reader.ReadUInt64();
        ulong metaCount = reader.ReadUInt64();
        if (tensorCount > MaxCount || metaCount > MaxCount)
            throw new InvalidDataException($"'{path}' declares implausible counts (tensors={tensorCount}, metadata={metaCount}).");

        var metadata = new Dictionary<string, object>(StringComparer.Ordinal);
        for (ulong i = 0; i < metaCount; i++)
        {
            string key = ReadGgufString(reader, path);
            metadata[key] = ReadValue(reader, path);
        }

        int alignment = metadata.TryGetValue("general.alignment", out var av) ? ToInt32(av, 32) : 32;
        if (alignment <= 0 || (alignment & (alignment - 1)) != 0) alignment = 32; // ggml requires a positive power of two

        var infos = new List<TensorInfo>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        for (ulong i = 0; i < tensorCount; i++)
        {
            string name = ReadGgufString(reader, path);
            if (!seen.Add(name)) throw new InvalidDataException($"'{path}' has a duplicate tensor '{name}'.");
            uint nDims = reader.ReadUInt32();
            if (nDims is 0 or > 8) throw new InvalidDataException($"tensor '{name}' has unsupported rank {nDims}.");
            var dims = new int[nDims];
            long elements = 1;
            for (int d = 0; d < nDims; d++)
            {
                ulong ne = reader.ReadUInt64();
                if (ne > int.MaxValue) throw new InvalidDataException($"tensor '{name}' dimension {d} ({ne}) exceeds the array limit.");
                dims[d] = (int)ne;
                elements *= (long)ne;
                if (elements > int.MaxValue) throw new InvalidDataException($"tensor '{name}' has {elements} elements, exceeding the {int.MaxValue} array limit.");
            }
            uint ggmlType = reader.ReadUInt32();
            ulong offset = reader.ReadUInt64();
            infos.Add(new TensorInfo(name, dims, (int)elements, ggmlType, offset));
        }

        long dataStart = AlignUp(stream.Position, alignment);

        var state = new StateDict();
        foreach (var info in infos)
            state[info.Name] = ReadTensor(info, stream, dataStart, fileLength, backend);

        return new GgufFile { Version = version, Metadata = metadata, Tensors = state };
    }

    private readonly record struct TensorInfo(string Name, int[] Dims, int Elements, uint GgmlType, ulong Offset);

    private static Tensor ReadTensor(TensorInfo info, FileStream stream, long dataStart, long fileLength, IComputeBackend backend)
    {
        int elementSize = info.GgmlType switch
        {
            GgmlF32 => 4,
            GgmlF16 or GgmlBF16 => 2,
            _ => throw new NotSupportedException(
                $"tensor '{info.Name}' is ggml type {GgmlTypeName(info.GgmlType)} (quantized); dequantization is ticket S3-3."),
        };
        long byteLength = (long)info.Elements * elementSize;
        if (byteLength > int.MaxValue) throw new InvalidDataException($"tensor '{info.Name}' is too large to load ({byteLength} bytes).");

        // Validate the unsigned offset against the data section BEFORE any cast/addition, so a huge offset can't
        // overflow a long and slip past the check (which would surface as an opaque Seek error, not a clean one).
        long available = fileLength - dataStart;
        if (available < 0) throw new InvalidDataException($"'{info.Name}': the GGUF data section starts past end of file.");
        if (info.Offset > (ulong)available || byteLength > available - (long)info.Offset)
            throw new InvalidDataException($"tensor '{info.Name}' data [offset {info.Offset}, +{byteLength} bytes] is out of bounds (data section is {available} bytes).");
        long begin = dataStart + (long)info.Offset;

        var raw = new byte[byteLength];
        stream.Seek(begin, SeekOrigin.Begin);
        ReadExactly(stream, raw, info.Name);

        float[] floats = info.GgmlType switch
        {
            GgmlF32 => F32ToFloats(raw, info.Elements),
            GgmlF16 => F16ToFloats(raw, info.Elements),
            _ => BF16ToFloats(raw, info.Elements), // GgmlBF16 (gated above)
        };

        // GGUF stores dims innermost-first; reverse to our row-major shape (the byte layout already matches).
        var shape = new int[info.Dims.Length];
        for (int i = 0; i < shape.Length; i++) shape[i] = info.Dims[info.Dims.Length - 1 - i];
        return backend.FromHost(floats, new Shape(shape), DType.F32);
    }

    private static float[] F32ToFloats(byte[] raw, int count)
    {
        var result = new float[count];
        MemoryMarshal.Cast<byte, float>(raw).CopyTo(result);
        return result;
    }

    private static float[] F16ToFloats(byte[] raw, int count)
    {
        var src = MemoryMarshal.Cast<byte, ushort>(raw);
        var result = new float[count];
        for (int i = 0; i < count; i++) result[i] = (float)BitConverter.UInt16BitsToHalf(src[i]);
        return result;
    }

    private static float[] BF16ToFloats(byte[] raw, int count)
    {
        var src = MemoryMarshal.Cast<byte, ushort>(raw);
        var result = new float[count];
        for (int i = 0; i < count; i++) result[i] = BitConverter.UInt32BitsToSingle((uint)src[i] << 16);
        return result;
    }

    // GGUF string: a uint64 byte length followed by UTF-8 bytes.
    private static string ReadGgufString(BinaryReader reader, string path)
    {
        ulong length = reader.ReadUInt64();
        if (length > MaxStringBytes) throw new InvalidDataException($"'{path}' has an implausibly long string ({length} bytes).");
        byte[] bytes = reader.ReadBytes((int)length);
        if (bytes.Length != (int)length) throw new InvalidDataException($"'{path}' is truncated reading a string.");
        return Encoding.UTF8.GetString(bytes);
    }

    private static object ReadValue(BinaryReader reader, string path)
    {
        uint type = reader.ReadUInt32();
        if (type == 9) return ReadArray(reader, path);   // ARRAY
        return ReadScalar(reader, type, path);
    }

    private static object[] ReadArray(BinaryReader reader, string path)
    {
        uint elemType = reader.ReadUInt32();
        if (elemType == 9) throw new InvalidDataException($"'{path}' has a nested array (not allowed by GGUF).");
        ulong count = reader.ReadUInt64();
        if (count > MaxCount) throw new InvalidDataException($"'{path}' has an implausibly large array ({count} elements).");
        var array = new object[count];
        for (ulong i = 0; i < count; i++) array[i] = ReadScalar(reader, elemType, path);
        return array;
    }

    private static object ReadScalar(BinaryReader reader, uint type, string path) => type switch
    {
        0 => reader.ReadByte(),        // UINT8
        1 => reader.ReadSByte(),       // INT8
        2 => reader.ReadUInt16(),      // UINT16
        3 => reader.ReadInt16(),       // INT16
        4 => reader.ReadUInt32(),      // UINT32
        5 => reader.ReadInt32(),       // INT32
        6 => reader.ReadSingle(),      // FLOAT32
        7 => reader.ReadBoolean(),     // BOOL (1 byte)
        8 => ReadGgufString(reader, path), // STRING
        10 => reader.ReadUInt64(),     // UINT64
        11 => reader.ReadInt64(),      // INT64
        12 => reader.ReadDouble(),     // FLOAT64
        _ => throw new InvalidDataException($"'{path}' has an unknown metadata value type {type}."),
    };

    private static int ToInt32(object value, int fallback) => value switch
    {
        byte b => b, sbyte sb => sb, ushort us => us, short s => s,
        uint ui => (int)ui, int i => i, ulong ul => (int)ul, long l => (int)l,
        _ => fallback,
    };

    private static long AlignUp(long position, int alignment) => (position + alignment - 1) / alignment * alignment;

    private static string GgmlTypeName(uint type) => type switch
    {
        2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1", 8 => "Q8_0", 9 => "Q8_1",
        10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K", 14 => "Q6_K", 15 => "Q8_K",
        _ => $"type {type}",
    };

    private static void ReadExactly(Stream stream, Span<byte> buffer, string name)
    {
        int total = 0;
        while (total < buffer.Length)
        {
            int read = stream.Read(buffer[total..]);
            if (read == 0) throw new InvalidDataException($"tensor '{name}' is truncated.");
            total += read;
        }
    }
}
