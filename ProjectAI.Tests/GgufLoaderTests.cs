using System.Runtime.InteropServices;
using System.Text;
using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using ProjectAI.Formats;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>
/// Tests the GGUF loader (S1-5) against synthetic GGUF v3 files built in-test (no real model needed): the format
/// parse, F32/F16 materialization, the innermost-first → row-major dim reversal, metadata typing, and the clear
/// rejections (quantized blocks, bad magic, unsupported version).
/// </summary>
public class GgufLoaderTests
{
    [Fact]
    public void LoadsFloatTensors_ReversesDims_AndParsesMetadata()
    {
        // GGUF stores dims innermost-first, so ne=[3,2] is our row-major [2,3]; data is the [2,3] row-major floats.
        var aData = F32Bytes([1, 2, 3, 4, 5, 6]);
        var bData = F16Bytes([0.5f, -0.25f, 1.0f]);
        byte[] gguf = BuildGguf(
            version: 3,
            metadata:
            [
                ("general.architecture", VtString, "llama"),
                ("llama.block_count", VtUint32, 2u),
                ("test.floats", VtArray, (VtFloat32, new object[] { 1.5f, 2.5f })),
            ],
            tensors:
            [
                ("a", GgmlF32, [3, 2], aData),
                ("b", GgmlF16, [3], bData),
            ]);

        string path = WriteTemp(gguf);
        try
        {
            using var be = new CpuComputeBackend();
            var file = new GgufLoader().LoadFile(path, be);

            Assert.Equal(3u, file.Version);
            Assert.Equal("llama", file.Metadata["general.architecture"]);
            Assert.Equal(2, Convert.ToInt32(file.Metadata["llama.block_count"]));
            var floats = (object[])file.Metadata["test.floats"];
            Assert.Equal([1.5f, 2.5f], floats.Select(Convert.ToSingle));

            var a = file.Tensors["a"];
            Assert.Equal([2, 3], Dims(a));                          // ne=[3,2] → [2,3]
            Assert.Equal([1f, 2, 3, 4, 5, 6], Host(be, a));

            var b = file.Tensors["b"];
            Assert.Equal([3], Dims(b));
            Assert.Equal([0.5f, -0.25f, 1.0f], Host(be, b));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void RejectsQuantizedTensorType_PointingToS3_3()
    {
        const uint q8_0 = 8;
        byte[] gguf = BuildGguf(3, [], [("w", q8_0, [4], new byte[34])]);
        string path = WriteTemp(gguf);
        try
        {
            using var be = new CpuComputeBackend();
            var ex = Assert.Throws<NotSupportedException>(() => new GgufLoader().Load(path, be));
            Assert.Contains("Q8_0", ex.Message);
            Assert.Contains("S3-3", ex.Message);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void RejectsBadMagic()
    {
        byte[] gguf = BuildGguf(3, [], [("a", GgmlF32, [1], F32Bytes([1]))]);
        gguf[0] = (byte)'X'; // corrupt the magic
        string path = WriteTemp(gguf);
        try
        {
            using var be = new CpuComputeBackend();
            Assert.Throws<InvalidDataException>(() => new GgufLoader().Load(path, be));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void RejectsUnsupportedVersion()
    {
        byte[] gguf = BuildGguf(1, [], [("a", GgmlF32, [1], F32Bytes([1]))]); // v1 used 32-bit counts
        string path = WriteTemp(gguf);
        try
        {
            using var be = new CpuComputeBackend();
            Assert.Throws<InvalidDataException>(() => new GgufLoader().Load(path, be));
        }
        finally { File.Delete(path); }
    }

    // --- GGUF value-type tags + ggml tensor-type tags used by these tests ---
    private const byte VtUint32 = 4, VtFloat32 = 6, VtString = 8, VtArray = 9;
    private const uint GgmlF32 = 0, GgmlF16 = 1;

    // --- a minimal GGUF v2/v3 writer (mirrors the on-disk layout the loader parses) ---
    private static byte[] BuildGguf(
        uint version,
        (string Key, byte ValueType, object Value)[] metadata,
        (string Name, uint GgmlType, int[] Dims, byte[] Data)[] tensors,
        int alignment = 32)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);

        w.Write(0x46554747u);                 // "GGUF"
        w.Write(version);
        w.Write((ulong)tensors.Length);
        w.Write((ulong)metadata.Length);

        foreach (var (key, vt, value) in metadata)
        {
            WriteString(w, key);
            w.Write((uint)vt);
            WriteValue(w, vt, value);
        }

        // Lay tensor data out contiguously, each tensor's offset aligned (as ggml does).
        var offsets = new ulong[tensors.Length];
        ulong cursor = 0;
        for (int i = 0; i < tensors.Length; i++)
        {
            offsets[i] = cursor;
            cursor = (ulong)AlignUp((long)cursor + tensors[i].Data.Length, alignment);
        }

        for (int i = 0; i < tensors.Length; i++)
        {
            WriteString(w, tensors[i].Name);
            w.Write((uint)tensors[i].Dims.Length);
            foreach (int ne in tensors[i].Dims) w.Write((ulong)ne);
            w.Write(tensors[i].GgmlType);
            w.Write(offsets[i]);
        }

        long dataStart = AlignUp(ms.Position, alignment);
        Pad(w, ms, dataStart);
        for (int i = 0; i < tensors.Length; i++)
        {
            Pad(w, ms, dataStart + (long)offsets[i]);
            w.Write(tensors[i].Data);
        }

        w.Flush();
        return ms.ToArray();
    }

    private static void WriteValue(BinaryWriter w, byte valueType, object value)
    {
        switch (valueType)
        {
            case VtUint32: w.Write(Convert.ToUInt32(value)); break;
            case VtFloat32: w.Write(Convert.ToSingle(value)); break;
            case VtString: WriteString(w, (string)value); break;
            case VtArray:
                var (elemType, items) = ((byte ElemType, object[] Items))value;
                w.Write((uint)elemType);
                w.Write((ulong)items.Length);
                foreach (var item in items) WriteValue(w, elemType, item);
                break;
            default: throw new NotSupportedException($"test writer can't emit value type {valueType}.");
        }
    }

    private static void WriteString(BinaryWriter w, string s)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }

    private static void Pad(BinaryWriter w, MemoryStream ms, long target)
    {
        while (ms.Position < target) w.Write((byte)0);
    }

    private static long AlignUp(long position, int alignment) => (position + alignment - 1) / alignment * alignment;

    private static byte[] F32Bytes(float[] values) => MemoryMarshal.AsBytes<float>(values).ToArray();

    private static byte[] F16Bytes(float[] values)
    {
        var bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
            BinaryPrimitivesWriteHalf(bytes.AsSpan(i * 2), (Half)values[i]);
        return bytes;
    }

    private static void BinaryPrimitivesWriteHalf(Span<byte> dst, Half value)
    {
        ushort bits = BitConverter.HalfToUInt16Bits(value);
        dst[0] = (byte)bits;
        dst[1] = (byte)(bits >> 8);
    }

    private static string WriteTemp(byte[] bytes)
    {
        string path = Path.Combine(Path.GetTempPath(), $"projectai_{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, bytes);
        return path;
    }

    private static int[] Dims(Tensor t)
    {
        var dims = new int[t.Shape.Rank];
        for (int i = 0; i < dims.Length; i++) dims[i] = t.Shape[i];
        return dims;
    }

    private static float[] Host(IComputeBackend be, Tensor t)
    {
        var buf = new float[t.ElementCount];
        be.ToHost(t, buf);
        return buf;
    }
}
