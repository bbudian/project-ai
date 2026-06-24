using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;
using ProjectAI.Backends.Cpu;
using ProjectAI.Core;
using ProjectAI.Formats;
using Xunit;

namespace ProjectAI.Tests;

/// <summary>Tests for the S1-4 safetensors loader: F32 bit-equality, F16/BF16 casts, and malformed-file rejection.</summary>
public class SafetensorsTests
{
    private static float[] Host(CpuComputeBackend be, Tensor t)
    {
        var buf = new float[t.ElementCount];
        be.ToHost(t, buf);
        return buf;
    }

    [Fact]
    public void Loads_F32_BitExact()
    {
        using var be = new CpuComputeBackend();
        var dir = NewTempDir();
        try
        {
            float[] values = [1.5f, -2.25f, 0f, 3.14159f, float.MaxValue, -1e-7f];
            string path = WriteContiguous(dir, ("w", "F32", [2, 3], Bytes(values)));

            var sd = new SafetensorsLoader().Load(path, be);
            Assert.Equal(1, sd.Count);
            Assert.Equal(new Shape(2, 3), sd["w"].Shape);
            float[] got = Host(be, sd["w"]);
            for (int i = 0; i < values.Length; i++) Assert.Equal(values[i], got[i]); // bit-exact
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void Loads_F16_Cast()
    {
        using var be = new CpuComputeBackend();
        var dir = NewTempDir();
        try
        {
            float[] values = [1.5f, -2.0f, 0.5f, 0.1f]; // last one isn't exactly representable → tests the rounding
            var bytes = new byte[values.Length * 2];
            for (int i = 0; i < values.Length; i++)
                BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i * 2), BitConverter.HalfToUInt16Bits((Half)values[i]));

            var sd = new SafetensorsLoader().Load(WriteContiguous(dir, ("w", "F16", [4], bytes)), be);
            float[] got = Host(be, sd["w"]);
            for (int i = 0; i < values.Length; i++) Assert.Equal((float)(Half)values[i], got[i]);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void Loads_BF16_Cast()
    {
        using var be = new CpuComputeBackend();
        var dir = NewTempDir();
        try
        {
            // All exactly representable in bfloat16 (low 16 mantissa bits zero), so the widening is exact.
            float[] values = [1.5f, -2.0f, 0.5f, 256f];
            var bytes = new byte[values.Length * 2];
            for (int i = 0; i < values.Length; i++)
                BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i * 2), (ushort)(BitConverter.SingleToUInt32Bits(values[i]) >> 16));

            var sd = new SafetensorsLoader().Load(WriteContiguous(dir, ("w", "BF16", [4], bytes)), be);
            float[] got = Host(be, sd["w"]);
            for (int i = 0; i < values.Length; i++) Assert.Equal(values[i], got[i]); // exact
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void Loads_MultipleTensors_AndSkipsMetadata()
    {
        using var be = new CpuComputeBackend();
        var dir = NewTempDir();
        try
        {
            float[] a = [1f, 2f], b = [3f, 4f, 5f];
            string header =
                "{\"__metadata__\":{\"format\":\"pt\"}," +
                "\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}," +
                "\"b\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[8,20]}}";
            string path = WriteRaw(dir, header, [Bytes(a), Bytes(b)]);

            var sd = new SafetensorsLoader().Load(path, be);
            Assert.Equal(2, sd.Count); // __metadata__ is not a tensor
            Assert.Equal(a, Host(be, sd["a"]));
            Assert.Equal(b, Host(be, sd["b"]));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Theory]
    [InlineData("tiny")]          // file shorter than the 8-byte length prefix
    [InlineData("header-overrun")] // declared header length exceeds the file
    [InlineData("bad-json")]
    [InlineData("bad-dtype")]
    [InlineData("size-mismatch")]
    [InlineData("gap")]
    [InlineData("duplicate")]
    [InlineData("string-offsets")]
    public void Rejects_MalformedFile(string kind)
    {
        using var be = new CpuComputeBackend();
        var dir = NewTempDir();
        try
        {
            string path = kind switch
            {
                "tiny" => WriteBytes(dir, [1, 2, 3, 4]),
                "header-overrun" => WriteBytes(dir, HeaderOverrun()),
                "bad-json" => WriteRaw(dir, "{ not valid json", []),
                "bad-dtype" => WriteRaw(dir, "{\"w\":{\"dtype\":\"F8_E4M3\",\"shape\":[2],\"data_offsets\":[0,2]}}", [new byte[2]]),
                "size-mismatch" => WriteRaw(dir, "{\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}", [new byte[4]]),
                "gap" => WriteRaw(dir, "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[8,12]}}", [new byte[12]]),
                "duplicate" => WriteRaw(dir, "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}", [new byte[8]]),
                "string-offsets" => WriteRaw(dir, "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[\"0\",\"4\"]}}", [new byte[4]]),
                _ => throw new ArgumentOutOfRangeException(nameof(kind)),
            };
            Assert.Throws<InvalidDataException>(() => new SafetensorsLoader().Load(path, be));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    // --- helpers ---

    private static byte[] Bytes(float[] values) => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();

    private static string WriteContiguous(string dir, params (string Name, string Dtype, int[] Shape, byte[] Data)[] tensors)
    {
        var header = new StringBuilder("{");
        var chunks = new List<byte[]>();
        long offset = 0;
        for (int i = 0; i < tensors.Length; i++)
        {
            var (name, dtype, shape, data) = tensors[i];
            if (i > 0) header.Append(',');
            header.Append($"\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{string.Join(",", shape)}],\"data_offsets\":[{offset},{offset + data.Length}]}}");
            chunks.Add(data);
            offset += data.Length;
        }
        header.Append('}');
        return WriteRaw(dir, header.ToString(), chunks);
    }

    private static string WriteRaw(string dir, string headerJson, IEnumerable<byte[]> dataChunks)
    {
        var header = Encoding.UTF8.GetBytes(headerJson);
        using var ms = new MemoryStream();
        Span<byte> length = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(length, (ulong)header.Length);
        ms.Write(length);
        ms.Write(header);
        foreach (var chunk in dataChunks) ms.Write(chunk);
        return WriteBytes(dir, ms.ToArray());
    }

    private static byte[] HeaderOverrun()
    {
        var bytes = new byte[16];
        BinaryPrimitives.WriteUInt64LittleEndian(bytes, 9999); // header length far past the file
        return bytes;
    }

    private static string WriteBytes(string dir, byte[] bytes)
    {
        string path = Path.Combine(dir, "m.safetensors");
        File.WriteAllBytes(path, bytes);
        return path;
    }

    private static string NewTempDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), "paist-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }
}
