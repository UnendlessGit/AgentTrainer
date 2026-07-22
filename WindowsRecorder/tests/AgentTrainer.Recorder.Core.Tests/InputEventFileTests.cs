using System.Buffers.Binary;
using AgentTrainer.Recorder.Core;

namespace AgentTrainer.Recorder.Core.Tests;

public sealed class InputEventFileTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"AgentTrainerTests-{Guid.NewGuid():N}");

    public InputEventFileTests() => Directory.CreateDirectory(_root);

    [Fact]
    public void WriterMatchesTheNativeBinaryContractExactly()
    {
        var path = Path.Combine(_root, "events.atrevents");
        using (var writer = new InputEventWriter(path))
        {
            writer.Append(new InputSample(0x0102_0304_0506_0708, InputEventKind.Key,
                X: 1.25, Y: -2.5, DeltaX: 3.75, DeltaY: -4.5, Button: 2,
                ScrollX: 5.5, ScrollY: -6.25, KeyCode: 0x1234,
                Modifiers: QuartzModifierFlags.Command | QuartzModifierFlags.Shift, IsDown: true));
        }

        var bytes = File.ReadAllBytes(path);
        Assert.Equal(InputEventWriter.HeaderSize + InputEventWriter.RecordSize, bytes.Length);
        Assert.Equal("ATREVT01"u8.ToArray(), bytes[..8]);
        Assert.Equal(1u, BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(8, 4)));
        Assert.Equal(0x0102_0304_0506_0708ul, BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(12, 8)));
        Assert.Equal((byte)InputEventKind.Key, bytes[20]);
        Assert.Equal(1, bytes[21]);
        Assert.Equal(2, bytes[22]);
        Assert.Equal(0x1234, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(24, 2)));
        Assert.Equal(1.25, ReadDouble(bytes, 36));
        Assert.Equal(-6.25, ReadDouble(bytes, 76));

        var decoded = Assert.Single(InputEventReader.ReadAll(path));
        Assert.Equal((ushort)0x1234, decoded.KeyCode);
        Assert.True(decoded.IsDown);
        Assert.Equal(-4.5, decoded.DeltaY);
    }

    [Fact]
    public void ReaderRejectsTruncationUnknownKindsAndDecreasingTime()
    {
        var truncated = Path.Combine(_root, "truncated.atrevents");
        File.WriteAllBytes(truncated, [.. "ATREVT01"u8, 1, 0, 0, 0, 1]);
        Assert.Throws<InvalidDataException>(() => InputEventReader.ReadAll(truncated));

        var unknown = Path.Combine(_root, "unknown.atrevents");
        using (var writer = new InputEventWriter(unknown)) writer.Append(new InputSample(1, InputEventKind.Key));
        var unknownBytes = File.ReadAllBytes(unknown);
        unknownBytes[20] = 255;
        File.WriteAllBytes(unknown, unknownBytes);
        Assert.Throws<InvalidDataException>(() => InputEventReader.ReadAll(unknown));

        var reversed = Path.Combine(_root, "reversed.atrevents");
        using (var writer = new InputEventWriter(reversed))
        {
            writer.Append(new InputSample(20, InputEventKind.MouseMove));
            writer.Append(new InputSample(20, InputEventKind.MouseMove));
        }
        var reversedBytes = File.ReadAllBytes(reversed);
        BinaryPrimitives.WriteUInt64LittleEndian(reversedBytes.AsSpan(84, 8), 19);
        File.WriteAllBytes(reversed, reversedBytes);
        Assert.Throws<InvalidDataException>(() => InputEventReader.ReadAll(reversed));
    }

    [Fact]
    public void WriterRejectsNonFiniteValuesAndDecreasingTimestamps()
    {
        var path = Path.Combine(_root, "invalid.atrevents");
        using var writer = new InputEventWriter(path);
        writer.Append(new InputSample(2, InputEventKind.MouseMove));
        Assert.Throws<InvalidDataException>(() => writer.Append(new InputSample(1, InputEventKind.MouseMove)));
        Assert.Throws<InvalidDataException>(() => writer.Append(new InputSample(3, InputEventKind.Scroll, ScrollY: double.NaN)));
    }

    private static double ReadDouble(byte[] bytes, int offset) =>
        BitConverter.Int64BitsToDouble(BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(offset, 8)));

    public void Dispose() => Directory.Delete(_root, recursive: true);
}
