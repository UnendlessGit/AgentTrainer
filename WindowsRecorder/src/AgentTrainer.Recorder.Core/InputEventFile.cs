using System.Buffers;
using System.Buffers.Binary;
using System.Text;

namespace AgentTrainer.Recorder.Core;

public sealed class InputEventWriter : IDisposable
{
    public static ReadOnlySpan<byte> Magic => "ATREVT01"u8;
    public const uint FormatVersion = 1;
    public const int HeaderSize = 12;
    public const int RecordSize = 72;
    private const int BufferCapacity = 256 * 1024;

    private readonly FileStream _stream;
    private readonly byte[] _buffer;
    private readonly object _gate = new();
    private int _bufferCount;
    private bool _closed;
    private ulong? _lastTimestamp;

    public InputEventWriter(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        _stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read, 4096,
            FileOptions.SequentialScan | FileOptions.WriteThrough);
        _buffer = ArrayPool<byte>.Shared.Rent(BufferCapacity + RecordSize);
        _stream.Write(Magic);
        Span<byte> version = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(version, FormatVersion);
        _stream.Write(version);
    }

    public int Count { get; private set; }

    public void Append(InputSample sample)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_closed, this);
            if (_lastTimestamp is { } previous && sample.TimestampNanos < previous)
                throw new InvalidDataException("Input timestamps must be nondecreasing.");
            if (!Enum.IsDefined(sample.Kind) || !AllFinite(sample))
                throw new InvalidDataException("Input sample contains an unsupported kind or non-finite value.");

            Span<byte> record = stackalloc byte[RecordSize];
            record.Clear();
            BinaryPrimitives.WriteUInt64LittleEndian(record, sample.TimestampNanos);
            record[8] = (byte)sample.Kind;
            record[9] = sample.IsDown ? (byte)1 : (byte)0;
            record[10] = sample.Button;
            BinaryPrimitives.WriteUInt16LittleEndian(record[12..], sample.KeyCode);
            BinaryPrimitives.WriteUInt64LittleEndian(record[16..], sample.Modifiers);
            WriteDouble(record[24..], sample.X);
            WriteDouble(record[32..], sample.Y);
            WriteDouble(record[40..], sample.DeltaX);
            WriteDouble(record[48..], sample.DeltaY);
            WriteDouble(record[56..], sample.ScrollX);
            WriteDouble(record[64..], sample.ScrollY);
            record.CopyTo(_buffer.AsSpan(_bufferCount));
            _bufferCount += RecordSize;
            Count++;
            _lastTimestamp = sample.TimestampNanos;
            if (_bufferCount >= BufferCapacity) FlushBuffer();
        }
    }

    public int Finish()
    {
        lock (_gate)
        {
            if (_closed) return Count;
            FlushBuffer();
            _stream.Flush(flushToDisk: true);
            _stream.Dispose();
            _closed = true;
            ArrayPool<byte>.Shared.Return(_buffer);
            return Count;
        }
    }

    public void Dispose() => Finish();

    private void FlushBuffer()
    {
        if (_bufferCount == 0) return;
        _stream.Write(_buffer, 0, _bufferCount);
        _bufferCount = 0;
    }

    private static bool AllFinite(InputSample value) =>
        double.IsFinite(value.X) && double.IsFinite(value.Y)
        && double.IsFinite(value.DeltaX) && double.IsFinite(value.DeltaY)
        && double.IsFinite(value.ScrollX) && double.IsFinite(value.ScrollY);

    private static void WriteDouble(Span<byte> destination, double value) =>
        BinaryPrimitives.WriteInt64LittleEndian(destination, BitConverter.DoubleToInt64Bits(value));
}

public sealed record InputEventSummary(
    int Count,
    int KeyEventCount,
    int MouseEventCount,
    SortedSet<ushort> UsedKeyCodes,
    IReadOnlyList<InputSample> Preview,
    MouseDiagnostics Mouse,
    InputSample? First,
    InputSample? Last);

public sealed record MouseDiagnostics
{
    public int MoveEventCount { get; init; }
    public int NonzeroDeltaCount { get; init; }
    public int AbsolutePositionChangeCount { get; init; }
    public int OutOfCaptureBoundsCount { get; init; }
    public double AccumulatedDeltaMagnitude { get; init; }
    public double MaximumDeltaMagnitude { get; init; }
    public double NonzeroDeltaFraction => (double)NonzeroDeltaCount / Math.Max(1, MoveEventCount);
    public double AbsolutePositionChangeFraction => (double)AbsolutePositionChangeCount / Math.Max(1, MoveEventCount - 1);
    public double MeanActiveDeltaMagnitude => AccumulatedDeltaMagnitude / Math.Max(1, NonzeroDeltaCount);
    public bool IsGameCamera => MoveEventCount >= 20 && NonzeroDeltaCount > 0 && AbsolutePositionChangeFraction < 0.05;
    public bool PositionsAreValid => OutOfCaptureBoundsCount == 0;
}

public static class InputEventReader
{
    public static InputEventSummary Summarize(string path, int previewLimit = 80, CodableRect? globalRect = null)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 128 * 1024, FileOptions.SequentialScan);
        ValidateHeaderAndLength(stream);
        var count = checked((int)((stream.Length - InputEventWriter.HeaderSize) / InputEventWriter.RecordSize));
        var preview = new List<InputSample>(Math.Min(Math.Max(0, previewLimit), count));
        var keys = new SortedSet<ushort>();
        var keyEvents = 0;
        var mouseEvents = 0;
        var moveCount = 0;
        var nonzeroDelta = 0;
        var positionChanges = 0;
        var outside = 0;
        var accumulated = 0.0;
        var maximum = 0.0;
        (double X, double Y)? previousPosition = null;
        InputSample? first = null;
        InputSample? last = null;
        ulong? priorTimestamp = null;
        var recordBytes = new byte[InputEventWriter.RecordSize];

        for (var index = 0; index < count; index++)
        {
            stream.ReadExactly(recordBytes);
            var sample = Decode(recordBytes);
            if (priorTimestamp is { } prior && sample.TimestampNanos < prior)
                throw new InvalidDataException("AgentTrainer input events are not ordered by capture time.");
            priorTimestamp = sample.TimestampNanos;
            first ??= sample;
            last = sample;
            if (preview.Count < previewLimit) preview.Add(sample);
            switch (sample.Kind)
            {
                case InputEventKind.Key:
                    keyEvents++;
                    keys.Add(sample.KeyCode);
                    break;
                case InputEventKind.Flags:
                    AddModifierKeys(keys, sample.Modifiers);
                    break;
                case InputEventKind.MouseMove:
                    mouseEvents++;
                    moveCount++;
                    var magnitude = Math.Abs(sample.DeltaX) + Math.Abs(sample.DeltaY);
                    if (magnitude > 0)
                    {
                        nonzeroDelta++;
                        accumulated += magnitude;
                        maximum = Math.Max(maximum, magnitude);
                    }
                    if (previousPosition is { } point &&
                        (Math.Abs(sample.X - point.X) > 0.01 || Math.Abs(sample.Y - point.Y) > 0.01)) positionChanges++;
                    previousPosition = (sample.X, sample.Y);
                    if (globalRect is { } rect && !ContainsWithTolerance(rect, sample.X, sample.Y)) outside++;
                    break;
                case InputEventKind.MouseButton:
                case InputEventKind.Scroll:
                    mouseEvents++;
                    break;
            }
        }

        return new InputEventSummary(count, keyEvents, mouseEvents, keys, preview,
            new MouseDiagnostics
            {
                MoveEventCount = moveCount,
                NonzeroDeltaCount = nonzeroDelta,
                AbsolutePositionChangeCount = positionChanges,
                OutOfCaptureBoundsCount = outside,
                AccumulatedDeltaMagnitude = accumulated,
                MaximumDeltaMagnitude = maximum
            }, first, last);
    }

    public static IReadOnlyList<InputSample> ReadAll(string path) => Summarize(path, int.MaxValue).Preview;

    private static void ValidateHeaderAndLength(FileStream stream)
    {
        if (stream.Length < InputEventWriter.HeaderSize ||
            (stream.Length - InputEventWriter.HeaderSize) % InputEventWriter.RecordSize != 0)
            throw new InvalidDataException("AgentTrainer input event file is incomplete.");
        Span<byte> header = stackalloc byte[InputEventWriter.HeaderSize];
        stream.ReadExactly(header);
        if (!header[..8].SequenceEqual(InputEventWriter.Magic) ||
            BinaryPrimitives.ReadUInt32LittleEndian(header[8..]) != InputEventWriter.FormatVersion)
            throw new InvalidDataException("AgentTrainer input event header is unsupported.");
    }

    private static InputSample Decode(ReadOnlySpan<byte> record)
    {
        var rawKind = record[8];
        if (!Enum.IsDefined(typeof(InputEventKind), rawKind))
            throw new InvalidDataException("AgentTrainer input event file contains an unknown event kind.");
        var sample = new InputSample(
            BinaryPrimitives.ReadUInt64LittleEndian(record),
            (InputEventKind)rawKind,
            ReadDouble(record[24..]), ReadDouble(record[32..]),
            ReadDouble(record[40..]), ReadDouble(record[48..]),
            record[10], ReadDouble(record[56..]), ReadDouble(record[64..]),
            BinaryPrimitives.ReadUInt16LittleEndian(record[12..]),
            BinaryPrimitives.ReadUInt64LittleEndian(record[16..]),
            record[9] != 0);
        if (!new[] { sample.X, sample.Y, sample.DeltaX, sample.DeltaY, sample.ScrollX, sample.ScrollY }.All(double.IsFinite))
            throw new InvalidDataException("AgentTrainer input event file contains a non-finite control value.");
        return sample;
    }

    private static double ReadDouble(ReadOnlySpan<byte> value) =>
        BitConverter.Int64BitsToDouble(BinaryPrimitives.ReadInt64LittleEndian(value));

    private static void AddModifierKeys(SortedSet<ushort> result, ulong flags)
    {
        if ((flags & QuartzModifierFlags.Shift) != 0) result.Add(56);
        if ((flags & QuartzModifierFlags.Control) != 0) result.Add(59);
        if ((flags & QuartzModifierFlags.Option) != 0) result.Add(58);
        if ((flags & QuartzModifierFlags.Command) != 0) result.Add(55);
    }

    private static bool ContainsWithTolerance(CodableRect rect, double x, double y) =>
        x >= rect.X - 0.5 && y >= rect.Y - 0.5
        && x <= rect.X + rect.Width + 0.5 && y <= rect.Y + rect.Height + 0.5;
}
