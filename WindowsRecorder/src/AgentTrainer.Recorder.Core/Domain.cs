using System.Text.Json.Serialization;

namespace AgentTrainer.Recorder.Core;

public static class CaptureKinds
{
    public const string Display = "Display";
    public const string Window = "Window";
    public const string WindowRegion = "Window Region";
    public const string ScreenRegion = "Screen Region";

    public static readonly IReadOnlyList<string> All = [Display, Window, WindowRegion, ScreenRegion];
    public static bool IsValid(string value) => All.Contains(value, StringComparer.Ordinal);
}

public sealed record CodableRect
{
    public double X { get; init; }
    public double Y { get; init; }
    public double Width { get; init; }
    public double Height { get; init; }

    [JsonIgnore]
    public bool IsFinite => double.IsFinite(X) && double.IsFinite(Y) && double.IsFinite(Width) && double.IsFinite(Height);

    public CodableRect Intersect(CodableRect other)
    {
        var left = Math.Max(X, other.X);
        var top = Math.Max(Y, other.Y);
        var right = Math.Min(X + Width, other.X + other.Width);
        var bottom = Math.Min(Y + Height, other.Y + other.Height);
        return new CodableRect { X = left, Y = top, Width = Math.Max(0, right - left), Height = Math.Max(0, bottom - top) };
    }
}

public sealed record CaptureSpec
{
    public string Kind { get; init; } = CaptureKinds.Display;
    public uint? DisplayID { get; init; }
    public uint? WindowID { get; init; }
    public CodableRect? Region { get; init; }
    public double RequestedFPS { get; init; } = 60;
    public bool ShowsCursor { get; init; }
}

public sealed record RecordingManifest
{
    public int SchemaVersion { get; init; } = 2;
    public Guid Id { get; init; }
    public string Name { get; init; } = "Recording";
    public DateTimeOffset CreatedAt { get; init; }
    public ulong HostStartNanos { get; init; }
    public double Duration { get; init; }
    public CaptureSpec Capture { get; init; } = new();
    public CodableRect GlobalRect { get; init; } = new();
    public int PixelWidth { get; init; }
    public int PixelHeight { get; init; }
    public double DeliveredFPS { get; init; }
    public int EventCount { get; init; }
    public string VideoFile { get; init; } = "capture.mov";
    public string EventFile { get; init; } = "events.atrevents";
    public double TrimStart { get; init; }
    public double? TrimEnd { get; init; }
    public Guid? FolderID { get; init; }
    public string? ThumbnailFile { get; init; }
    public SortedSet<ushort>? ExcludedKeyCodes { get; init; }

    [JsonIgnore]
    public double EffectiveEnd => TrimEnd ?? Duration;

    [JsonIgnore]
    public double EffectiveDuration => Math.Max(0, EffectiveEnd - TrimStart);

    [JsonIgnore]
    public bool IsStructurallyValid
    {
        get
        {
            var safeNames = new[] { VideoFile, EventFile }.Concat(ThumbnailFile is null ? [] : [ThumbnailFile]);
            return SchemaVersion is >= 1 and <= 2
                && Id != Guid.Empty
                && !string.IsNullOrWhiteSpace(Name)
                && double.IsFinite(Duration) && Duration >= 0
                && CaptureKinds.IsValid(Capture.Kind)
                && double.IsFinite(Capture.RequestedFPS) && Capture.RequestedFPS is > 0 and <= 1000
                && GlobalRect.IsFinite && GlobalRect.Width >= 0 && GlobalRect.Height >= 0
                && PixelWidth is > 0 and <= 32768 && PixelHeight is > 0 and <= 32768
                && double.IsFinite(DeliveredFPS) && DeliveredFPS is >= 0 and <= 1000
                && EventCount >= 0
                && double.IsFinite(TrimStart) && TrimStart >= 0 && TrimStart <= Duration
                && double.IsFinite(EffectiveEnd) && EffectiveEnd >= TrimStart && EffectiveEnd <= Duration
                && safeNames.All(RecordingPath.IsSafeLeafName);
        }
    }
}

public sealed record RecordingFolder
{
    public Guid Id { get; init; }
    public string Name { get; init; } = "Recordings";
    public DateTimeOffset CreatedAt { get; init; }
}

public sealed record RecordingItem(RecordingManifest Manifest, string DirectoryPath)
{
    public Guid Id => Manifest.Id;
    public string VideoPath => Path.Combine(DirectoryPath, Manifest.VideoFile);
    public string EventPath => Path.Combine(DirectoryPath, Manifest.EventFile);
    public string? ThumbnailPath => Manifest.ThumbnailFile is { } name ? Path.Combine(DirectoryPath, name) : null;
}

public enum InputEventKind : byte
{
    MouseMove = 1,
    MouseButton = 2,
    Scroll = 3,
    Key = 4,
    Flags = 5
}

public readonly record struct InputSample(
    ulong TimestampNanos,
    InputEventKind Kind,
    double X = 0,
    double Y = 0,
    double DeltaX = 0,
    double DeltaY = 0,
    byte Button = 0,
    double ScrollX = 0,
    double ScrollY = 0,
    ushort KeyCode = 0,
    ulong Modifiers = 0,
    bool IsDown = false,
    // Runtime-only identity used to recognize the recorder hotkey when a VM or
    // synthetic keyboard supplies an inconsistent scan code. It is never
    // serialized into the portable 72-byte event record.
    ushort NativeVirtualKey = 0);

public static class QuartzModifierFlags
{
    public const ulong CapsLock = 0x0001_0000;
    public const ulong Shift = 0x0002_0000;
    public const ulong Control = 0x0004_0000;
    public const ulong Option = 0x0008_0000;
    public const ulong Command = 0x0010_0000;
    public const ulong Relevant = Shift | Control | Option | Command;
}

public static class RecordingPath
{
    public static bool IsSafeLeafName(string value) =>
        !string.IsNullOrEmpty(value)
        && value is not "." and not ".."
        && value.IndexOfAny(['/', '\\', ':', '\0']) < 0
        && Path.GetFileName(value).Equals(value, StringComparison.Ordinal);
}
