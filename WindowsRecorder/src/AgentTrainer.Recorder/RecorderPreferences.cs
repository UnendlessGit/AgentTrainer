using System.Text.Json;
using System.Text.Json.Serialization;
using AgentTrainer.Recorder.Core;

namespace AgentTrainer.Recorder;

internal sealed record RecorderPreferences
{
    internal const string DefaultCaptureKind = CaptureKinds.Display;
    public string CaptureKind { get; init; } = DefaultCaptureKind;
    public int FramesPerSecond { get; init; } = 60;
    public bool ShowsCursor { get; init; }
    public double TrimStart { get; init; } = 0.5;
    public double TrimEnd { get; init; } = 0.5;
    public Guid? DestinationFolderID { get; init; }
    public string? DisplayDeviceName { get; init; }
    public CodableRect? ScreenRegion { get; init; }
    public CodableRect? WindowRegion { get; init; }
    public SortedSet<ushort> ExcludedKeyCodes { get; init; } = [];
    public bool PreferHevc { get; init; } = true;
    public RecordingHotkeyBinding Hotkey { get; init; } = RecordingHotkeyBinding.Default;
    public bool ShowCaptureHud { get; init; } = true;
    public bool OpenLibraryAfterRecording { get; init; } = true;
    public bool MinimizeWhileRecording { get; init; }
}

internal sealed class PreferencesStore
{
    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    internal PreferencesStore(string applicationRoot) => Path = System.IO.Path.Combine(applicationRoot, "recorder-settings.json");
    internal string Path { get; }

    internal RecorderPreferences Load()
    {
        try
        {
            if (!File.Exists(Path)) return new RecorderPreferences();
            var value = JsonSerializer.Deserialize<RecorderPreferences>(File.ReadAllBytes(Path), Options) ?? new RecorderPreferences();
            return Sanitize(value);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or JsonException)
        {
            AppLog.Write("Settings", error.ToString());
            return new RecorderPreferences();
        }
    }

    internal void Save(RecorderPreferences preferences)
    {
        var value = Sanitize(preferences);
        Directory.CreateDirectory(System.IO.Path.GetDirectoryName(Path)!);
        var temporary = Path + $".tmp-{Guid.NewGuid():N}";
        try
        {
            File.WriteAllBytes(temporary, JsonSerializer.SerializeToUtf8Bytes(value, Options));
            File.Move(temporary, Path, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporary)) File.Delete(temporary);
        }
    }

    private static RecorderPreferences Sanitize(RecorderPreferences value) => value with
    {
        CaptureKind = CaptureKinds.IsValid(value.CaptureKind) ? value.CaptureKind : RecorderPreferences.DefaultCaptureKind,
        FramesPerSecond = Math.Clamp(value.FramesPerSecond, 1, 240),
        TrimStart = double.IsFinite(value.TrimStart) ? Math.Clamp(value.TrimStart, 0, 3600) : 0.5,
        TrimEnd = double.IsFinite(value.TrimEnd) ? Math.Clamp(value.TrimEnd, 0, 3600) : 0.5,
        ExcludedKeyCodes = new SortedSet<ushort>(value.ExcludedKeyCodes.Where(code => code < 128)),
        Hotkey = RecordingHotkeyBinding.Sanitize(value.Hotkey)
    };
}
