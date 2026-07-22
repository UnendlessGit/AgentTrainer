using System.Text.Json.Serialization;
using AgentTrainer.Recorder.Core;

namespace AgentTrainer.Recorder;

internal sealed record RecordingHotkeyBinding
{
    internal static RecordingHotkeyBinding Default { get; } = new();

    public ushort VirtualKey { get; init; } = 0x52; // R
    public ushort MacKeyCode { get; init; } = 15;
    public bool Control { get; init; } = true;
    public bool Alt { get; init; } = true;
    public bool Shift { get; init; }
    public bool Windows { get; init; } = true;

    [JsonIgnore]
    internal uint NativeModifiers =>
        (Control ? NativeMethods.ModControl : 0)
        | (Alt ? NativeMethods.ModAlt : 0)
        | (Shift ? NativeMethods.ModShift : 0)
        | (Windows ? NativeMethods.ModWin : 0);

    [JsonIgnore]
    internal ulong QuartzModifiers =>
        (Control ? QuartzModifierFlags.Control : 0)
        | (Alt ? QuartzModifierFlags.Option : 0)
        | (Shift ? QuartzModifierFlags.Shift : 0)
        | (Windows ? QuartzModifierFlags.Command : 0);

    [JsonIgnore]
    internal bool HasModifier => Control || Alt || Shift || Windows;

    [JsonIgnore]
    internal string DisplayText
    {
        get
        {
            var parts = new List<string>(5);
            if (Control) parts.Add("Ctrl");
            if (Alt) parts.Add("Alt");
            if (Shift) parts.Add("Shift");
            if (Windows) parts.Add("Win");
            parts.Add(RecordingHotkeyCatalog.Find(VirtualKey)?.Name ?? $"VK {VirtualKey}");
            return string.Join("+", parts);
        }
    }

    internal static RecordingHotkeyBinding Sanitize(RecordingHotkeyBinding? value)
    {
        if (value is null || !value.HasModifier) return Default;
        var key = RecordingHotkeyCatalog.Find(value.VirtualKey);
        return key is null ? Default : value with { MacKeyCode = key.MacKeyCode };
    }
}

internal sealed record RecordingHotkeyKey(ushort VirtualKey, ushort MacKeyCode, string Name);

internal static class RecordingHotkeyCatalog
{
    private static readonly IReadOnlyList<RecordingHotkeyKey> Keys =
    [
        new(0x41, 0, "A"), new(0x42, 11, "B"), new(0x43, 8, "C"), new(0x44, 2, "D"),
        new(0x45, 14, "E"), new(0x46, 3, "F"), new(0x47, 5, "G"), new(0x48, 4, "H"),
        new(0x49, 34, "I"), new(0x4A, 38, "J"), new(0x4B, 40, "K"), new(0x4C, 37, "L"),
        new(0x4D, 46, "M"), new(0x4E, 45, "N"), new(0x4F, 31, "O"), new(0x50, 35, "P"),
        new(0x51, 12, "Q"), new(0x52, 15, "R"), new(0x53, 1, "S"), new(0x54, 17, "T"),
        new(0x55, 32, "U"), new(0x56, 9, "V"), new(0x57, 13, "W"), new(0x58, 7, "X"),
        new(0x59, 16, "Y"), new(0x5A, 6, "Z"),
        new(0x30, 29, "0"), new(0x31, 18, "1"), new(0x32, 19, "2"), new(0x33, 20, "3"),
        new(0x34, 21, "4"), new(0x35, 23, "5"), new(0x36, 22, "6"), new(0x37, 26, "7"),
        new(0x38, 28, "8"), new(0x39, 25, "9"),
        new(0x70, 122, "F1"), new(0x71, 120, "F2"), new(0x72, 99, "F3"), new(0x73, 118, "F4"),
        new(0x74, 96, "F5"), new(0x75, 97, "F6"), new(0x76, 98, "F7"), new(0x77, 100, "F8"),
        new(0x78, 101, "F9"), new(0x79, 109, "F10"), new(0x7A, 103, "F11"), new(0x7B, 111, "F12"),
        new(0x1B, 53, "Esc"), new(0x20, 49, "Space")
    ];

    internal static RecordingHotkeyKey? Find(ushort virtualKey) => Keys.FirstOrDefault(value => value.VirtualKey == virtualKey);
}
