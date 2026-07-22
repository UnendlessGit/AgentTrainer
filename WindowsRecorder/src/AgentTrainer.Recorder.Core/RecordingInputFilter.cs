namespace AgentTrainer.Recorder.Core;

/// <summary>
/// Shared semantic filter for a recording session. Modifier transitions are
/// buffered until the next event reveals whether they belong to the recorder's
/// own global shortcut, matching the native macOS event-tap behavior.
/// </summary>
public sealed class RecordingInputFilter
{
    private readonly SortedSet<ushort> _excludedKeyCodes;
    private readonly ushort _shortcutKeyCode;
    private readonly ushort _shortcutVirtualKey;
    private readonly ulong _shortcutModifiers;
    private readonly List<InputSample> _pendingFlags = [];
    private bool _suppressedShortcutKey;
    private bool _suppressingModifierRelease;

    public RecordingInputFilter(
        IEnumerable<ushort> excludedKeyCodes,
        ushort shortcutKeyCode,
        ulong shortcutModifiers,
        ushort shortcutVirtualKey = 0)
    {
        _excludedKeyCodes = new SortedSet<ushort>(excludedKeyCodes);
        _shortcutKeyCode = shortcutKeyCode;
        _shortcutModifiers = shortcutModifiers;
        _shortcutVirtualKey = shortcutVirtualKey;
    }

    public IReadOnlyList<InputSample> Process(InputSample sample)
    {
        if (_suppressingModifierRelease)
        {
            if (sample.Kind == InputEventKind.Flags)
            {
                if ((sample.Modifiers & QuartzModifierFlags.Relevant) == 0)
                {
                    _suppressedShortcutKey = false;
                    _suppressingModifierRelease = false;
                }
                return [];
            }
            if (sample.Kind == InputEventKind.Key) return [];
            return [sample with { Modifiers = SanitizeModifiers(sample.Modifiers & ~_shortcutModifiers) }];
        }

        List<InputSample> hotkeyFiltered;
        switch (sample.Kind)
        {
            case InputEventKind.Flags:
                if (_suppressedShortcutKey)
                {
                    if ((sample.Modifiers & QuartzModifierFlags.Relevant) == 0)
                    {
                        _suppressedShortcutKey = false;
                        _suppressingModifierRelease = false;
                    }
                    return [];
                }
                _pendingFlags.Add(sample);
                if ((sample.Modifiers & QuartzModifierFlags.Relevant) != 0) return [];
                hotkeyFiltered = DrainPending();
                break;

            case InputEventKind.Key:
                if (IsShortcutKey(sample)
                    && (sample.Modifiers & QuartzModifierFlags.Relevant) == _shortcutModifiers)
                {
                    _pendingFlags.Clear();
                    _suppressingModifierRelease = _shortcutModifiers != 0;
                    _suppressedShortcutKey = sample.IsDown;
                    return [];
                }
                if (_suppressedShortcutKey && sample.KeyCode == _shortcutKeyCode)
                {
                    if (!sample.IsDown) _suppressedShortcutKey = false;
                    return [];
                }
                hotkeyFiltered = DrainPending();
                hotkeyFiltered.Add(sample);
                break;

            default:
                hotkeyFiltered = DrainPending();
                hotkeyFiltered.Add(sample);
                break;
        }

        var result = new List<InputSample>(hotkeyFiltered.Count);
        foreach (var value in hotkeyFiltered)
        {
            if (value.Kind == InputEventKind.Key && _excludedKeyCodes.Contains(value.KeyCode)) continue;
            result.Add(value with { Modifiers = SanitizeModifiers(value.Modifiers) });
        }
        return result;
    }

    /// <summary>
    /// Marks the already-active system hotkey chord before/while the capture
    /// lifecycle changes. This covers release-only input when recording starts
    /// after WM_HOTKEY and clears modifier events buffered before WM_HOTKEY is
    /// dispatched during stop.
    /// </summary>
    public void SuppressActiveShortcut(ulong currentModifiers)
    {
        _pendingFlags.Clear();
        if ((currentModifiers & _shortcutModifiers) != _shortcutModifiers) return;
        _suppressedShortcutKey = true;
        _suppressingModifierRelease = _shortcutModifiers != 0;
    }

    public ulong SanitizeModifiers(ulong modifiers)
    {
        if (_excludedKeyCodes.Overlaps([56, 60])) modifiers &= ~QuartzModifierFlags.Shift;
        if (_excludedKeyCodes.Overlaps([59, 62])) modifiers &= ~QuartzModifierFlags.Control;
        if (_excludedKeyCodes.Overlaps([58, 61])) modifiers &= ~QuartzModifierFlags.Option;
        if (_excludedKeyCodes.Overlaps([54, 55])) modifiers &= ~QuartzModifierFlags.Command;
        return modifiers;
    }

    private List<InputSample> DrainPending()
    {
        var result = new List<InputSample>(_pendingFlags);
        _pendingFlags.Clear();
        return result;
    }

    private bool IsShortcutKey(InputSample sample) =>
        sample.KeyCode == _shortcutKeyCode
        || (_shortcutVirtualKey != 0 && sample.NativeVirtualKey == _shortcutVirtualKey);
}
