using System.Buffers;
using System.Buffers.Binary;
using System.ComponentModel;
using System.Windows.Interop;
using AgentTrainer.Recorder.Core;

namespace AgentTrainer.Recorder;

internal sealed record InputStateSnapshot(
    IReadOnlyCollection<ushort> Keys,
    IReadOnlyCollection<byte> Buttons,
    ulong Modifiers,
    double DeltaX,
    double DeltaY,
    double ScrollX,
    double ScrollY)
{
    internal static readonly InputStateSnapshot Empty = new([], [], 0, 0, 0, 0, 0);
}

internal sealed class RawInputService : IDisposable
{
    private const ushort MouseLeftDown = 0x0001;
    private const ushort MouseLeftUp = 0x0002;
    private const ushort MouseRightDown = 0x0004;
    private const ushort MouseRightUp = 0x0008;
    private const ushort MouseMiddleDown = 0x0010;
    private const ushort MouseMiddleUp = 0x0020;
    private const ushort MouseButton4Down = 0x0040;
    private const ushort MouseButton4Up = 0x0080;
    private const ushort MouseButton5Down = 0x0100;
    private const ushort MouseButton5Up = 0x0200;
    private const ushort MouseWheel = 0x0400;
    private const ushort MouseHorizontalWheel = 0x0800;
    private const ushort MouseMoveAbsolute = 0x0001;
    private const int VirtualKeyCapsLock = 0x14;
    private const int WheelDelta = 120;
    private static readonly int HeaderSize = System.Runtime.InteropServices.Marshal.SizeOf<NativeMethods.RawInputHeader>();

    private readonly HashSet<uint> _pressedPhysicalKeys = [];
    private readonly HashSet<ushort> _pressedModifierKeys = [];
    private readonly HashSet<ushort> _keys = [];
    private readonly HashSet<byte> _buttons = [];
    private HwndSource? _source;
    private ulong _modifiers;
    private NativeMethods.Point _lastPointer;
    private bool _hasPointer;
    private bool _capsLock;
    private long _lastStateReportTicks;
    private bool _disposed;

    internal event Action<InputSample>? SampleReceived;
    internal event Action<InputStateSnapshot>? StateChanged;

    internal ulong CurrentModifiers => _modifiers;
    internal NativeMethods.Point CurrentPointer
    {
        get
        {
            if (NativeMethods.GetCursorPos(out var value)) return value;
            return _lastPointer;
        }
    }

    internal void Attach(HwndSource source)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_source is not null) return;
        _source = source;
        _capsLock = (NativeMethods.GetKeyState(VirtualKeyCapsLock) & 1) != 0;
        if (_capsLock) _modifiers |= QuartzModifierFlags.CapsLock;
        _ = NativeMethods.GetCursorPos(out _lastPointer);
        _hasPointer = true;
        source.AddHook(WindowProcedure);
        var devices = new[]
        {
            new NativeMethods.RawInputDevice { UsagePage = 0x01, Usage = 0x02, Flags = NativeMethods.RidevInputSink, Target = source.Handle },
            new NativeMethods.RawInputDevice { UsagePage = 0x01, Usage = 0x06, Flags = NativeMethods.RidevInputSink, Target = source.Handle }
        };
        if (!NativeMethods.RegisterRawInputDevices(devices, (uint)devices.Length,
                (uint)System.Runtime.InteropServices.Marshal.SizeOf<NativeMethods.RawInputDevice>()))
        {
            source.RemoveHook(WindowProcedure);
            _source = null;
            throw new Win32Exception(System.Runtime.InteropServices.Marshal.GetLastWin32Error(), "Windows could not register raw keyboard and mouse input.");
        }
    }

    private IntPtr WindowProcedure(IntPtr window, int message, IntPtr wParam, IntPtr lParam, ref bool handled)
    {
        _ = window;
        _ = wParam;
        if (message == NativeMethods.WmInput)
        {
            ConsumeRawInput(lParam);
            handled = false;
        }
        return IntPtr.Zero;
    }

    private void ConsumeRawInput(IntPtr handle)
    {
        uint size = 0;
        if (NativeMethods.GetRawInputData(handle, NativeMethods.RidInput, null, ref size, (uint)HeaderSize) != 0 || size < HeaderSize)
            return;
        var buffer = ArrayPool<byte>.Shared.Rent(checked((int)size));
        try
        {
            if (NativeMethods.GetRawInputData(handle, NativeMethods.RidInput, buffer, ref size, (uint)HeaderSize) != size) return;
            var type = BinaryPrimitives.ReadUInt32LittleEndian(buffer.AsSpan(0, 4));
            var timestamp = NativeMethods.HostNanos();
            if (type == NativeMethods.RimTypeMouse) ConsumeMouse(buffer.AsSpan(HeaderSize), timestamp);
            else if (type == NativeMethods.RimTypeKeyboard) ConsumeKeyboard(buffer.AsSpan(HeaderSize), timestamp);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    private void ConsumeMouse(ReadOnlySpan<byte> data, ulong timestamp)
    {
        if (data.Length < 24) return;
        var flags = BinaryPrimitives.ReadUInt16LittleEndian(data);
        var buttonFlags = BinaryPrimitives.ReadUInt16LittleEndian(data[4..]);
        var buttonData = BinaryPrimitives.ReadUInt16LittleEndian(data[6..]);
        var rawX = BinaryPrimitives.ReadInt32LittleEndian(data[12..]);
        var rawY = BinaryPrimitives.ReadInt32LittleEndian(data[16..]);
        if (NativeMethods.GetCursorPos(out var pointer))
        {
            var moved = (flags & MouseMoveAbsolute) == 0
                ? rawX != 0 || rawY != 0
                : !_hasPointer || pointer.X != _lastPointer.X || pointer.Y != _lastPointer.Y;
            if (moved)
            {
                Emit(new InputSample(timestamp, InputEventKind.MouseMove, pointer.X, pointer.Y,
                    (flags & MouseMoveAbsolute) == 0 ? rawX : pointer.X - _lastPointer.X,
                    (flags & MouseMoveAbsolute) == 0 ? rawY : pointer.Y - _lastPointer.Y,
                    Modifiers: _modifiers));
            }
            _lastPointer = pointer;
            _hasPointer = true;
        }
        else
        {
            pointer = _lastPointer;
        }

        EmitButton(buttonFlags, MouseLeftDown, MouseLeftUp, 0, pointer, timestamp);
        EmitButton(buttonFlags, MouseRightDown, MouseRightUp, 1, pointer, timestamp);
        EmitButton(buttonFlags, MouseMiddleDown, MouseMiddleUp, 2, pointer, timestamp);
        EmitButton(buttonFlags, MouseButton4Down, MouseButton4Up, 3, pointer, timestamp);
        EmitButton(buttonFlags, MouseButton5Down, MouseButton5Up, 4, pointer, timestamp);
        if ((buttonFlags & MouseWheel) != 0)
        {
            var delta = unchecked((short)buttonData) / (double)WheelDelta * 3.0;
            Emit(new InputSample(timestamp, InputEventKind.Scroll, pointer.X, pointer.Y, ScrollY: delta, Modifiers: _modifiers));
        }
        if ((buttonFlags & MouseHorizontalWheel) != 0)
        {
            var delta = unchecked((short)buttonData) / (double)WheelDelta * 3.0;
            Emit(new InputSample(timestamp, InputEventKind.Scroll, pointer.X, pointer.Y, ScrollX: delta, Modifiers: _modifiers));
        }
    }

    private void EmitButton(ushort flags, ushort downMask, ushort upMask, byte button, NativeMethods.Point pointer, ulong timestamp)
    {
        if ((flags & downMask) != 0) Emit(new InputSample(timestamp, InputEventKind.MouseButton, pointer.X, pointer.Y, Button: button, Modifiers: _modifiers, IsDown: true));
        if ((flags & upMask) != 0) Emit(new InputSample(timestamp, InputEventKind.MouseButton, pointer.X, pointer.Y, Button: button, Modifiers: _modifiers));
    }

    private void ConsumeKeyboard(ReadOnlySpan<byte> data, ulong timestamp)
    {
        if (data.Length < 16) return;
        var makeCode = BinaryPrimitives.ReadUInt16LittleEndian(data);
        var flags = BinaryPrimitives.ReadUInt16LittleEndian(data[2..]);
        var virtualKey = BinaryPrimitives.ReadUInt16LittleEndian(data[6..]);
        var isDown = (flags & NativeMethods.RiKeyBreak) == 0;
        var extended = (flags & NativeMethods.RiKeyE0) != 0;
        var e1 = (flags & NativeMethods.RiKeyE1) != 0;
        var physicalID = (uint)makeCode | (extended ? 1u << 16 : 0) | (e1 ? 1u << 17 : 0);
        if (isDown && !_pressedPhysicalKeys.Add(physicalID)) return;
        if (!isDown) _pressedPhysicalKeys.Remove(physicalID);
        var keyCode = MacKeyMap.Translate(makeCode, extended || e1, virtualKey);
        if (keyCode is null) return;

        if (keyCode == 57)
        {
            if (!isDown) return;
            _capsLock = !_capsLock;
            if (_capsLock) _modifiers |= QuartzModifierFlags.CapsLock;
            else _modifiers &= ~QuartzModifierFlags.CapsLock;
            Emit(new InputSample(timestamp, InputEventKind.Flags, KeyCode: keyCode.Value, Modifiers: _modifiers));
            return;
        }
        if (MacKeyMap.IsModifierKey(keyCode.Value))
        {
            var mask = MacKeyMap.ModifierMask(keyCode.Value);
            if (isDown) _pressedModifierKeys.Add(keyCode.Value);
            else _pressedModifierKeys.Remove(keyCode.Value);
            if (_pressedModifierKeys.Any(code => MacKeyMap.ModifierMask(code) == mask)) _modifiers |= mask;
            else _modifiers &= ~mask;
            Emit(new InputSample(timestamp, InputEventKind.Flags, KeyCode: keyCode.Value, Modifiers: _modifiers));
            return;
        }
        Emit(new InputSample(timestamp, InputEventKind.Key, KeyCode: keyCode.Value, Modifiers: _modifiers,
            IsDown: isDown, NativeVirtualKey: virtualKey));
    }

    private void Emit(InputSample sample)
    {
        switch (sample.Kind)
        {
            case InputEventKind.Key:
                if (sample.IsDown) _keys.Add(sample.KeyCode); else _keys.Remove(sample.KeyCode);
                break;
            case InputEventKind.Flags:
                SetModifierState([56, 60], QuartzModifierFlags.Shift, sample.Modifiers);
                SetModifierState([59, 62], QuartzModifierFlags.Control, sample.Modifiers);
                SetModifierState([58, 61], QuartzModifierFlags.Option, sample.Modifiers);
                SetModifierState([55, 54], QuartzModifierFlags.Command, sample.Modifiers);
                break;
            case InputEventKind.MouseButton:
                if (sample.IsDown) _buttons.Add(sample.Button); else _buttons.Remove(sample.Button);
                break;
        }
        SampleReceived?.Invoke(sample);
        var now = Environment.TickCount64;
        var controlsChanged = sample.Kind is InputEventKind.Key or InputEventKind.Flags or InputEventKind.MouseButton;
        if (controlsChanged || now - _lastStateReportTicks >= 33)
        {
            _lastStateReportTicks = now;
            StateChanged?.Invoke(new InputStateSnapshot(_keys.Order().ToArray(), _buttons.Order().ToArray(), _modifiers,
                sample.Kind == InputEventKind.MouseMove ? sample.DeltaX : 0,
                sample.Kind == InputEventKind.MouseMove ? sample.DeltaY : 0,
                sample.Kind == InputEventKind.Scroll ? sample.ScrollX : 0,
                sample.Kind == InputEventKind.Scroll ? sample.ScrollY : 0));
        }
    }

    private void SetModifierState(IEnumerable<ushort> codes, ulong mask, ulong flags)
    {
        if ((flags & mask) != 0) _keys.Add(codes.First());
        else foreach (var code in codes) _keys.Remove(code);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _source?.RemoveHook(WindowProcedure);
        _source = null;
        SampleReceived = null;
        StateChanged?.Invoke(InputStateSnapshot.Empty);
        StateChanged = null;
    }
}

internal sealed class RecordingInputSink
{
    private readonly object _gate = new();
    private readonly InputEventWriter _writer;
    private readonly RecordingInputFilter _filter;
    private readonly ulong _shortcutModifiers;
    private readonly bool _stripShortcutModifiersAtStart;
    private ulong? _hostStart;
    private ulong? _lastEvent;

    internal RecordingInputSink(InputEventWriter writer, IEnumerable<ushort> excludedKeyCodes, RecordingHotkeyBinding hotkey, bool startedByGlobalHotkey)
    {
        _writer = writer;
        _shortcutModifiers = hotkey.QuartzModifiers;
        _stripShortcutModifiersAtStart = startedByGlobalHotkey;
        _filter = new RecordingInputFilter(excludedKeyCodes, hotkey.MacKeyCode, hotkey.QuartzModifiers, hotkey.VirtualKey);
    }

    internal ulong HostStart => _hostStart ?? 0;
    internal ulong LastEvent => _lastEvent ?? 0;

    internal void Accept(InputSample rawSample)
    {
        lock (_gate)
        {
            foreach (var sample in _filter.Process(rawSample))
            {
                if (_hostStart is not { } start || sample.TimestampNanos < start) continue;
                _writer.Append(sample);
                _lastEvent = sample.TimestampNanos;
            }
        }
    }

    internal void StartAtFirstFrame(ulong hostNanos, NativeMethods.Point pointer, ulong currentModifiers)
    {
        lock (_gate)
        {
            if (_hostStart is not null) return;
            var modifiers = _filter.SanitizeModifiers(_stripShortcutModifiersAtStart ? currentModifiers & ~_shortcutModifiers : currentModifiers);
            _writer.Append(new InputSample(hostNanos, InputEventKind.MouseMove, pointer.X, pointer.Y, Modifiers: modifiers));
            _hostStart = hostNanos;
            _lastEvent = hostNanos;
        }
    }

    internal void SuppressActiveHotkey(ulong currentModifiers)
    {
        lock (_gate) _filter.SuppressActiveShortcut(currentModifiers);
    }
}
