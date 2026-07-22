using System.ComponentModel;
using System.Windows.Interop;

namespace AgentTrainer.Recorder;

internal sealed class GlobalRecordingHotkey : IDisposable
{
    private const int HotkeyID = 0x4154;
    private HwndSource? _source;
    private RecordingHotkeyBinding _binding = RecordingHotkeyBinding.Default;
    private bool _disposed;

    internal event Action? Pressed;

    internal RecordingHotkeyBinding Binding => _binding;
    internal bool IsAttached => _source is not null;

    internal void Attach(HwndSource source, RecordingHotkeyBinding binding)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_source is not null) return;
        source.AddHook(WindowProcedure);
        var sanitized = RecordingHotkeyBinding.Sanitize(binding);
        if (!Register(source, sanitized))
        {
            source.RemoveHook(WindowProcedure);
            throw new Win32Exception(System.Runtime.InteropServices.Marshal.GetLastWin32Error(),
                $"{sanitized.DisplayText} is already registered by Windows or another application.");
        }
        _binding = sanitized;
        _source = source;
    }

    internal void Update(RecordingHotkeyBinding binding)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var sanitized = RecordingHotkeyBinding.Sanitize(binding);
        if (sanitized == _binding) return;
        if (_source is not { } source)
        {
            _binding = sanitized;
            return;
        }

        _ = NativeMethods.UnregisterHotKey(source.Handle, HotkeyID);
        if (Register(source, sanitized))
        {
            _binding = sanitized;
            return;
        }

        var error = System.Runtime.InteropServices.Marshal.GetLastWin32Error();
        if (!Register(source, _binding))
            throw new Win32Exception(error, "Windows rejected the new shortcut and the previous shortcut could not be restored. Restart AgentTrainer Recorder.");
        throw new Win32Exception(error, $"{sanitized.DisplayText} is already registered by Windows or another application.");
    }

    private static bool Register(HwndSource source, RecordingHotkeyBinding binding) =>
        NativeMethods.RegisterHotKey(source.Handle, HotkeyID, binding.NativeModifiers | NativeMethods.ModNoRepeat, binding.VirtualKey);

    private IntPtr WindowProcedure(IntPtr window, int message, IntPtr wParam, IntPtr lParam, ref bool handled)
    {
        _ = window;
        _ = lParam;
        if (message == NativeMethods.WmHotkey && wParam.ToInt32() == HotkeyID)
        {
            handled = true;
            Pressed?.Invoke();
        }
        return IntPtr.Zero;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_source is { } source)
        {
            _ = NativeMethods.UnregisterHotKey(source.Handle, HotkeyID);
            source.RemoveHook(WindowProcedure);
        }
        _source = null;
        Pressed = null;
    }
}
