using AgentTrainer.Recorder.Core;
using ScreenRecorderLib;
using ScreenRecorder = ScreenRecorderLib.Recorder;

namespace AgentTrainer.Recorder;

internal sealed record CaptureSourceItem(
    string Name,
    string Detail,
    string StableKey,
    string? DeviceName,
    IntPtr WindowHandle,
    CodableRect Bounds,
    uint PortableID)
{
    internal bool IsDisplay => DeviceName is not null;
    public override string ToString() => $"{Name} — {Detail}";
}

internal sealed record CapturePlan(
    CaptureSpec Spec,
    CodableRect GlobalRect,
    RecordingSourceBase RecordingSource,
    int PixelWidth,
    int PixelHeight);

internal static class CaptureSourceCatalog
{
    internal static IReadOnlyList<CaptureSourceItem> GetDisplays()
    {
        var boundsByDevice = EnumerateMonitorBounds();
        return ScreenRecorder.GetDisplays()
            .Where(display => !string.IsNullOrWhiteSpace(display.DeviceName) && boundsByDevice.ContainsKey(display.DeviceName))
            .Select(display =>
            {
                var bounds = boundsByDevice[display.DeviceName];
                var name = string.IsNullOrWhiteSpace(display.FriendlyName) ? display.DeviceName : display.FriendlyName;
                return new CaptureSourceItem(name, $"{bounds.Width:0} × {bounds.Height:0}", $"display:{display.DeviceName}",
                    display.DeviceName, IntPtr.Zero, bounds, StableID(display.DeviceName));
            })
            .OrderBy(value => value.Bounds.X).ThenBy(value => value.Bounds.Y).ToArray();
    }

    internal static IReadOnlyList<CaptureSourceItem> GetWindows()
    {
        var processID = (uint)Environment.ProcessId;
        var result = new List<CaptureSourceItem>();
        foreach (var window in ScreenRecorder.GetWindows())
        {
            if (!NativeMethods.IsWindow(window.Handle) || window.IsMinmimized()) continue;
            _ = NativeMethods.GetWindowThreadProcessId(window.Handle, out var ownerID);
            if (ownerID == processID) continue;
            if (!TryGetWindowBounds(window.Handle, out var bounds) || bounds.Width < 2 || bounds.Height < 2) continue;
            var title = string.IsNullOrWhiteSpace(window.Title) ? "Untitled window" : window.Title.Trim();
            result.Add(new CaptureSourceItem(title, $"{bounds.Width:0} × {bounds.Height:0}", $"window:{window.Handle.ToInt64():X}",
                null, window.Handle, bounds, StableID(window.Handle.ToInt64().ToString("X16", System.Globalization.CultureInfo.InvariantCulture))));
        }
        return result.OrderBy(value => value.Name, StringComparer.OrdinalIgnoreCase).ToArray();
    }

    internal static CapturePlan CreatePlan(CaptureSourceItem source, string captureKind, CodableRect? requestedRegion, int framesPerSecond, bool showsCursor)
    {
        if (!CaptureKinds.IsValid(captureKind)) throw new InvalidOperationException("Choose a supported capture type.");
        if (source.IsDisplay != (captureKind is CaptureKinds.Display or CaptureKinds.ScreenRegion))
            throw new InvalidOperationException("The selected source does not match the capture type.");

        CodableRect globalRect;
        ScreenRect? localCrop = null;
        CodableRect? storedRegion = null;
        if (captureKind == CaptureKinds.ScreenRegion)
        {
            var region = RequireRegion(requestedRegion).Intersect(source.Bounds);
            RequireArea(region);
            globalRect = region;
            storedRegion = region;
            localCrop = new ScreenRect(region.X - source.Bounds.X, region.Y - source.Bounds.Y, region.Width, region.Height);
        }
        else if (captureKind == CaptureKinds.WindowRegion)
        {
            var localBounds = new CodableRect { Width = source.Bounds.Width, Height = source.Bounds.Height };
            var region = RequireRegion(requestedRegion).Intersect(localBounds);
            RequireArea(region);
            storedRegion = region;
            globalRect = new CodableRect { X = source.Bounds.X + region.X, Y = source.Bounds.Y + region.Y, Width = region.Width, Height = region.Height };
            localCrop = new ScreenRect(region.X, region.Y, region.Width, region.Height);
        }
        else
        {
            globalRect = source.Bounds;
        }

        var width = EvenDimension(globalRect.Width);
        var height = EvenDimension(globalRect.Height);
        RecordingSourceBase recordingSource;
        if (source.DeviceName is { } deviceName)
        {
            recordingSource = new DisplayRecordingSource(deviceName)
            {
                RecorderApi = RecorderApi.DesktopDuplication,
                IsCursorCaptureEnabled = showsCursor,
                IsBorderRequired = false,
                SourceRect = localCrop
            };
        }
        else
        {
            if (!NativeMethods.IsWindow(source.WindowHandle)) throw new InvalidOperationException("The selected window is no longer available.");
            recordingSource = new WindowRecordingSource(source.WindowHandle)
            {
                IsCursorCaptureEnabled = showsCursor,
                IsBorderRequired = false,
                SourceRect = localCrop
            };
        }

        var spec = new CaptureSpec
        {
            Kind = captureKind,
            DisplayID = source.IsDisplay ? source.PortableID : null,
            WindowID = source.IsDisplay ? null : source.PortableID,
            Region = storedRegion,
            RequestedFPS = framesPerSecond,
            ShowsCursor = showsCursor
        };
        return new CapturePlan(spec, globalRect, recordingSource, width, height);
    }

    private static Dictionary<string, CodableRect> EnumerateMonitorBounds()
    {
        var result = new Dictionary<string, CodableRect>(StringComparer.OrdinalIgnoreCase);
        NativeMethods.MonitorEnum callback = delegate(IntPtr monitor, IntPtr deviceContext, ref NativeMethods.Rect monitorRect, IntPtr data)
        {
            _ = deviceContext;
            _ = monitorRect;
            _ = data;
            var info = new NativeMethods.MonitorInfo { Size = (uint)System.Runtime.InteropServices.Marshal.SizeOf<NativeMethods.MonitorInfo>(), DeviceName = string.Empty };
            if (NativeMethods.GetMonitorInfo(monitor, ref info))
            {
                result[info.DeviceName] = new CodableRect { X = info.Monitor.Left, Y = info.Monitor.Top, Width = info.Monitor.Width, Height = info.Monitor.Height };
            }
            return true;
        };
        if (!NativeMethods.EnumDisplayMonitors(IntPtr.Zero, IntPtr.Zero, callback, IntPtr.Zero))
            throw new InvalidOperationException("Windows could not enumerate connected displays.");
        return result;
    }

    private static bool TryGetWindowBounds(IntPtr window, out CodableRect bounds)
    {
        NativeMethods.Rect rect;
        if (NativeMethods.DwmGetWindowAttribute(window, NativeMethods.DwmwaExtendedFrameBounds, out rect,
                System.Runtime.InteropServices.Marshal.SizeOf<NativeMethods.Rect>()) != 0
            && !NativeMethods.GetWindowRect(window, out rect))
        {
            bounds = new CodableRect();
            return false;
        }
        bounds = new CodableRect { X = rect.Left, Y = rect.Top, Width = rect.Width, Height = rect.Height };
        return true;
    }

    private static CodableRect RequireRegion(CodableRect? region)
    {
        if (region is null || !region.IsFinite) throw new InvalidOperationException("Enter or draw a valid capture region.");
        return region;
    }

    private static void RequireArea(CodableRect region)
    {
        if (region.Width < 2 || region.Height < 2) throw new InvalidOperationException("The selected region is empty or outside the source.");
    }

    private static int EvenDimension(double value) => Math.Clamp(((int)Math.Floor(value)) & ~1, 2, 32768);

    private static uint StableID(string value)
    {
        const uint offset = 2166136261;
        const uint prime = 16777619;
        var result = offset;
        foreach (var character in value)
        {
            result ^= character;
            result *= prime;
        }
        return result;
    }
}
