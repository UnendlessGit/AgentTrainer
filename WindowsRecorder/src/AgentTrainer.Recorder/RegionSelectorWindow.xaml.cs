using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Interop;
using AgentTrainer.Recorder.Core;
using ScreenRecorderLib;
using ScreenRecorder = ScreenRecorderLib.Recorder;

namespace AgentTrainer.Recorder;

public partial class RegionSelectorWindow : Window
{
    private readonly CodableRect _screenBounds;
    private NativeMethods.Point _dragStart;
    private bool _dragging;

    internal RegionSelectorWindow(CodableRect screenBounds)
    {
        _screenBounds = screenBounds;
        InitializeComponent();
        SourceInitialized += (_, _) =>
        {
            var handle = new WindowInteropHelper(this).Handle;
            _ = NativeMethods.SetWindowPos(handle, NativeMethods.HwndTopmost,
                checked((int)Math.Round(_screenBounds.X)), checked((int)Math.Round(_screenBounds.Y)),
                checked((int)Math.Round(_screenBounds.Width)), checked((int)Math.Round(_screenBounds.Height)),
                NativeMethods.SwpNoActivate | NativeMethods.SwpShowWindow);
            _ = ScreenRecorder.SetExcludeFromCapture(handle, true);
        };
    }

    internal CodableRect? Selection { get; private set; }

    private void OnMouseDown(object sender, MouseButtonEventArgs args)
    {
        _ = sender;
        _ = args;
        if (!NativeMethods.GetCursorPos(out _dragStart)) return;
        _dragStart = Clamp(_dragStart);
        _dragging = true;
        CaptureMouse();
        SelectionBorder.Visibility = Visibility.Visible;
        UpdateSelection(_dragStart);
    }

    private void OnMouseMove(object sender, MouseEventArgs args)
    {
        _ = sender;
        _ = args;
        if (_dragging && NativeMethods.GetCursorPos(out var point)) UpdateSelection(Clamp(point));
    }

    private void OnMouseUp(object sender, MouseButtonEventArgs args)
    {
        _ = sender;
        _ = args;
        if (!_dragging) return;
        _dragging = false;
        ReleaseMouseCapture();
        if (!NativeMethods.GetCursorPos(out var point)) point = _dragStart;
        point = Clamp(point);
        var left = Math.Min(_dragStart.X, point.X);
        var top = Math.Min(_dragStart.Y, point.Y);
        var width = Math.Abs(point.X - _dragStart.X);
        var height = Math.Abs(point.Y - _dragStart.Y);
        if (width >= 2 && height >= 2)
        {
            Selection = new CodableRect { X = left, Y = top, Width = width, Height = height };
            DialogResult = true;
        }
        else
        {
            SelectionBorder.Visibility = Visibility.Collapsed;
        }
    }

    private void OnKeyDown(object sender, KeyEventArgs args)
    {
        _ = sender;
        if (args.Key == Key.Escape)
        {
            args.Handled = true;
            DialogResult = false;
        }
    }

    private void UpdateSelection(NativeMethods.Point point)
    {
        var left = Math.Min(_dragStart.X, point.X);
        var top = Math.Min(_dragStart.Y, point.Y);
        var width = Math.Abs(point.X - _dragStart.X);
        var height = Math.Abs(point.Y - _dragStart.Y);
        var scaleX = ActualWidth / Math.Max(1, _screenBounds.Width);
        var scaleY = ActualHeight / Math.Max(1, _screenBounds.Height);
        Canvas.SetLeft(SelectionBorder, (left - _screenBounds.X) * scaleX);
        Canvas.SetTop(SelectionBorder, (top - _screenBounds.Y) * scaleY);
        SelectionBorder.Width = Math.Max(1, width * scaleX);
        SelectionBorder.Height = Math.Max(1, height * scaleY);
        SizeLabel.Text = $"{width} × {height}";
    }

    private NativeMethods.Point Clamp(NativeMethods.Point point) => new()
    {
        X = Math.Clamp(point.X, checked((int)_screenBounds.X), checked((int)(_screenBounds.X + _screenBounds.Width))),
        Y = Math.Clamp(point.Y, checked((int)_screenBounds.Y), checked((int)(_screenBounds.Y + _screenBounds.Height)))
    };
}
