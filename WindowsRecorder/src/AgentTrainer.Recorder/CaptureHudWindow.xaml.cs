using System.Windows;
using System.Windows.Interop;
using System.Windows.Threading;
using AgentTrainer.Recorder.Core;
using ScreenRecorderLib;
using ScreenRecorder = ScreenRecorderLib.Recorder;

namespace AgentTrainer.Recorder;

public partial class CaptureHudWindow : Window
{
    private readonly DispatcherTimer _timer;
    private DateTimeOffset _startedAt;

    internal CaptureHudWindow()
    {
        InitializeComponent();
        _timer = new DispatcherTimer(TimeSpan.FromMilliseconds(250), DispatcherPriority.Background, (_, _) => UpdateDuration(), Dispatcher);
        SourceInitialized += (_, _) => _ = ScreenRecorder.SetExcludeFromCapture(new WindowInteropHelper(this).Handle, true);
        Loaded += (_, _) =>
        {
            Left = SystemParameters.WorkArea.Right - ActualWidth - 18;
            Top = SystemParameters.WorkArea.Top + 18;
        };
    }

    internal event Action? StopRequested;

    internal void Begin(DateTimeOffset startedAt, string hotkey)
    {
        _startedAt = startedAt;
        HotkeyLabel.Text = $"{hotkey} to stop & save";
        ControlsLabel.Text = "Waiting for input";
        UpdateDuration();
        if (!IsVisible) Show();
        _timer.Start();
    }

    internal void UpdateControls(InputStateSnapshot state)
    {
        var keys = state.Keys.Select(MacKeyMap.Name).ToArray();
        var buttons = state.Buttons.Select(value => $"Mouse {value + 1}");
        var controls = keys.Concat(buttons).ToArray();
        ControlsLabel.Text = controls.Length == 0
            ? $"Mouse Δ{state.DeltaX:0}, {state.DeltaY:0}"
            : string.Join("  ", controls);
    }

    internal void End()
    {
        _timer.Stop();
        Hide();
    }

    private void UpdateDuration()
    {
        var elapsed = DateTimeOffset.UtcNow - _startedAt;
        DurationLabel.Text = $"REC {(int)elapsed.TotalMinutes:00}:{elapsed.Seconds:00}";
    }

    private void OnStop(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        StopRequested?.Invoke();
    }

    protected override void OnClosed(EventArgs e)
    {
        _timer.Stop();
        StopRequested = null;
        base.OnClosed(e);
    }
}
