using System.Diagnostics.CodeAnalysis;
using System.Windows;
using System.Windows.Threading;

namespace AgentTrainer.Recorder;

[SuppressMessage("Design", "CA1001:Types that own disposable fields should be disposable", Justification = "WPF owns the App lifetime; OnExit always releases the single-instance mutex.")]
public partial class App : Application
{
    private const string InstanceMutexName = "AgentTrainerRecorder-B4A1B0EE-4B4E-4C8B-BA8D-C402AF84EE55";
    private Mutex? _instanceMutex;
    private bool _ownsInstanceMutex;

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        DispatcherUnhandledException += HandleDispatcherException;
        AppDomain.CurrentDomain.UnhandledException += (_, args) =>
            AppLog.Write("Fatal", args.ExceptionObject?.ToString() ?? "Unknown fatal error");
        TaskScheduler.UnobservedTaskException += (_, args) =>
        {
            AppLog.Write("Background", args.Exception.ToString());
            args.SetObserved();
        };

        try
        {
            if (!NativeRuntimeBootstrap.EnsureAvailable())
            {
                Shutdown();
                return;
            }
            _instanceMutex = new Mutex(initiallyOwned: true, InstanceMutexName, out _ownsInstanceMutex);
            if (!_ownsInstanceMutex)
            {
                MessageBox.Show("AgentTrainer Recorder is already open.", "AgentTrainer Recorder", MessageBoxButton.OK, MessageBoxImage.Information);
                Shutdown();
                return;
            }
            NativeMethods.TryEnablePerMonitorDpi();
            var window = new MainWindow();
            MainWindow = window;
            window.Show();
        }
        catch (Exception error)
        {
            AppLog.Write("Startup", error.ToString());
            MessageBox.Show($"AgentTrainer Recorder could not start.\n\n{error.Message}\n\nDetails were written to the application log.",
                "AgentTrainer Recorder", MessageBoxButton.OK, MessageBoxImage.Error);
            Shutdown(1);
        }
    }

    protected override void OnExit(ExitEventArgs e)
    {
        if (_ownsInstanceMutex) _instanceMutex?.ReleaseMutex();
        _instanceMutex?.Dispose();
        _instanceMutex = null;
        _ownsInstanceMutex = false;
        base.OnExit(e);
    }

    private static void HandleDispatcherException(object sender, DispatcherUnhandledExceptionEventArgs args)
    {
        AppLog.Write("UI", args.Exception.ToString());
        MessageBox.Show(args.Exception.Message, "AgentTrainer Recorder", MessageBoxButton.OK, MessageBoxImage.Error);
        args.Handled = true;
    }
}
