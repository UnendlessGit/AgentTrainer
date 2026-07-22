using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Windows;

namespace AgentTrainer.Recorder;

/// <summary>
/// Keeps the portable build usable on clean Windows installations. The mixed
/// native ScreenRecorderLib assembly needs the Microsoft VC++ x64 runtime even
/// though the application itself is published self-contained.
/// </summary>
internal static class NativeRuntimeBootstrap
{
    private static readonly string[] RequiredLibraries =
    [
        "VCRUNTIME140.dll",
        "VCRUNTIME140_1.dll",
        "MSVCP140.dll"
    ];

    internal static bool EnsureAvailable()
    {
        if (CanLoadRequiredLibraries()) return true;

        var installer = Path.Combine(AppContext.BaseDirectory, "VC_redist.x64.exe");
        if (!File.Exists(installer))
        {
            MessageBox.Show(
                "Recording requires the Microsoft Visual C++ 2015–2022 x64 runtime. " +
                "VC_redist.x64.exe was not found next to AgentTrainer Recorder. Re-extract the complete portable package or use the installer.",
                "AgentTrainer Recorder prerequisite",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
            return false;
        }

        var choice = MessageBox.Show(
            "AgentTrainer Recorder needs Microsoft's Visual C++ 2015–2022 x64 runtime for hardware capture.\n\n" +
            "Install the bundled Microsoft-signed runtime now? The recorder will restart when installation finishes.",
            "Install recording prerequisite",
            MessageBoxButton.YesNo,
            MessageBoxImage.Information);
        if (choice != MessageBoxResult.Yes) return false;

        try
        {
            using var process = Process.Start(new ProcessStartInfo
            {
                FileName = installer,
                Arguments = "/install /passive /norestart",
                UseShellExecute = true,
                Verb = "runas",
                WorkingDirectory = AppContext.BaseDirectory
            });
            if (process is null) throw new InvalidOperationException("Windows did not start the Microsoft runtime installer.");
            process.WaitForExit();
            if (process.ExitCode is not (0 or 1638 or 3010))
                throw new InvalidOperationException($"The Microsoft runtime installer exited with code {process.ExitCode}.");

            var executable = Environment.ProcessPath
                ?? throw new InvalidOperationException("The recorder executable path is unavailable.");
            _ = Process.Start(new ProcessStartInfo
            {
                FileName = executable,
                UseShellExecute = true,
                WorkingDirectory = AppContext.BaseDirectory
            });
            return false;
        }
        catch (Exception error)
        {
            AppLog.Write("VC++ runtime", error.ToString());
            MessageBox.Show(
                $"The Microsoft runtime could not be installed.\n\n{error.Message}\n\n" +
                "Run VC_redist.x64.exe next to the recorder, then open AgentTrainer Recorder again.",
                "AgentTrainer Recorder prerequisite",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
            return false;
        }
    }

    private static bool CanLoadRequiredLibraries()
    {
        foreach (var library in RequiredLibraries)
        {
            if (!NativeLibrary.TryLoad(library, out var handle)) return false;
            NativeLibrary.Free(handle);
        }
        return true;
    }
}
