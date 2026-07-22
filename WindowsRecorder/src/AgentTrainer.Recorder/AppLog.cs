using System.Globalization;
using System.Text;

namespace AgentTrainer.Recorder;

internal static class AppLog
{
    private static readonly object Gate = new();
    internal static string DirectoryPath { get; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "AgentTrainer Recorder", "Logs");
    internal static string CurrentPath => Path.Combine(DirectoryPath, $"recorder-{DateTime.UtcNow:yyyy-MM-dd}.log");

    internal static void Write(string category, string message)
    {
        try
        {
            lock (Gate)
            {
                Directory.CreateDirectory(DirectoryPath);
                var line = string.Create(CultureInfo.InvariantCulture, $"{DateTimeOffset.UtcNow:O} [{category}] {message}{Environment.NewLine}");
                File.AppendAllText(CurrentPath, line, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
            }
        }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }
}
