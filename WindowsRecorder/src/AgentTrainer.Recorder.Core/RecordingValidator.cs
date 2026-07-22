namespace AgentTrainer.Recorder.Core;

public sealed record ValidatedRecording(
    RecordingManifest Manifest,
    InputEventSummary Events,
    bool HasThumbnail,
    long VideoBytes);

/// <summary>
/// Treats portable recordings as untrusted data. The macOS importer applies
/// the same checks and additionally asks AVFoundation to decode the first
/// video frame before publishing an import.
/// </summary>
public static class RecordingValidator
{
    private const int MaximumManifestBytes = 1024 * 1024;

    public static ValidatedRecording ValidatePackage(string directoryPath, bool requireAtrrecordExtension = true)
    {
        var directory = Path.GetFullPath(directoryPath);
        if (requireAtrrecordExtension && !Path.GetExtension(directory).Equals(".atrrecord", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("A recording package must end in .atrrecord.");
        RequireUnlinkedDirectory(directory);

        var manifestPath = ResolveLeaf(directory, "manifest.json");
        RequireRegularFile(manifestPath);
        var manifestInfo = new FileInfo(manifestPath);
        if (manifestInfo.Length is <= 0 or > MaximumManifestBytes)
            throw new InvalidDataException("The recording manifest is empty or unreasonably large.");

        RecordingManifest manifest;
        try
        {
            manifest = RecordingJson.Deserialize<RecordingManifest>(File.ReadAllBytes(manifestPath));
        }
        catch (Exception error) when (error is IOException or System.Text.Json.JsonException or InvalidDataException)
        {
            throw new InvalidDataException("The AgentTrainer recording manifest cannot be read.", error);
        }

        if (!manifest.IsStructurallyValid || manifest.HostStartNanos == 0)
            throw new InvalidDataException("The AgentTrainer recording manifest is invalid.");
        if (new[] { manifest.VideoFile, manifest.EventFile, manifest.ThumbnailFile }
            .Where(name => name is not null)
            .Distinct(StringComparer.OrdinalIgnoreCase).Count() != (manifest.ThumbnailFile is null ? 2 : 3))
            throw new InvalidDataException("Recording payload filenames must be distinct.");
        if (new[] { manifest.VideoFile, manifest.EventFile, manifest.ThumbnailFile }
            .Any(name => string.Equals(name, "manifest.json", StringComparison.OrdinalIgnoreCase)))
            throw new InvalidDataException("A recording payload may not replace its manifest.");

        var videoPath = ResolveLeaf(directory, manifest.VideoFile);
        var eventPath = ResolveLeaf(directory, manifest.EventFile);
        RequireRegularFile(videoPath);
        RequireRegularFile(eventPath);
        var videoBytes = new FileInfo(videoPath).Length;
        if (videoBytes <= 0) throw new InvalidDataException("The recording video is empty.");

        var events = InputEventReader.Summarize(eventPath, globalRect: manifest.GlobalRect);
        if (events.Count != manifest.EventCount)
            throw new InvalidDataException($"The manifest declares {manifest.EventCount} input events but the stream contains {events.Count}.");
        if (events.First is { } first && first.TimestampNanos < manifest.HostStartNanos)
            throw new InvalidDataException("The input stream starts before the first captured video frame.");
        if (events.Last is { } last)
        {
            var eventDuration = (last.TimestampNanos - manifest.HostStartNanos) / 1_000_000_000.0;
            var tolerance = Math.Max(1.0, 2.0 / Math.Max(1.0, manifest.Capture.RequestedFPS));
            if (eventDuration > manifest.Duration + tolerance)
                throw new InvalidDataException("The input stream extends beyond the recording duration.");
        }

        var hasThumbnail = false;
        if (manifest.ThumbnailFile is { } thumbnailName)
        {
            var thumbnailPath = ResolveLeaf(directory, thumbnailName);
            if (File.Exists(thumbnailPath))
            {
                RequireRegularFile(thumbnailPath);
                hasThumbnail = new FileInfo(thumbnailPath).Length > 0;
            }
        }

        return new ValidatedRecording(manifest, events, hasThumbnail, videoBytes);
    }

    public static string ResolveLeaf(string directoryPath, string name)
    {
        if (!RecordingPath.IsSafeLeafName(name)) throw new InvalidDataException("A recording filename is unsafe.");
        var root = Path.GetFullPath(directoryPath).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        var candidate = Path.GetFullPath(Path.Combine(root, name));
        if (!Path.GetDirectoryName(candidate)!.Equals(root, PathComparison))
            throw new InvalidDataException("A recording filename escapes its package.");
        return candidate;
    }

    public static void RequireUnlinkedDirectory(string path)
    {
        var info = new DirectoryInfo(path);
        if (!info.Exists || (info.Attributes & FileAttributes.ReparsePoint) != 0)
            throw new InvalidDataException("A recording package must be a regular, unlinked directory.");
    }

    public static void RequireRegularFile(string path)
    {
        var info = new FileInfo(path);
        if (!info.Exists || (info.Attributes & (FileAttributes.Directory | FileAttributes.ReparsePoint)) != 0)
            throw new InvalidDataException($"The recording contains a missing, linked, or non-regular file: {info.Name}.");
    }

    private static StringComparison PathComparison => OperatingSystem.IsWindows()
        ? StringComparison.OrdinalIgnoreCase
        : StringComparison.Ordinal;
}
