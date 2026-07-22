using System.IO.Compression;

namespace AgentTrainer.Recorder.Core;

/// <summary>
/// Durable Record/Library storage shared by the Windows UI and its tests.
/// Manifests are committed last and imports are validated as a complete batch
/// before anything becomes visible in the library.
/// </summary>
public sealed class RecordingLibrary
{
    private readonly object _gate = new();
    private readonly string _recordingsRoot;
    private readonly string _foldersPath;

    public RecordingLibrary(string rootPath)
    {
        RootPath = Path.GetFullPath(rootPath);
        _recordingsRoot = Path.Combine(RootPath, "Recordings");
        _foldersPath = Path.Combine(RootPath, "recording-folders.json");
    }

    public string RootPath { get; }
    public string RecordingsPath => _recordingsRoot;

    public void Prepare()
    {
        lock (_gate)
        {
            Directory.CreateDirectory(RootPath);
            Directory.CreateDirectory(_recordingsRoot);
            CleanupPrivateStages();
            _ = NormalizeFoldersCore();
        }
    }

    public IReadOnlyList<RecordingFolder> ListFolders()
    {
        lock (_gate) return ReadFoldersCore().OrderBy(value => value.CreatedAt).ToArray();
    }

    public Guid NormalizeFolders()
    {
        lock (_gate) return NormalizeFoldersCore();
    }

    public RecordingFolder CreateFolder(string name)
    {
        var cleanName = CleanName(name, "Folder");
        lock (_gate)
        {
            var folders = ReadFoldersCore();
            if (folders.Any(value => value.Name.Equals(cleanName, StringComparison.OrdinalIgnoreCase)))
                throw new InvalidOperationException("A recording folder with that name already exists.");
            var folder = new RecordingFolder { Id = Guid.NewGuid(), Name = cleanName, CreatedAt = DateTimeOffset.UtcNow };
            folders.Add(folder);
            WriteFoldersCore(folders);
            return folder;
        }
    }

    public void RenameFolder(Guid id, string name)
    {
        var cleanName = CleanName(name, "Folder");
        lock (_gate)
        {
            var folders = ReadFoldersCore();
            var index = folders.FindIndex(value => value.Id == id);
            if (index < 0) throw new InvalidOperationException("The recording folder no longer exists.");
            if (folders.Any(value => value.Id != id && value.Name.Equals(cleanName, StringComparison.OrdinalIgnoreCase)))
                throw new InvalidOperationException("A recording folder with that name already exists.");
            folders[index] = folders[index] with { Name = cleanName };
            WriteFoldersCore(folders);
        }
    }

    public void DeleteFolder(Guid id, bool includingRecordings)
    {
        lock (_gate)
        {
            var folders = ReadFoldersCore();
            if (!folders.Any(value => value.Id == id)) return;
            var remaining = folders.Where(value => value.Id != id).ToList();
            if (remaining.Count == 0)
                remaining.Add(new RecordingFolder { Id = Guid.NewGuid(), Name = "Recordings", CreatedAt = DateTimeOffset.UtcNow });
            foreach (var item in ListRecordingsCore().Where(value => value.Manifest.FolderID == id))
            {
                if (includingRecordings) Directory.Delete(item.DirectoryPath, recursive: true);
                else WriteManifestCore(item.DirectoryPath, item.Manifest with { FolderID = remaining[0].Id });
            }
            WriteFoldersCore(remaining);
        }
    }

    public IReadOnlyList<RecordingItem> ListRecordings()
    {
        lock (_gate) return ListRecordingsCore();
    }

    public string CreateRecordingStage(Guid id)
    {
        lock (_gate)
        {
            Directory.CreateDirectory(_recordingsRoot);
            var path = Path.Combine(_recordingsRoot, $".recording-{id:D}");
            if (Directory.Exists(path)) throw new IOException("A recording with this staging identifier already exists.");
            Directory.CreateDirectory(path);
            return path;
        }
    }

    public RecordingItem PublishRecording(string stagingPath, RecordingManifest manifest)
    {
        lock (_gate)
        {
            var stage = RequirePrivateStage(stagingPath, ".recording-");
            if (!ReadFoldersCore().Any(value => value.Id == manifest.FolderID))
                throw new InvalidOperationException("Choose an existing library folder before recording.");
            if (!manifest.IsStructurallyValid || manifest.HostStartNanos == 0)
                throw new InvalidDataException("The completed recording manifest is invalid.");

            WriteManifestCore(stage, manifest);
            _ = RecordingValidator.ValidatePackage(stage, requireAtrrecordExtension: false);
            var destination = Path.Combine(_recordingsRoot, $"{manifest.Id:D}.atrrecord");
            if (Directory.Exists(destination)) throw new IOException("That recording already exists in the library.");
            Directory.Move(stage, destination);
            return new RecordingItem(manifest, destination);
        }
    }

    public void AbandonStage(string stagingPath)
    {
        lock (_gate)
        {
            var path = RequirePrivateStage(stagingPath, ".recording-");
            if (Directory.Exists(path)) Directory.Delete(path, recursive: true);
        }
    }

    public void RenameRecording(Guid id, string name)
    {
        var cleanName = CleanName(name, "Recording");
        lock (_gate)
        {
            var item = FindRecordingCore(id);
            WriteManifestCore(item.DirectoryPath, item.Manifest with { Name = cleanName });
        }
    }

    public void MoveRecording(Guid id, Guid folderId)
    {
        lock (_gate)
        {
            if (!ReadFoldersCore().Any(value => value.Id == folderId))
                throw new InvalidOperationException("The destination recording folder no longer exists.");
            var item = FindRecordingCore(id);
            WriteManifestCore(item.DirectoryPath, item.Manifest with { FolderID = folderId });
        }
    }

    public void DeleteRecording(Guid id)
    {
        lock (_gate)
        {
            var item = FindRecordingCore(id);
            Directory.Delete(item.DirectoryPath, recursive: true);
        }
    }

    public IReadOnlyList<RecordingItem> ImportRecordings(IEnumerable<string> selectedPaths, Guid folderId)
    {
        lock (_gate)
        {
            if (!ReadFoldersCore().Any(value => value.Id == folderId))
                throw new InvalidOperationException("Choose an existing recording folder before importing.");
            var sources = ExpandRecordingSources(selectedPaths).Distinct(PathComparer).ToArray();
            if (sources.Length == 0) throw new InvalidDataException("Choose one or more .atrrecord folders, or a folder containing them.");
            if (sources.Any(IsInsideLibrary)) throw new InvalidOperationException("Recordings already in this library cannot be imported again in place.");

            // No destination is created until every source passes validation.
            var validated = sources.Select(path => (Path: path, Value: RecordingValidator.ValidatePackage(path))).ToArray();
            var stages = new List<string>();
            var published = new List<string>();
            try
            {
                var results = new List<RecordingItem>(validated.Length);
                foreach (var source in validated)
                {
                    var id = Guid.NewGuid();
                    var stage = Path.Combine(_recordingsRoot, $".import-{id:D}");
                    Directory.CreateDirectory(stage);
                    stages.Add(stage);
                    CopyPayload(source.Path, source.Value.Manifest.VideoFile, stage);
                    CopyPayload(source.Path, source.Value.Manifest.EventFile, stage);
                    var manifest = source.Value.Manifest with { Id = id, FolderID = folderId };
                    if (manifest.ThumbnailFile is { } thumbnail && source.Value.HasThumbnail) CopyPayload(source.Path, thumbnail, stage);
                    else manifest = manifest with { ThumbnailFile = null };
                    WriteManifestCore(stage, manifest);
                    _ = RecordingValidator.ValidatePackage(stage, requireAtrrecordExtension: false);
                    var destination = Path.Combine(_recordingsRoot, $"{id:D}.atrrecord");
                    Directory.Move(stage, destination);
                    stages.Remove(stage);
                    published.Add(destination);
                    results.Add(new RecordingItem(manifest, destination));
                }
                return results;
            }
            catch
            {
                foreach (var path in stages.Concat(published))
                    if (Directory.Exists(path)) Directory.Delete(path, recursive: true);
                throw;
            }
        }
    }

    public string ExportRecording(Guid id, string destinationRoot)
    {
        lock (_gate)
        {
            var item = FindRecordingCore(id);
            _ = RecordingValidator.ValidatePackage(item.DirectoryPath);
            var root = Path.GetFullPath(destinationRoot);
            Directory.CreateDirectory(root);
            var destination = UniqueExportPath(root, SafeExportStem(item.Manifest.Name));
            var stage = destination + ".exporting";
            try
            {
                CopyPackage(item.DirectoryPath, stage);
                _ = RecordingValidator.ValidatePackage(stage, requireAtrrecordExtension: false);
                Directory.Move(stage, destination);
                return destination;
            }
            catch
            {
                if (Directory.Exists(stage)) Directory.Delete(stage, recursive: true);
                throw;
            }
        }
    }

    /// <summary>
    /// Creates the single-file transfer format used by the Windows UI. The ZIP
    /// always contains one complete .atrrecord directory and is committed by an
    /// atomic rename only after the source package passes strict validation.
    /// </summary>
    public string ExportRecordingArchive(Guid id, string destinationPath)
    {
        lock (_gate)
        {
            var item = FindRecordingCore(id);
            _ = RecordingValidator.ValidatePackage(item.DirectoryPath);
            var destination = Path.GetFullPath(destinationPath);
            if (!destination.EndsWith(".atrrecord.zip", StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("Portable recording archives must end in .atrrecord.zip.");
            if (IsInsideLibrary(destination))
                throw new InvalidOperationException("Export recordings outside the active AgentTrainer library.");
            Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
            var stage = destination + $".exporting-{Guid.NewGuid():N}";
            try
            {
                ZipFile.CreateFromDirectory(item.DirectoryPath, stage, CompressionLevel.Optimal, includeBaseDirectory: true);
                if (new FileInfo(stage).Length == 0) throw new InvalidDataException("The portable recording archive is empty.");
                File.Move(stage, destination, overwrite: true);
                return destination;
            }
            catch
            {
                if (File.Exists(stage)) File.Delete(stage);
                throw;
            }
        }
    }

    private Guid NormalizeFoldersCore()
    {
        var folders = ReadFoldersCore();
        var recordings = ListRecordingsCore();
        var valid = folders.Select(value => value.Id).ToHashSet();
        var orphaned = recordings.Where(value => value.Manifest.FolderID is not { } id || !valid.Contains(id)).ToArray();
        if (folders.Count == 0 || orphaned.Length > 0)
        {
            var destination = folders.FirstOrDefault(value => value.Name.Equals("Recordings", StringComparison.OrdinalIgnoreCase));
            if (destination is null)
            {
                destination = new RecordingFolder { Id = Guid.NewGuid(), Name = "Recordings", CreatedAt = DateTimeOffset.UtcNow };
                folders.Add(destination);
            }
            WriteFoldersCore(folders);
            foreach (var recording in orphaned)
                WriteManifestCore(recording.DirectoryPath, recording.Manifest with { FolderID = destination.Id });
            return destination.Id;
        }
        return folders[0].Id;
    }

    private List<RecordingFolder> ReadFoldersCore()
    {
        if (!File.Exists(_foldersPath)) return [];
        try
        {
            var folders = RecordingJson.Deserialize<List<RecordingFolder>>(File.ReadAllBytes(_foldersPath));
            return folders.Where(value => value.Id != Guid.Empty && !string.IsNullOrWhiteSpace(value.Name))
                .GroupBy(value => value.Id).Select(group => group.First()).ToList();
        }
        catch (Exception error) when (error is IOException or System.Text.Json.JsonException or InvalidDataException)
        {
            try
            {
                var recoveryPath = Path.Combine(RootPath, $"recording-folders.corrupt-{DateTimeOffset.UtcNow:yyyyMMdd-HHmmss}.json");
                File.Move(_foldersPath, recoveryPath, overwrite: false);
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
            _ = error;
            return [];
        }
    }

    private void WriteFoldersCore(IEnumerable<RecordingFolder> folders) =>
        AtomicWrite(_foldersPath, RecordingJson.Serialize(folders.OrderBy(value => value.CreatedAt).ToArray()));

    private RecordingItem[] ListRecordingsCore()
    {
        if (!Directory.Exists(_recordingsRoot)) return [];
        var result = new List<RecordingItem>();
        foreach (var directory in Directory.EnumerateDirectories(_recordingsRoot, "*.atrrecord", SearchOption.TopDirectoryOnly))
        {
            try
            {
                RecordingValidator.RequireUnlinkedDirectory(directory);
                var manifestPath = RecordingValidator.ResolveLeaf(directory, "manifest.json");
                RecordingValidator.RequireRegularFile(manifestPath);
                var manifest = RecordingJson.Deserialize<RecordingManifest>(File.ReadAllBytes(manifestPath));
                if (manifest.IsStructurallyValid && manifest.Id != Guid.Empty) result.Add(new RecordingItem(manifest, directory));
            }
            catch (Exception error) when (error is IOException or UnauthorizedAccessException or InvalidDataException or System.Text.Json.JsonException)
            {
                // A partial or corrupt package stays hidden, matching the macOS library scan.
            }
        }
        return result.OrderByDescending(value => value.Manifest.CreatedAt).ToArray();
    }

    private RecordingItem FindRecordingCore(Guid id) =>
        ListRecordingsCore().FirstOrDefault(value => value.Id == id)
        ?? throw new InvalidOperationException("The recording no longer exists.");

    private string RequirePrivateStage(string path, string prefix)
    {
        var fullPath = Path.GetFullPath(path);
        if (!Path.GetDirectoryName(fullPath)!.Equals(_recordingsRoot, PathComparison)
            || !Path.GetFileName(fullPath).StartsWith(prefix, StringComparison.Ordinal))
            throw new InvalidOperationException("The staging directory is outside this recording library.");
        return fullPath;
    }

    private void CleanupPrivateStages()
    {
        foreach (var path in Directory.EnumerateDirectories(_recordingsRoot, ".recording-*", SearchOption.TopDirectoryOnly)
            .Concat(Directory.EnumerateDirectories(_recordingsRoot, ".import-*", SearchOption.TopDirectoryOnly)))
        {
            try { Directory.Delete(path, recursive: true); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }

    private static IEnumerable<string> ExpandRecordingSources(IEnumerable<string> selectedPaths)
    {
        foreach (var selected in selectedPaths)
        {
            var path = Path.GetFullPath(selected);
            if (!Directory.Exists(path)) continue;
            if (Path.GetExtension(path).Equals(".atrrecord", StringComparison.OrdinalIgnoreCase))
            {
                yield return path;
                continue;
            }
            var recordings = Path.Combine(path, "Recordings");
            var root = Directory.Exists(recordings) ? recordings : path;
            foreach (var candidate in Directory.EnumerateDirectories(root, "*.atrrecord", SearchOption.TopDirectoryOnly))
                yield return Path.GetFullPath(candidate);
        }
    }

    private bool IsInsideLibrary(string path)
    {
        var relative = Path.GetRelativePath(_recordingsRoot, path);
        return relative != ".." && !relative.StartsWith($"..{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
            && !Path.IsPathRooted(relative);
    }

    private static void CopyPayload(string sourceRoot, string leafName, string destinationRoot)
    {
        var source = RecordingValidator.ResolveLeaf(sourceRoot, leafName);
        RecordingValidator.RequireRegularFile(source);
        File.Copy(source, Path.Combine(destinationRoot, leafName), overwrite: false);
    }

    private static void CopyPackage(string source, string destination)
    {
        RecordingValidator.RequireUnlinkedDirectory(source);
        Directory.CreateDirectory(destination);
        foreach (var file in Directory.EnumerateFiles(source, "*", SearchOption.TopDirectoryOnly))
        {
            RecordingValidator.RequireRegularFile(file);
            File.Copy(file, Path.Combine(destination, Path.GetFileName(file)), overwrite: false);
        }
    }

    private static void WriteManifestCore(string directory, RecordingManifest manifest) =>
        AtomicWrite(Path.Combine(directory, "manifest.json"), RecordingJson.Serialize(manifest));

    private static void AtomicWrite(string destination, byte[] data)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        var temporary = destination + $".tmp-{Guid.NewGuid():N}";
        try
        {
            using (var stream = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None, 16 * 1024,
                       FileOptions.WriteThrough))
            {
                stream.Write(data);
                stream.Flush(flushToDisk: true);
            }
            File.Move(temporary, destination, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporary)) File.Delete(temporary);
        }
    }

    private static string CleanName(string value, string noun)
    {
        var clean = value.Trim();
        if (clean.Length == 0) throw new ArgumentException($"{noun} names cannot be empty.", nameof(value));
        if (clean.Length > 200) throw new ArgumentException($"{noun} names cannot exceed 200 characters.", nameof(value));
        return clean;
    }

    private static string SafeExportStem(string name)
    {
        var invalid = Path.GetInvalidFileNameChars().ToHashSet();
        var value = new string(name.Trim().Select(character => invalid.Contains(character) ? '_' : character).ToArray()).Trim('.', ' ');
        return string.IsNullOrWhiteSpace(value) ? "Recording" : value[..Math.Min(value.Length, 100)];
    }

    private static string UniqueExportPath(string root, string stem)
    {
        var candidate = Path.Combine(root, $"{stem}.atrrecord");
        for (var suffix = 2; Directory.Exists(candidate) || File.Exists(candidate); suffix++)
            candidate = Path.Combine(root, $"{stem} {suffix}.atrrecord");
        return candidate;
    }

    private static StringComparer PathComparer => OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;
    private static StringComparison PathComparison => OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
}
