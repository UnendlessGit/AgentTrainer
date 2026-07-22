using System.IO.Compression;
using AgentTrainer.Recorder.Core;

namespace AgentTrainer.Recorder.Core.Tests;

public sealed class RecordingLibraryTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"AgentTrainerLibraryTests-{Guid.NewGuid():N}");

    [Fact]
    public void PublishingCreatesANativeCompatiblePackage()
    {
        var library = new RecordingLibrary(Path.Combine(_root, "Library"));
        library.Prepare();
        var folder = Assert.Single(library.ListFolders());
        var id = Guid.NewGuid();
        var stage = library.CreateRecordingStage(id);
        var manifest = RecordingTestData.CreatePackagePayload(stage, id, folder.Id);

        var item = library.PublishRecording(stage, manifest);

        Assert.EndsWith(".atrrecord", item.DirectoryPath, StringComparison.OrdinalIgnoreCase);
        Assert.False(Directory.Exists(stage));
        var validated = RecordingValidator.ValidatePackage(item.DirectoryPath);
        Assert.Equal(3, validated.Events.Count);
        Assert.Equal(id, validated.Manifest.Id);
    }

    [Fact]
    public void ImportAssignsFreshIdsAndPreservesPortableControlData()
    {
        var source = Path.Combine(_root, "Windows capture.atrrecord");
        var sourceID = Guid.NewGuid();
        _ = RecordingTestData.CreatePackage(source, sourceID, null);
        var sourceEvents = File.ReadAllBytes(Path.Combine(source, "events.atrevents"));

        var library = new RecordingLibrary(Path.Combine(_root, "Library"));
        library.Prepare();
        var folder = Assert.Single(library.ListFolders());
        var imported = Assert.Single(library.ImportRecordings([source], folder.Id));

        Assert.NotEqual(sourceID, imported.Id);
        Assert.Equal(folder.Id, imported.Manifest.FolderID);
        Assert.Equal(sourceEvents, File.ReadAllBytes(imported.EventPath));
        Assert.Equal(3, imported.Manifest.EventCount);
        Assert.True(Directory.Exists(source));
    }

    [Fact]
    public void CorruptBatchPublishesNothing()
    {
        var good = Path.Combine(_root, "Good.atrrecord");
        _ = RecordingTestData.CreatePackage(good, Guid.NewGuid(), null);
        var bad = Path.Combine(_root, "Bad.atrrecord");
        _ = RecordingTestData.CreatePackage(bad, Guid.NewGuid(), null);
        File.WriteAllBytes(Path.Combine(bad, "events.atrevents"), [1, 2, 3]);

        var library = new RecordingLibrary(Path.Combine(_root, "Library"));
        library.Prepare();
        var folder = Assert.Single(library.ListFolders());
        Assert.Throws<InvalidDataException>(() => library.ImportRecordings([good, bad], folder.Id));
        Assert.Empty(library.ListRecordings());
    }

    [Fact]
    public void RenameMoveFolderAndExportRemainValidated()
    {
        var library = new RecordingLibrary(Path.Combine(_root, "Library"));
        library.Prepare();
        var firstFolder = Assert.Single(library.ListFolders());
        var secondFolder = library.CreateFolder("Gamepad sessions");
        var source = Path.Combine(_root, "Source.atrrecord");
        _ = RecordingTestData.CreatePackage(source, Guid.NewGuid(), null);
        var item = Assert.Single(library.ImportRecordings([source], firstFolder.Id));

        library.RenameRecording(item.Id, "Renamed capture");
        library.MoveRecording(item.Id, secondFolder.Id);
        var changed = Assert.Single(library.ListRecordings());
        Assert.Equal("Renamed capture", changed.Manifest.Name);
        Assert.Equal(secondFolder.Id, changed.Manifest.FolderID);

        var exported = library.ExportRecording(item.Id, Path.Combine(_root, "Exports"));
        Assert.Equal(item.Id, RecordingValidator.ValidatePackage(exported).Manifest.Id);

        var archive = library.ExportRecordingArchive(item.Id, Path.Combine(_root, "Transfers", "Renamed capture.atrrecord.zip"));
        Assert.True(File.Exists(archive));
        using var zip = ZipFile.OpenRead(archive);
        Assert.Contains(zip.Entries, value => value.FullName.EndsWith(".atrrecord/manifest.json", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(zip.Entries, value => value.FullName.EndsWith(".atrrecord/events.atrevents", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(zip.Entries, value => value.FullName.EndsWith(".atrrecord/capture.mov", StringComparison.OrdinalIgnoreCase));
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }
}

internal static class RecordingTestData
{
    internal static RecordingManifest Manifest(Guid id, Guid? folderID) => new()
    {
        Id = id,
        Name = "Windows Recording",
        CreatedAt = new DateTimeOffset(2026, 7, 19, 12, 34, 56, TimeSpan.Zero),
        HostStartNanos = 9_000_000_000,
        Duration = 2,
        Capture = new CaptureSpec { Kind = CaptureKinds.ScreenRegion, Region = new CodableRect { X = 100, Y = 200, Width = 1280, Height = 720 }, RequestedFPS = 60, ShowsCursor = true },
        GlobalRect = new CodableRect { X = 100, Y = 200, Width = 1280, Height = 720 },
        PixelWidth = 1280,
        PixelHeight = 720,
        DeliveredFPS = 59.94,
        EventCount = 3,
        FolderID = folderID,
        ThumbnailFile = "thumbnail.jpg",
        ExcludedKeyCodes = [15, 55, 58, 59]
    };

    internal static RecordingManifest CreatePackage(string directory, Guid id, Guid? folderID)
    {
        Directory.CreateDirectory(directory);
        var manifest = CreatePackagePayload(directory, id, folderID);
        File.WriteAllBytes(Path.Combine(directory, "manifest.json"), RecordingJson.Serialize(manifest));
        return manifest;
    }

    internal static RecordingManifest CreatePackagePayload(string directory, Guid id, Guid? folderID)
    {
        Directory.CreateDirectory(directory);
        var manifest = Manifest(id, folderID);
        File.WriteAllBytes(Path.Combine(directory, "capture.mov"), "portable-video-placeholder"u8.ToArray());
        File.WriteAllBytes(Path.Combine(directory, "thumbnail.jpg"), "jpeg-placeholder"u8.ToArray());
        using (var writer = new InputEventWriter(Path.Combine(directory, "events.atrevents")))
        {
            writer.Append(new InputSample(manifest.HostStartNanos, InputEventKind.MouseMove, X: 101, Y: 202));
            writer.Append(new InputSample(manifest.HostStartNanos + 500_000_000, InputEventKind.Key, KeyCode: 13, IsDown: true));
            writer.Append(new InputSample(manifest.HostStartNanos + 1_000_000_000, InputEventKind.Key, KeyCode: 13));
        }
        return manifest;
    }
}
