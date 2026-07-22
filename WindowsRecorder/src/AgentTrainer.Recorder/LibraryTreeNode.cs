using System.Collections.ObjectModel;
using AgentTrainer.Recorder.Core;

namespace AgentTrainer.Recorder;

internal sealed class LibraryTreeNode
{
    private LibraryTreeNode(RecordingFolder? folder, RecordingItem? recording, string name, string detail)
    {
        Folder = folder;
        Recording = recording;
        Name = name;
        Detail = detail;
    }

    internal RecordingFolder? Folder { get; }
    internal RecordingItem? Recording { get; }
    public string Name { get; }
    public string Detail { get; }
    public bool IsFolder => Folder is not null;
    public bool IsExpanded { get; set; }
    public bool IsSelected { get; set; }
    public ObservableCollection<LibraryTreeNode> Children { get; } = [];

    internal static LibraryTreeNode ForFolder(RecordingFolder folder, IEnumerable<LibraryTreeNode> recordings, bool expanded)
    {
        var children = recordings.ToArray();
        var node = new LibraryTreeNode(folder, null, folder.Name,
            $"{children.Length} recording{(children.Length == 1 ? "" : "s")}") { IsExpanded = expanded };
        foreach (var child in children) node.Children.Add(child);
        return node;
    }

    internal static LibraryTreeNode ForRecording(RecordingItem item, bool selected = false) => new(
        null,
        item,
        item.Manifest.Name,
        $"{FormatDuration(item.Manifest.EffectiveDuration)}  •  {item.Manifest.PixelWidth}×{item.Manifest.PixelHeight}  •  {item.Manifest.CreatedAt.ToLocalTime():g}") { IsSelected = selected };

    private static string FormatDuration(double seconds)
    {
        var value = Math.Max(0, (int)Math.Ceiling(seconds));
        return $"{value / 60}:{value % 60:00}";
    }
}
