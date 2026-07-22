using AgentTrainer.Recorder.Core;

namespace AgentTrainer.Recorder.Core.Tests;

public sealed class RecordingInputFilterTests
{
    private const ulong ShortcutModifiers = QuartzModifierFlags.Control | QuartzModifierFlags.Option | QuartzModifierFlags.Command;

    [Fact]
    public void RecorderShortcutAndItsBufferedModifierReleasesAreRemovedCompletely()
    {
        var filter = new RecordingInputFilter([], 15, ShortcutModifiers);
        Assert.Empty(filter.Process(new InputSample(1, InputEventKind.Flags, KeyCode: 59, Modifiers: QuartzModifierFlags.Control)));
        Assert.Empty(filter.Process(new InputSample(2, InputEventKind.Flags, KeyCode: 58, Modifiers: QuartzModifierFlags.Control | QuartzModifierFlags.Option)));
        Assert.Empty(filter.Process(new InputSample(3, InputEventKind.Flags, KeyCode: 55, Modifiers: ShortcutModifiers)));
        Assert.Empty(filter.Process(new InputSample(4, InputEventKind.Key, KeyCode: 15, Modifiers: ShortcutModifiers, IsDown: true)));
        Assert.Empty(filter.Process(new InputSample(5, InputEventKind.Key, KeyCode: 15, Modifiers: ShortcutModifiers)));
        Assert.Empty(filter.Process(new InputSample(6, InputEventKind.Flags, KeyCode: 55, Modifiers: QuartzModifierFlags.Control | QuartzModifierFlags.Option)));
        Assert.Empty(filter.Process(new InputSample(7, InputEventKind.Flags, KeyCode: 58, Modifiers: QuartzModifierFlags.Control)));
        Assert.Empty(filter.Process(new InputSample(8, InputEventKind.Flags, KeyCode: 59)));
    }

    [Fact]
    public void SyntheticKeyboardVirtualKeyStillRemovesRecorderShortcut()
    {
        var filter = new RecordingInputFilter([], 15, ShortcutModifiers, 0x52);
        Assert.Empty(filter.Process(new InputSample(1, InputEventKind.Flags, KeyCode: 59, Modifiers: QuartzModifierFlags.Control)));
        Assert.Empty(filter.Process(new InputSample(2, InputEventKind.Flags, KeyCode: 58, Modifiers: QuartzModifierFlags.Control | QuartzModifierFlags.Option)));
        Assert.Empty(filter.Process(new InputSample(3, InputEventKind.Flags, KeyCode: 55, Modifiers: ShortcutModifiers)));
        Assert.Empty(filter.Process(new InputSample(4, InputEventKind.Key, KeyCode: 11, Modifiers: ShortcutModifiers,
            IsDown: true, NativeVirtualKey: 0x52)));
        Assert.Empty(filter.Process(new InputSample(5, InputEventKind.Key, KeyCode: 11, Modifiers: ShortcutModifiers,
            NativeVirtualKey: 0x52)));
        Assert.Empty(filter.Process(new InputSample(6, InputEventKind.Flags, KeyCode: 55, Modifiers: QuartzModifierFlags.Control | QuartzModifierFlags.Option)));
        Assert.Empty(filter.Process(new InputSample(7, InputEventKind.Flags, KeyCode: 58, Modifiers: QuartzModifierFlags.Control)));
        Assert.Empty(filter.Process(new InputSample(8, InputEventKind.Flags, KeyCode: 59)));
    }

    [Fact]
    public void ConfiguredAlternativeShortcutIsRemovedWithoutDroppingOrdinaryKeys()
    {
        const ulong modifiers = QuartzModifierFlags.Control | QuartzModifierFlags.Shift;
        var filter = new RecordingInputFilter([], 101, modifiers, 0x78); // Ctrl+Shift+F9
        Assert.Empty(filter.Process(new InputSample(1, InputEventKind.Flags, KeyCode: 59, Modifiers: QuartzModifierFlags.Control)));
        Assert.Empty(filter.Process(new InputSample(2, InputEventKind.Flags, KeyCode: 56, Modifiers: modifiers)));
        Assert.Empty(filter.Process(new InputSample(3, InputEventKind.Key, KeyCode: 101, Modifiers: modifiers,
            IsDown: true, NativeVirtualKey: 0x78)));
        Assert.Empty(filter.Process(new InputSample(4, InputEventKind.Key, KeyCode: 101, Modifiers: modifiers,
            NativeVirtualKey: 0x78)));
        Assert.Empty(filter.Process(new InputSample(5, InputEventKind.Flags, KeyCode: 56, Modifiers: QuartzModifierFlags.Control)));
        Assert.Empty(filter.Process(new InputSample(6, InputEventKind.Flags, KeyCode: 59)));

        var ordinary = Assert.Single(filter.Process(new InputSample(7, InputEventKind.Key, KeyCode: 0, IsDown: true, NativeVirtualKey: 0x41)));
        Assert.Equal((ushort)0, ordinary.KeyCode);
    }

    [Fact]
    public void ReleaseOnlyStartupChordIsRemovedAfterExplicitSuppression()
    {
        var filter = new RecordingInputFilter([], 15, ShortcutModifiers, 0x52);
        filter.SuppressActiveShortcut(ShortcutModifiers);
        Assert.Empty(filter.Process(new InputSample(1, InputEventKind.Key, KeyCode: 11, Modifiers: ShortcutModifiers,
            NativeVirtualKey: 0x52)));
        Assert.Empty(filter.Process(new InputSample(2, InputEventKind.Flags, KeyCode: 55, Modifiers: QuartzModifierFlags.Control | QuartzModifierFlags.Option)));
        Assert.Empty(filter.Process(new InputSample(3, InputEventKind.Flags, KeyCode: 58, Modifiers: QuartzModifierFlags.Control)));
        Assert.Empty(filter.Process(new InputSample(4, InputEventKind.Flags, KeyCode: 59)));

        var ordinary = Assert.Single(filter.Process(new InputSample(5, InputEventKind.Key, KeyCode: 0, IsDown: true)));
        Assert.Equal((ushort)0, ordinary.KeyCode);
    }

    [Fact]
    public void OrdinaryModifierChordRetainsOriginalTimestampsAndOrder()
    {
        var filter = new RecordingInputFilter([], 15, ShortcutModifiers);
        Assert.Empty(filter.Process(new InputSample(10, InputEventKind.Flags, KeyCode: 59, Modifiers: QuartzModifierFlags.Control)));
        var output = filter.Process(new InputSample(11, InputEventKind.Key, KeyCode: 1, Modifiers: QuartzModifierFlags.Control, IsDown: true));
        Assert.Equal(2, output.Count);
        Assert.Equal(10ul, output[0].TimestampNanos);
        Assert.Equal(InputEventKind.Flags, output[0].Kind);
        Assert.Equal(11ul, output[1].TimestampNanos);
        Assert.Equal((ushort)1, output[1].KeyCode);
    }

    [Fact]
    public void BlacklistDropsKeysAndSanitizesModifierFlagsOnEveryEvent()
    {
        var filter = new RecordingInputFilter([13, 58], 15, ShortcutModifiers);
        Assert.Empty(filter.Process(new InputSample(1, InputEventKind.Key, KeyCode: 13, IsDown: true)));
        var sample = Assert.Single(filter.Process(new InputSample(2, InputEventKind.MouseMove,
            Modifiers: QuartzModifierFlags.Shift | QuartzModifierFlags.Option)));
        Assert.Equal(QuartzModifierFlags.Shift, sample.Modifiers);
        Assert.Equal(QuartzModifierFlags.Shift, filter.SanitizeModifiers(QuartzModifierFlags.Shift | QuartzModifierFlags.Option));
    }
}
