using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Interop;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;
using AgentTrainer.Recorder.Core;
using Microsoft.Win32;
using ScreenRecorderLib;
using ScreenRecorder = ScreenRecorderLib.Recorder;

namespace AgentTrainer.Recorder;

public partial class MainWindow : Window, IDisposable
{
    private readonly string _applicationRoot = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "AgentTrainer Recorder");
    private readonly RecordingLibrary _library;
    private readonly PreferencesStore _preferencesStore;
    private readonly RawInputService _rawInput = new();
    private readonly GlobalRecordingHotkey _globalHotkey = new();
    private readonly CaptureHudWindow _hud = new();
    private readonly ObservableCollection<KeyChoice> _keyChoices = [];
    private readonly DispatcherTimer _videoTimer;
    private readonly RecordingCoordinator _coordinator;
    private RecorderPreferences _preferences;
    private IReadOnlyList<CaptureSourceItem> _captureSources = [];
    private IReadOnlyList<RecordingFolder> _folders = [];
    private IReadOnlyList<RecordingItem> _recordings = [];
    private IReadOnlyList<LibraryTreeNode> _libraryTreeNodes = [];
    private HwndSource? _windowSource;
    private CancellationTokenSource? _detailCancellation;
    private RecordingItem? _inspectedRecording;
    private bool _isInitializing = true;
    private bool _isUpdatingInspector;
    private bool _isPlaying;
    private bool _cancelledStart;
    private bool _isCapturingHotkey;
    private bool _minimizedForRecording;
    private bool _allowClose;
    private bool _resourcesDisposed;

    public MainWindow()
    {
        _library = new RecordingLibrary(_applicationRoot);
        _library.Prepare();
        _preferencesStore = new PreferencesStore(_applicationRoot);
        _preferences = _preferencesStore.Load();
        _coordinator = new RecordingCoordinator(_library, _rawInput);
        InitializeComponent();

        _videoTimer = new DispatcherTimer(TimeSpan.FromMilliseconds(100), DispatcherPriority.Background, CheckVideoTrimEnd, Dispatcher);
        SetCaptureKindSelection(_preferences.CaptureKind);
        FramesPerSecondCombo.Text = _preferences.FramesPerSecond.ToString(CultureInfo.CurrentCulture);
        ShowCursorCheck.IsChecked = _preferences.ShowsCursor;
        TrimStartText.Text = _preferences.TrimStart.ToString("0.###", CultureInfo.CurrentCulture);
        TrimEndText.Text = _preferences.TrimEnd.ToString("0.###", CultureInfo.CurrentCulture);
        RecordingSortCombo.SelectedIndex = 0;
        PopulateRegion(_preferences.CaptureKind == CaptureKinds.WindowRegion ? _preferences.WindowRegion : _preferences.ScreenRegion);
        PopulateKeyChoices();
        ApplyPreferencesToSettings();
        RefreshLibraryView();
        UpdateRegionVisibility();
        ShowRecordPage(this, new RoutedEventArgs());

        SourceInitialized += OnSourceInitialized;
        Loaded += OnLoaded;
        Closing += OnClosing;
        Closed += OnClosed;
        _coordinator.StatusChanged += OnRecordingStatusChanged;
        _rawInput.StateChanged += OnInputStateChanged;
        _globalHotkey.Pressed += OnGlobalHotkey;
        _hud.StopRequested += OnGlobalHotkey;
        PreviewKeyDown += OnPreviewKeyDown;
        _isInitializing = false;
    }

    private void OnSourceInitialized(object? sender, EventArgs args)
    {
        _ = sender;
        _ = args;
        var source = (HwndSource)PresentationSource.FromVisual(this);
        _windowSource = source;
        _rawInput.Attach(source);
        if (!ScreenRecorder.SetExcludeFromCapture(source.Handle, true))
            AppLog.Write("Capture exclusion", "Windows did not confirm exclusion for the main window.");
        try { _globalHotkey.Attach(source, _preferences.Hotkey); }
        catch (Exception error)
        {
            AppLog.Write("Hotkey", error.ToString());
            ActivityStatus.Text = "Ready — global shortcut unavailable; use the Record button";
        }
    }

    private async void OnLoaded(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        await RefreshSourcesAsync();
    }

    private async Task RefreshSourcesAsync()
    {
        if (_coordinator.State != RecordingState.Idle) return;
        var kind = SelectedCaptureKind;
        ActivityStatus.Text = kind is CaptureKinds.Display or CaptureKinds.ScreenRegion ? "Finding displays…" : "Finding windows…";
        CaptureSourceCombo.IsEnabled = false;
        try
        {
            var sources = await Task.Run(() => kind is CaptureKinds.Display or CaptureKinds.ScreenRegion
                ? CaptureSourceCatalog.GetDisplays()
                : CaptureSourceCatalog.GetWindows());
            if (kind != SelectedCaptureKind) return;
            _captureSources = sources;
            CaptureSourceCombo.ItemsSource = sources;
            CaptureSourceItem? preferred = null;
            if (kind is CaptureKinds.Display or CaptureKinds.ScreenRegion && _preferences.DisplayDeviceName is { } deviceName)
                preferred = sources.FirstOrDefault(value => value.DeviceName?.Equals(deviceName, StringComparison.OrdinalIgnoreCase) == true);
            CaptureSourceCombo.SelectedItem = preferred ?? (sources.Count > 0 ? sources[0] : null);
            if (sources.Count == 0) ActivityStatus.Text = "No compatible capture sources are available";
            else ActivityStatus.Text = $"Ready — {sources.Count} source{(sources.Count == 1 ? "" : "s")} available";
            SetReadyPill();
            if (kind == CaptureKinds.ScreenRegion && _preferences.ScreenRegion is null && CaptureSourceCombo.SelectedItem is CaptureSourceItem source)
                PopulateRegion(source.Bounds);
            if (kind == CaptureKinds.WindowRegion && CaptureSourceCombo.SelectedItem is CaptureSourceItem window && _preferences.WindowRegion is null)
                PopulateRegion(new CodableRect { Width = window.Bounds.Width, Height = window.Bounds.Height });
        }
        catch (Exception error)
        {
            PresentError("Capture sources could not be loaded", error);
        }
        finally
        {
            CaptureSourceCombo.IsEnabled = _coordinator.State == RecordingState.Idle;
        }
    }

    private string SelectedCaptureKind =>
        CaptureWindowButton.IsChecked == true ? CaptureKinds.Window
        : CaptureWindowRegionButton.IsChecked == true ? CaptureKinds.WindowRegion
        : CaptureScreenRegionButton.IsChecked == true ? CaptureKinds.ScreenRegion
        : CaptureKinds.Display;

    private async void CaptureKindChecked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        if (_isInitializing) return;
        UpdateRegionVisibility();
        await RefreshSourcesAsync();
    }

    private void SetCaptureKindSelection(string kind)
    {
        CaptureDisplayButton.IsChecked = kind == CaptureKinds.Display;
        CaptureWindowButton.IsChecked = kind == CaptureKinds.Window;
        CaptureWindowRegionButton.IsChecked = kind == CaptureKinds.WindowRegion;
        CaptureScreenRegionButton.IsChecked = kind == CaptureKinds.ScreenRegion;
        if (CaptureKinds.IsValid(kind)) return;
        CaptureDisplayButton.IsChecked = true;
    }

    private void UpdateRegionVisibility()
    {
        var kind = SelectedCaptureKind;
        RegionPanel.Visibility = kind is CaptureKinds.ScreenRegion or CaptureKinds.WindowRegion ? Visibility.Visible : Visibility.Collapsed;
        DrawRegionButton.Visibility = kind == CaptureKinds.ScreenRegion ? Visibility.Visible : Visibility.Collapsed;
        RegionHelpText.Text = kind == CaptureKinds.WindowRegion
            ? "Window-region coordinates are relative to the selected window."
            : "Screen-region coordinates use global physical pixels.";
    }

    private async void RefreshSourcesClicked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        await RefreshSourcesAsync();
    }

    private void DrawRegionClicked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        if (CaptureSourceCombo.SelectedItem is not CaptureSourceItem source || !source.IsDisplay)
        {
            PresentMessage("Choose a display before drawing a region.", MessageBoxImage.Information);
            return;
        }
        var selector = new RegionSelectorWindow(source.Bounds) { Owner = this };
        if (selector.ShowDialog() == true && selector.Selection is { } selection) PopulateRegion(selection);
    }

    private void PopulateRegion(CodableRect? region)
    {
        if (region is null) return;
        RegionXText.Text = region.X.ToString("0.##", CultureInfo.CurrentCulture);
        RegionYText.Text = region.Y.ToString("0.##", CultureInfo.CurrentCulture);
        RegionWidthText.Text = region.Width.ToString("0.##", CultureInfo.CurrentCulture);
        RegionHeightText.Text = region.Height.ToString("0.##", CultureInfo.CurrentCulture);
    }

    private CodableRect ReadRegion() => new()
    {
        X = ReadFiniteDouble(RegionXText, "Region X"),
        Y = ReadFiniteDouble(RegionYText, "Region Y"),
        Width = ReadFiniteDouble(RegionWidthText, "Region width"),
        Height = ReadFiniteDouble(RegionHeightText, "Region height")
    };

    private void PopulateKeyChoices()
    {
        _keyChoices.Clear();
        for (ushort code = 0; code < 128; code++)
            _keyChoices.Add(new KeyChoice(code, $"{MacKeyMap.Name(code)}  · {code}", _preferences.ExcludedKeyCodes.Contains(code)));
        KeyBlacklistItems.ItemsSource = _keyChoices;
        UpdateExcludedKeysSummary();
    }

    private void KeySearchChanged(object sender, TextChangedEventArgs args)
    {
        _ = sender;
        _ = args;
        var search = KeySearchText.Text.Trim();
        KeyBlacklistItems.ItemsSource = string.IsNullOrEmpty(search)
            ? _keyChoices
            : _keyChoices.Where(value => value.Display.Contains(search, StringComparison.OrdinalIgnoreCase)).ToArray();
    }

    private void BlacklistChanged(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        UpdateExcludedKeysSummary();
        SavePreferencesBestEffort();
    }

    private void UpdateExcludedKeysSummary()
    {
        var names = _keyChoices.Where(value => value.IsExcluded).Select(value => MacKeyMap.Name(value.Code)).ToArray();
        ExcludedKeysSummaryText.Text = names.Length == 0
            ? "No excluded keys."
            : $"Excluded: {string.Join("  ", names)}";
    }

    private async void RecordClicked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        await ToggleRecordingAsync();
    }

    private void OnGlobalHotkey()
    {
        _ = ToggleRecordingAsync(startedByGlobalHotkey: true);
    }

    private async Task ToggleRecordingAsync(bool startedByGlobalHotkey = false)
    {
        switch (_coordinator.State)
        {
            case RecordingState.Starting:
                _cancelledStart = true;
                _coordinator.RequestCancelStart();
                ActivityStatus.Text = "Cancelling recording start…";
                return;
            case RecordingState.Recording:
                if (startedByGlobalHotkey) _coordinator.SuppressActiveHotkeyInput();
                await StopRecordingAsync();
                return;
            case RecordingState.Stopping:
                return;
        }

        try
        {
            _cancelledStart = false;
            var request = BuildRecordingRequest(startedByGlobalHotkey);
            SetCaptureConfigurationEnabled(false);
            await _coordinator.StartAsync(request);
        }
        catch (OperationCanceledException) when (_cancelledStart)
        {
            ActivityStatus.Text = "Recording start cancelled";
        }
        catch (Exception error)
        {
            PresentError("Recording could not start", error);
        }
        finally
        {
            if (_coordinator.State == RecordingState.Idle) SetCaptureConfigurationEnabled(true);
        }
    }

    private RecordingRequest BuildRecordingRequest(bool startedByGlobalHotkey)
    {
        if (CaptureSourceCombo.SelectedItem is not CaptureSourceItem source)
            throw new InvalidOperationException("Choose a display or window to record.");
        if (DestinationFolderCombo.SelectedValue is not Guid folderID)
            throw new InvalidOperationException("Choose or create a destination folder.");
        var fps = ReadFramesPerSecond();
        var trimStart = Math.Max(0, ReadFiniteDouble(TrimStartText, "Trim first"));
        var trimEnd = Math.Max(0, ReadFiniteDouble(TrimEndText, "Trim last"));
        var region = SelectedCaptureKind is CaptureKinds.ScreenRegion or CaptureKinds.WindowRegion ? ReadRegion() : null;
        var plan = CaptureSourceCatalog.CreatePlan(source, SelectedCaptureKind, region, fps, ShowCursorCheck.IsChecked == true);
        var excluded = _keyChoices.Where(value => value.IsExcluded).Select(value => value.Code).ToSortedSet();
        _preferences = CurrentPreferences(folderID, source, region) with { FramesPerSecond = fps, TrimStart = trimStart, TrimEnd = trimEnd };
        _preferencesStore.Save(_preferences);
        return new RecordingRequest(plan, folderID, excluded, trimStart, trimEnd, _preferences.PreferHevc, _preferences.Hotkey, startedByGlobalHotkey);
    }

    private async Task StopRecordingAsync()
    {
        try
        {
            var item = await _coordinator.StopAsync();
            if (item is not null)
            {
                RefreshLibraryView(item.Id);
                if (_preferences.OpenLibraryAfterRecording) ShowLibraryPage(this, new RoutedEventArgs());
            }
        }
        catch (Exception error)
        {
            PresentError("Recording could not be saved", error);
        }
    }

    private void OnRecordingStatusChanged(RecordingStatusUpdate update)
    {
        _ = Dispatcher.BeginInvoke(() =>
        {
            ActivityStatus.Text = update.Message;
            switch (update.State)
            {
                case RecordingState.Starting:
                    StatusPillText.Text = "Starting";
                    StatusPillText.Foreground = (Brush)FindResource("AmberBrush");
                    StatusPillDot.Fill = (Brush)FindResource("AmberBrush");
                    StatusPill.Background = new SolidColorBrush(Color.FromRgb(58, 45, 21));
                    RecordButton.Content = "Cancel Start";
                    TopRecordButton.Content = "■";
                    TopRecordButton.ToolTip = "Cancel recording start";
                    RecordButton.IsEnabled = true;
                    break;
                case RecordingState.Recording:
                    StatusPillText.Text = "Recording";
                    StatusPillText.Foreground = (Brush)FindResource("CoralBrush");
                    StatusPillDot.Fill = (Brush)FindResource("CoralBrush");
                    StatusPill.Background = new SolidColorBrush(Color.FromRgb(62, 24, 25));
                    RecordButton.Content = "Stop & Save";
                    TopRecordButton.Content = "■";
                    TopRecordButton.ToolTip = "Stop and save recording";
                    RecordButton.IsEnabled = true;
                    if (_preferences.ShowCaptureHud) _hud.Begin(update.StartedAt ?? DateTimeOffset.UtcNow, _preferences.Hotkey.DisplayText);
                    if (_preferences.MinimizeWhileRecording && WindowState != WindowState.Minimized)
                    {
                        _minimizedForRecording = true;
                        WindowState = WindowState.Minimized;
                    }
                    break;
                case RecordingState.Stopping:
                    StatusPillText.Text = "Saving";
                    StatusPillText.Foreground = (Brush)FindResource("AmberBrush");
                    StatusPillDot.Fill = (Brush)FindResource("AmberBrush");
                    StatusPill.Background = new SolidColorBrush(Color.FromRgb(58, 45, 21));
                    RecordButton.Content = "Saving…";
                    TopRecordButton.Content = "…";
                    TopRecordButton.ToolTip = "Saving recording";
                    RecordButton.IsEnabled = false;
                    _hud.End();
                    break;
                case RecordingState.Idle:
                    SetReadyPill();
                    RecordButton.Content = "Record";
                    TopRecordButton.Content = "●";
                    TopRecordButton.ToolTip = "Start recording";
                    RecordButton.IsEnabled = true;
                    SetCaptureConfigurationEnabled(true);
                    _hud.End();
                    if (_minimizedForRecording)
                    {
                        _minimizedForRecording = false;
                        WindowState = WindowState.Normal;
                        Activate();
                    }
                    break;
            }
        });
    }

    private void OnInputStateChanged(InputStateSnapshot state)
    {
        if (_coordinator.State != RecordingState.Recording || !_preferences.ShowCaptureHud) return;
        if (Dispatcher.CheckAccess()) _hud.UpdateControls(state);
        else _ = Dispatcher.BeginInvoke(() => _hud.UpdateControls(state));
    }

    private void SetReadyPill()
    {
        StatusPillText.Text = "Local only";
        StatusPillText.Foreground = (Brush)FindResource("GreenBrush");
        StatusPillDot.Fill = (Brush)FindResource("GreenBrush");
        StatusPill.Background = new SolidColorBrush(Color.FromRgb(23, 53, 42));
    }

    private void SetCaptureConfigurationEnabled(bool enabled)
    {
        CaptureTypePanel.IsEnabled = enabled;
        CaptureSourceCombo.IsEnabled = enabled;
        DestinationFolderCombo.IsEnabled = enabled;
        RegionPanel.IsEnabled = enabled;
        FramesPerSecondCombo.IsEnabled = enabled;
        ShowCursorCheck.IsEnabled = enabled;
        TrimStartText.IsEnabled = enabled;
        TrimEndText.IsEnabled = enabled;
        KeyBlacklistItems.IsEnabled = enabled;
    }

    private int ReadFramesPerSecond()
    {
        var text = FramesPerSecondCombo.Text;
        if (int.TryParse(text, NumberStyles.Integer, CultureInfo.CurrentCulture, out var value) && value is >= 1 and <= 240) return value;
        throw new InvalidOperationException("FPS must be a whole number from 1 through 240.");
    }

    private static double ReadFiniteDouble(TextBox field, string name)
    {
        if ((double.TryParse(field.Text, NumberStyles.Float, CultureInfo.CurrentCulture, out var value)
             || double.TryParse(field.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out value))
            && double.IsFinite(value)) return value;
        throw new InvalidOperationException($"{name} must be a finite number.");
    }

    private RecorderPreferences CurrentPreferences(Guid? folderID = null, CaptureSourceItem? source = null, CodableRect? region = null)
    {
        var kind = SelectedCaptureKind;
        return _preferences with
        {
            CaptureKind = kind,
            FramesPerSecond = int.TryParse(FramesPerSecondCombo.Text, out var fps) ? Math.Clamp(fps, 1, 240) : _preferences.FramesPerSecond,
            ShowsCursor = ShowCursorCheck.IsChecked == true,
            TrimStart = TryReadDouble(TrimStartText.Text, _preferences.TrimStart),
            TrimEnd = TryReadDouble(TrimEndText.Text, _preferences.TrimEnd),
            DestinationFolderID = folderID ?? DestinationFolderCombo.SelectedValue as Guid? ?? _preferences.DestinationFolderID,
            DisplayDeviceName = source?.DeviceName ?? (CaptureSourceCombo.SelectedItem as CaptureSourceItem)?.DeviceName ?? _preferences.DisplayDeviceName,
            ScreenRegion = kind == CaptureKinds.ScreenRegion ? region ?? TryReadRegion() : _preferences.ScreenRegion,
            WindowRegion = kind == CaptureKinds.WindowRegion ? region ?? TryReadRegion() : _preferences.WindowRegion,
            ExcludedKeyCodes = _keyChoices.Where(value => value.IsExcluded).Select(value => value.Code).ToSortedSet(),
            Hotkey = _preferences.Hotkey,
            ShowCaptureHud = SettingsShowHudCheck.IsChecked == true,
            OpenLibraryAfterRecording = SettingsOpenLibraryCheck.IsChecked == true,
            MinimizeWhileRecording = SettingsMinimizeCheck.IsChecked == true,
            PreferHevc = SettingsPreferHevcCheck.IsChecked == true
        };
    }

    private CodableRect? TryReadRegion()
    {
        try { return ReadRegion(); }
        catch (InvalidOperationException) { return null; }
    }

    private static double TryReadDouble(string text, double fallback) =>
        double.TryParse(text, NumberStyles.Float, CultureInfo.CurrentCulture, out var value) && double.IsFinite(value) ? value : fallback;

    private void SavePreferencesBestEffort()
    {
        try
        {
            _preferences = CurrentPreferences();
            _preferencesStore.Save(_preferences);
        }
        catch (Exception error) { AppLog.Write("Settings", error.ToString()); }
    }

    private void ApplyPreferencesToSettings()
    {
        SettingsShowHudCheck.IsChecked = _preferences.ShowCaptureHud;
        SettingsOpenLibraryCheck.IsChecked = _preferences.OpenLibraryAfterRecording;
        SettingsMinimizeCheck.IsChecked = _preferences.MinimizeWhileRecording;
        SettingsPreferHevcCheck.IsChecked = _preferences.PreferHevc;
        HotkeyCaptureButton.Content = _preferences.Hotkey.DisplayText;
        TopHotkeyLabel.Text = _preferences.Hotkey.DisplayText;
        LibraryPathText.Text = _library.RecordingsPath;
        HotkeyCaptureButton.IsEnabled = _coordinator.State == RecordingState.Idle;
    }

    private void SettingsChanged(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        if (_isInitializing) return;
        if (_coordinator.State != RecordingState.Idle)
        {
            ApplyPreferencesToSettings();
            PresentMessage("Stop the active recording before changing recording behavior.", MessageBoxImage.Information);
            return;
        }
        SavePreferencesBestEffort();
        if (!_preferences.ShowCaptureHud) _hud.End();
        ActivityStatus.Text = "Settings saved";
    }

    private void HotkeyCaptureClicked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        if (_coordinator.State != RecordingState.Idle)
        {
            PresentMessage("Stop recording before changing the global keybind.", MessageBoxImage.Information);
            return;
        }
        _isCapturingHotkey = true;
        HotkeyCaptureButton.Content = "Press shortcut…";
        HotkeyCaptureButton.Focus();
        ActivityStatus.Text = "Press one or more modifiers plus A–Z, 0–9, F1–F12, Esc, or Space • Esc alone cancels";
    }

    private void OnPreviewKeyDown(object sender, KeyEventArgs args)
    {
        _ = sender;
        if (!_isCapturingHotkey) return;
        args.Handled = true;
        var key = args.Key == Key.System ? args.SystemKey : args.Key;
        var modifiers = Keyboard.Modifiers;
        if (key == Key.Escape && modifiers == ModifierKeys.None)
        {
            CancelHotkeyCapture("Shortcut change cancelled");
            return;
        }
        if (key is Key.LeftCtrl or Key.RightCtrl or Key.LeftAlt or Key.RightAlt or Key.LeftShift or Key.RightShift or Key.LWin or Key.RWin)
        {
            HotkeyCaptureButton.Content = "Now press a key…";
            return;
        }

        var virtualKey = KeyInterop.VirtualKeyFromKey(key);
        var choice = virtualKey is >= 0 and <= ushort.MaxValue ? RecordingHotkeyCatalog.Find((ushort)virtualKey) : null;
        if (choice is null)
        {
            ActivityStatus.Text = "That key cannot be used globally. Choose A–Z, 0–9, F1–F12, Esc, or Space.";
            return;
        }
        var binding = new RecordingHotkeyBinding
        {
            VirtualKey = choice.VirtualKey,
            MacKeyCode = choice.MacKeyCode,
            Control = modifiers.HasFlag(ModifierKeys.Control),
            Alt = modifiers.HasFlag(ModifierKeys.Alt),
            Shift = modifiers.HasFlag(ModifierKeys.Shift),
            Windows = modifiers.HasFlag(ModifierKeys.Windows)
        };
        if (!binding.HasModifier)
        {
            ActivityStatus.Text = "Add Ctrl, Alt, Shift, or Win so ordinary typing never starts a recording.";
            return;
        }

        try
        {
            if (_globalHotkey.IsAttached) _globalHotkey.Update(binding);
            else if (_windowSource is { } source) _globalHotkey.Attach(source, binding);
            _preferences = CurrentPreferences() with { Hotkey = binding };
            _preferencesStore.Save(_preferences);
            _isCapturingHotkey = false;
            ApplyPreferencesToSettings();
            ActivityStatus.Text = $"Global recording shortcut set to {binding.DisplayText}";
        }
        catch (Exception error)
        {
            CancelHotkeyCapture(error.Message);
            PresentError("Shortcut could not be registered", error);
        }
    }

    private void CancelHotkeyCapture(string status)
    {
        _isCapturingHotkey = false;
        HotkeyCaptureButton.Content = _preferences.Hotkey.DisplayText;
        ActivityStatus.Text = status;
    }

    private void OpenLibraryFolderClicked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        OpenFolder(_library.RecordingsPath);
    }

    private void OpenLogsFolderClicked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        Directory.CreateDirectory(AppLog.DirectoryPath);
        OpenFolder(AppLog.DirectoryPath);
    }

    private static void OpenFolder(string path) =>
        Process.Start(new ProcessStartInfo("explorer.exe", $"\"{path}\"") { UseShellExecute = true });

    private void ResetSettingsClicked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        if (_coordinator.State != RecordingState.Idle)
        {
            PresentMessage("Stop recording before resetting settings.", MessageBoxImage.Information);
            return;
        }
        if (MessageBox.Show(this, "Reset the shortcut, capture defaults, key blacklist, and recording behavior? Your recordings are not affected.",
                "Reset settings", MessageBoxButton.YesNo, MessageBoxImage.Question, MessageBoxResult.No) != MessageBoxResult.Yes) return;
        try
        {
            var defaults = new RecorderPreferences();
            if (_globalHotkey.IsAttached) _globalHotkey.Update(defaults.Hotkey);
            else if (_windowSource is { } source) _globalHotkey.Attach(source, defaults.Hotkey);
            _preferences = defaults;
            _preferencesStore.Save(_preferences);
            _isInitializing = true;
            SetCaptureKindSelection(_preferences.CaptureKind);
            FramesPerSecondCombo.Text = _preferences.FramesPerSecond.ToString(CultureInfo.CurrentCulture);
            ShowCursorCheck.IsChecked = _preferences.ShowsCursor;
            TrimStartText.Text = _preferences.TrimStart.ToString("0.###", CultureInfo.CurrentCulture);
            TrimEndText.Text = _preferences.TrimEnd.ToString("0.###", CultureInfo.CurrentCulture);
            PopulateRegion(new CodableRect { X = 0, Y = 0, Width = 1280, Height = 720 });
            PopulateKeyChoices();
            ApplyPreferencesToSettings();
            _isInitializing = false;
            _ = RefreshSourcesAsync();
            ActivityStatus.Text = "Settings reset to defaults";
        }
        catch (Exception error)
        {
            _isInitializing = false;
            PresentError("Settings could not be reset", error);
        }
    }

    private void ShowRecordPage(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        RecordPage.Visibility = Visibility.Visible;
        LibraryPage.Visibility = Visibility.Collapsed;
        SettingsPage.Visibility = Visibility.Collapsed;
        SelectNavigation(RecordNavigationButton, (Brush)FindResource("CoralBrush"), Color.FromRgb(54, 27, 32));
        TopSectionIcon.Text = "●";
        TopSectionTitle.Text = "Record";
        TopSectionIcon.Foreground = TopSectionTitle.Foreground = (Brush)FindResource("CoralBrush");
    }

    private void ShowLibraryPage(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        RecordPage.Visibility = Visibility.Collapsed;
        LibraryPage.Visibility = Visibility.Visible;
        SettingsPage.Visibility = Visibility.Collapsed;
        SelectNavigation(LibraryNavigationButton, (Brush)FindResource("AmberBrush"), Color.FromRgb(55, 43, 22));
        TopSectionIcon.Text = "▣";
        TopSectionTitle.Text = "Library";
        TopSectionIcon.Foreground = TopSectionTitle.Foreground = (Brush)FindResource("AmberBrush");
        RefreshLibraryView(_inspectedRecording?.Id);
    }

    private void ShowSettingsPage(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        RecordPage.Visibility = Visibility.Collapsed;
        LibraryPage.Visibility = Visibility.Collapsed;
        SettingsPage.Visibility = Visibility.Visible;
        SelectNavigation(SettingsNavigationButton, (Brush)FindResource("AmberBrush"), Color.FromRgb(55, 43, 22));
        TopSectionIcon.Text = "⚙";
        TopSectionTitle.Text = "Settings";
        TopSectionIcon.Foreground = TopSectionTitle.Foreground = (Brush)FindResource("AmberBrush");
        ApplyPreferencesToSettings();
    }

    private void SelectNavigation(Button selected, Brush accent, Color fill)
    {
        foreach (var button in new[] { RecordNavigationButton, LibraryNavigationButton, SettingsNavigationButton })
        {
            button.Background = Brushes.Transparent;
            button.BorderBrush = Brushes.Transparent;
            button.Foreground = new SolidColorBrush(Color.FromRgb(184, 192, 205));
        }
        selected.Background = new SolidColorBrush(fill);
        selected.BorderBrush = new SolidColorBrush(Color.FromArgb(110, ((SolidColorBrush)accent).Color.R, ((SolidColorBrush)accent).Color.G, ((SolidColorBrush)accent).Color.B));
        selected.Foreground = accent;
    }

    private void RefreshLibraryView(Guid? selectID = null)
    {
        var previousDestination = DestinationFolderCombo.SelectedValue as Guid? ?? _preferences.DestinationFolderID;
        _library.NormalizeFolders();
        _folders = _library.ListFolders();
        _recordings = _library.ListRecordings();
        DestinationFolderCombo.ItemsSource = _folders;
        RecordingFolderCombo.ItemsSource = _folders;
        var destination = _folders.Any(value => value.Id == previousDestination) ? previousDestination : (_folders.Count > 0 ? _folders[0].Id : null);
        DestinationFolderCombo.SelectedValue = destination;
        ApplyRecordingFilter(selectID ?? _inspectedRecording?.Id);
    }

    private void ApplyRecordingFilter(Guid? selectID = null)
    {
        var expandedFolders = _libraryTreeNodes.Where(value => value.IsExpanded && value.Folder is not null)
            .Select(value => value.Folder!.Id).ToHashSet();
        var search = RecordingSearchText.Text.Trim();
        IEnumerable<RecordingItem> values = _recordings.Where(value =>
            string.IsNullOrEmpty(search) || value.Manifest.Name.Contains(search, StringComparison.CurrentCultureIgnoreCase));
        var sort = (RecordingSortCombo.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "Newest";
        values = sort switch
        {
            "Oldest" => values.OrderBy(value => value.Manifest.CreatedAt),
            "Name" => values.OrderBy(value => value.Manifest.Name, StringComparer.CurrentCultureIgnoreCase),
            "Duration" => values.OrderByDescending(value => value.Manifest.EffectiveDuration),
            _ => values.OrderByDescending(value => value.Manifest.CreatedAt)
        };
        var visible = values.ToArray();
        var selectedID = visible.Any(value => value.Id == selectID) ? selectID : visible.FirstOrDefault()?.Id;
        _libraryTreeNodes = _folders.Select(folder => LibraryTreeNode.ForFolder(
            folder,
            visible.Where(value => value.Manifest.FolderID == folder.Id)
                .Select(value => LibraryTreeNode.ForRecording(value, value.Id == selectedID)),
            expandedFolders.Contains(folder.Id) || !string.IsNullOrEmpty(search) || visible.Any(value => value.Id == selectedID && value.Manifest.FolderID == folder.Id)
        )).ToArray();
        RecordingTree.ItemsSource = _libraryTreeNodes;
        RecordingCountLabel.Text = $"{visible.Length} recording{(visible.Length == 1 ? "" : "s")}";
        if (selectedID is { } id && visible.FirstOrDefault(value => value.Id == id) is { } selection) _ = LoadInspectorAsync(selection);
        else ClearInspector();
    }

    private void RecordingFilterChanged(object sender, EventArgs args)
    {
        _ = sender;
        _ = args;
        if (!_isInitializing) ApplyRecordingFilter();
    }

    private async void RecordingTreeSelectionChanged(object sender, RoutedPropertyChangedEventArgs<object> args)
    {
        _ = sender;
        _ = args;
        if (RecordingTree.SelectedItem is not LibraryTreeNode node) return;
        if (node.Recording is { } item) await LoadInspectorAsync(item);
        else if (node.Folder is { } folder) DestinationFolderCombo.SelectedValue = folder.Id;
    }

    private async Task LoadInspectorAsync(RecordingItem item)
    {
        _detailCancellation?.Cancel();
        _detailCancellation?.Dispose();
        _detailCancellation = new CancellationTokenSource();
        var cancellationToken = _detailCancellation.Token;
        _inspectedRecording = item;
        _isUpdatingInspector = true;
        InspectorContent.IsEnabled = true;
        RecordingNameText.Text = item.Manifest.Name;
        DurationBadge.Text = FormatDuration(item.Manifest.EffectiveDuration);
        ResolutionBadge.Text = $"{item.Manifest.PixelWidth}×{item.Manifest.PixelHeight}";
        FpsBadge.Text = $"{item.Manifest.DeliveredFPS:0.#} FPS";
        InputsBadge.Text = $"{item.Manifest.EventCount:N0} inputs";
        CaptureDetailText.Text = $"{item.Manifest.Capture.Kind} • captured {item.Manifest.CreatedAt.ToLocalTime():g}";
        TrimDetailText.Text = item.Manifest.TrimStart > 0 || item.Manifest.EffectiveEnd < item.Manifest.Duration
            ? $"Training range {item.Manifest.TrimStart:0.##}s – {item.Manifest.EffectiveEnd:0.##}s"
            : "Full recording is used for training";
        RecordingFolderCombo.SelectedValue = item.Manifest.FolderID;
        _isUpdatingInspector = false;
        InspectorVideo.Stop();
        InspectorVideo.Source = new Uri(item.VideoPath, UriKind.Absolute);
        VideoPlaceholder.Visibility = Visibility.Collapsed;
        PlayPauseButton.Content = "Play";
        _isPlaying = false;
        RecordedKeysText.Text = "Loading input details…";
        InputCountsText.Text = "";
        MouseDiagnosticsText.Text = "";
        TimelineList.ItemsSource = null;
        try
        {
            var summary = await Task.Run(() => InputEventReader.Summarize(item.EventPath, 80, item.Manifest.GlobalRect), cancellationToken);
            if (cancellationToken.IsCancellationRequested || _inspectedRecording?.Id != item.Id) return;
            RecordedKeysText.Text = summary.UsedKeyCodes.Count == 0
                ? "No keyboard keys recorded"
                : string.Join("  ", summary.UsedKeyCodes.Select(MacKeyMap.Name));
            InputCountsText.Text = $"{summary.KeyEventCount:N0} key events • {summary.MouseEventCount:N0} pointer events";
            MouseDiagnosticsText.Text = summary.Mouse.MoveEventCount == 0
                ? "No pointer movement samples"
                : $"{(summary.Mouse.IsGameCamera ? "Game camera detected" : "Moving cursor detected")} • raw delta active {summary.Mouse.NonzeroDeltaFraction * 100:0.0}% • mean |Δ| {summary.Mouse.MeanActiveDeltaMagnitude:0.##} px • max {summary.Mouse.MaximumDeltaMagnitude:0.#} px";
            TimelineList.ItemsSource = summary.Preview.Select(value => TimelineLabel(value, item.Manifest)).ToArray();
        }
        catch (Exception error) when (error is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            if (!cancellationToken.IsCancellationRequested)
            {
                RecordedKeysText.Text = "Input details are unavailable";
                AppLog.Write("Inspector", error.ToString());
            }
        }
    }

    private void ClearInspector()
    {
        _detailCancellation?.Cancel();
        _inspectedRecording = null;
        InspectorContent.IsEnabled = false;
        InspectorVideo.Stop();
        InspectorVideo.Source = null;
        VideoPlaceholder.Visibility = Visibility.Visible;
        VideoPlaceholder.Text = "Select a recording";
    }

    private void InspectorMediaOpened(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        if (_inspectedRecording is { } item) InspectorVideo.Position = TimeSpan.FromSeconds(item.Manifest.TrimStart);
    }

    private void PlayPauseClicked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        if (_inspectedRecording is null) return;
        if (_isPlaying)
        {
            InspectorVideo.Pause();
            _videoTimer.Stop();
            PlayPauseButton.Content = "Play";
        }
        else
        {
            InspectorVideo.Play();
            _videoTimer.Start();
            PlayPauseButton.Content = "Pause";
        }
        _isPlaying = !_isPlaying;
    }

    private void RestartVideoClicked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        if (_inspectedRecording is not { } item) return;
        InspectorVideo.Position = TimeSpan.FromSeconds(item.Manifest.TrimStart);
        InspectorVideo.Play();
        _isPlaying = true;
        _videoTimer.Start();
        PlayPauseButton.Content = "Pause";
    }

    private void InspectorMediaEnded(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        ResetVideoToTrimStart();
    }

    private void CheckVideoTrimEnd(object? sender, EventArgs args)
    {
        _ = sender;
        _ = args;
        if (_inspectedRecording is { } item && InspectorVideo.Position.TotalSeconds >= item.Manifest.EffectiveEnd)
            ResetVideoToTrimStart();
    }

    private void ResetVideoToTrimStart()
    {
        if (_inspectedRecording is { } item) InspectorVideo.Position = TimeSpan.FromSeconds(item.Manifest.TrimStart);
        InspectorVideo.Pause();
        _videoTimer.Stop();
        _isPlaying = false;
        PlayPauseButton.Content = "Play";
    }

    private void CreateFolderClicked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        var name = TextPromptWindow.Show(this, "New recording folder", "Folder name");
        if (name is null) return;
        try
        {
            var folder = _library.CreateFolder(name);
            RefreshLibraryView();
            DestinationFolderCombo.SelectedValue = folder.Id;
        }
        catch (Exception error) { PresentError("Folder could not be created", error); }
    }

    private void CreateFolderInlineClicked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        var name = NewFolderText.Text.Trim();
        if (name.Length == 0) { PresentMessage("Enter a folder name first.", MessageBoxImage.Information); return; }
        try
        {
            var folder = _library.CreateFolder(name);
            NewFolderText.Clear();
            DestinationFolderCombo.SelectedValue = folder.Id;
            RefreshLibraryView();
            ActivityStatus.Text = $"Created {folder.Name}";
        }
        catch (Exception error) { PresentError("Folder could not be created", error); }
    }

    private RecordingFolder? SelectedLibraryFolder()
    {
        if (RecordingTree.SelectedItem is LibraryTreeNode { Folder: { } folder }) return folder;
        if (RecordingTree.SelectedItem is LibraryTreeNode { Recording: { } recording })
            return _folders.FirstOrDefault(value => value.Id == recording.Manifest.FolderID);
        var id = DestinationFolderCombo.SelectedValue as Guid?;
        return id is null ? null : _folders.FirstOrDefault(value => value.Id == id);
    }

    private void RenameFolderClicked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        var folder = SelectedLibraryFolder();
        if (folder is null) { PresentMessage("Choose a specific folder first.", MessageBoxImage.Information); return; }
        var name = TextPromptWindow.Show(this, "Rename recording folder", "Folder name", folder.Name);
        if (name is null) return;
        try { _library.RenameFolder(folder.Id, name); RefreshLibraryView(); }
        catch (Exception error) { PresentError("Folder could not be renamed", error); }
    }

    private void DeleteFolderClicked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        var folder = SelectedLibraryFolder();
        if (folder is null) { PresentMessage("Choose a specific folder first.", MessageBoxImage.Information); return; }
        if (MessageBox.Show(this, $"Delete “{folder.Name}” and every recording inside it?", "Delete folder",
                MessageBoxButton.YesNo, MessageBoxImage.Warning, MessageBoxResult.No) != MessageBoxResult.Yes) return;
        try { _library.DeleteFolder(folder.Id, includingRecordings: true); RefreshLibraryView(); }
        catch (Exception error) { PresentError("Folder could not be deleted", error); }
    }

    private async void ImportClicked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        if (_coordinator.State != RecordingState.Idle) { PresentMessage("Stop recording before importing.", MessageBoxImage.Information); return; }
        var dialog = new OpenFolderDialog
        {
            Title = "Choose .atrrecord folders or a folder containing recordings",
            Multiselect = true
        };
        if (dialog.ShowDialog(this) != true) return;
        var folderID = DestinationFolderCombo.SelectedValue as Guid? ?? (_folders.Count > 0 ? _folders[0].Id : null);
        if (folderID is null) { PresentMessage("Create a destination folder first.", MessageBoxImage.Information); return; }
        try
        {
            ActivityStatus.Text = "Validating and importing recordings…";
            var imported = await Task.Run(() => _library.ImportRecordings(dialog.FolderNames, folderID.Value));
            RefreshLibraryView(imported.Count > 0 ? imported[0].Id : null);
            ActivityStatus.Text = $"Imported {imported.Count} recording{(imported.Count == 1 ? "" : "s")}";
        }
        catch (Exception error) { PresentError("Recordings could not be imported", error); }
    }

    private async void ExportClicked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        if (_inspectedRecording is not { } item) { PresentMessage("Select a recording to export.", MessageBoxImage.Information); return; }
        var dialog = new SaveFileDialog
        {
            Title = "Export portable AgentTrainer recording",
            Filter = "AgentTrainer recording archive (*.atrrecord.zip)|*.atrrecord.zip",
            DefaultExt = ".atrrecord.zip",
            AddExtension = true,
            OverwritePrompt = true,
            FileName = $"{SafeTransferName(item.Manifest.Name)}.atrrecord.zip"
        };
        if (dialog.ShowDialog(this) != true) return;
        try
        {
            ActivityStatus.Text = "Validating and exporting recording…";
            var path = await Task.Run(() => _library.ExportRecordingArchive(item.Id, dialog.FileName));
            ActivityStatus.Text = $"Exported {Path.GetFileName(path)} — ready to import on Mac";
            if (MessageBox.Show(this, "One portable .atrrecord.zip file was created. On your Mac, open Library and click Import.\n\nOpen the exported file's location now?", "Export complete",
                    MessageBoxButton.YesNo, MessageBoxImage.Information) == MessageBoxResult.Yes) RevealPath(path);
        }
        catch (Exception error) { PresentError("Recording could not be exported", error); }
    }

    private void RefreshLibraryClicked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        RefreshLibraryView();
        ActivityStatus.Text = "Library refreshed";
    }

    private void RenameRecordingClicked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        if (_inspectedRecording is not { } item) return;
        try
        {
            _library.RenameRecording(item.Id, RecordingNameText.Text);
            RefreshLibraryView(item.Id);
            ActivityStatus.Text = "Recording renamed";
        }
        catch (Exception error) { PresentError("Recording could not be renamed", error); }
    }

    private void MoveRecordingFolderChanged(object sender, SelectionChangedEventArgs args)
    {
        _ = sender;
        _ = args;
        if (_isUpdatingInspector || _inspectedRecording is not { } item || RecordingFolderCombo.SelectedValue is not Guid folderID) return;
        try
        {
            _library.MoveRecording(item.Id, folderID);
            RefreshLibraryView(item.Id);
            ActivityStatus.Text = "Recording moved";
        }
        catch (Exception error) { PresentError("Recording could not be moved", error); }
    }

    private void DeleteRecordingClicked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        if (_inspectedRecording is not { } item) return;
        if (MessageBox.Show(this, $"Permanently delete “{item.Manifest.Name}”, its video, and input data?", "Delete recording",
                MessageBoxButton.YesNo, MessageBoxImage.Warning, MessageBoxResult.No) != MessageBoxResult.Yes) return;
        try
        {
            InspectorVideo.Stop();
            _library.DeleteRecording(item.Id);
            _inspectedRecording = null;
            RefreshLibraryView();
            ActivityStatus.Text = "Recording deleted";
        }
        catch (Exception error) { PresentError("Recording could not be deleted", error); }
    }

    private void RevealRecordingClicked(object sender, RoutedEventArgs args)
    {
        _ = sender;
        _ = args;
        if (_inspectedRecording is { } item) RevealPath(item.DirectoryPath);
    }

    private static void RevealPath(string path)
    {
        Process.Start(new ProcessStartInfo("explorer.exe", $"/select,\"{path}\"") { UseShellExecute = true });
    }

    private static string SafeTransferName(string value)
    {
        var invalid = Path.GetInvalidFileNameChars().ToHashSet();
        var clean = new string(value.Trim().Select(character => invalid.Contains(character) ? '_' : character).ToArray()).Trim('.', ' ');
        return string.IsNullOrWhiteSpace(clean) ? "Recording" : clean[..Math.Min(clean.Length, 100)];
    }

    private static string TimelineLabel(InputSample value, RecordingManifest manifest)
    {
        var time = value.TimestampNanos >= manifest.HostStartNanos
            ? (value.TimestampNanos - manifest.HostStartNanos) / 1_000_000_000.0
            : 0;
        var label = value.Kind switch
        {
            InputEventKind.Key => $"{MacKeyMap.Name(value.KeyCode)} {(value.IsDown ? "down" : "up")}",
            InputEventKind.MouseButton => $"Mouse {value.Button + 1} {(value.IsDown ? "down" : "up")}",
            InputEventKind.MouseMove => $"Mouse Δ{value.DeltaX:0}, {value.DeltaY:0}",
            InputEventKind.Scroll => $"Scroll {value.ScrollX:0.##}, {value.ScrollY:0.##}",
            InputEventKind.Flags => "Modifiers",
            _ => value.Kind.ToString()
        };
        return $"{time,8:0.000}s   {label}";
    }

    private static string FormatDuration(double seconds)
    {
        var value = Math.Max(0, (int)Math.Ceiling(seconds));
        return $"{value / 60}:{value % 60:00}";
    }

    private void PresentError(string title, Exception error)
    {
        AppLog.Write(title, error.ToString());
        ActivityStatus.Text = error.Message;
        var detail = error is FileNotFoundException or BadImageFormatException
            ? $"{error.Message}\n\nInstall the Microsoft Visual C++ 2015–2022 x64 runtime. Windows N/KN editions also need the Media Feature Pack."
            : error.Message;
        MessageBox.Show(this, detail, title, MessageBoxButton.OK, MessageBoxImage.Error);
    }

    private void PresentMessage(string message, MessageBoxImage image) =>
        MessageBox.Show(this, message, "AgentTrainer Recorder", MessageBoxButton.OK, image);

    private async void OnClosing(object? sender, CancelEventArgs args)
    {
        _ = sender;
        if (_allowClose || _coordinator.State == RecordingState.Idle) return;
        args.Cancel = true;
        if (MessageBox.Show(this, "Stop and save the active recording before closing?", "Recording in progress",
                MessageBoxButton.YesNo, MessageBoxImage.Question, MessageBoxResult.Yes) != MessageBoxResult.Yes) return;
        if (_coordinator.State == RecordingState.Starting)
        {
            _cancelledStart = true;
            _coordinator.RequestCancelStart();
            while (_coordinator.State != RecordingState.Idle) await Task.Delay(50);
        }
        else if (_coordinator.State == RecordingState.Recording) await StopRecordingAsync();
        while (_coordinator.State != RecordingState.Idle) await Task.Delay(50);
        _allowClose = true;
        Close();
    }

    private void OnClosed(object? sender, EventArgs args)
    {
        _ = sender;
        _ = args;
        Dispose();
    }

    public void Dispose()
    {
        if (_resourcesDisposed) return;
        _resourcesDisposed = true;
        SavePreferencesBestEffort();
        _detailCancellation?.Cancel();
        _detailCancellation?.Dispose();
        _videoTimer.Stop();
        PreviewKeyDown -= OnPreviewKeyDown;
        _coordinator.Dispose();
        _globalHotkey.Dispose();
        _rawInput.Dispose();
        _hud.Close();
        GC.SuppressFinalize(this);
    }

    private sealed class KeyChoice(ushort code, string display, bool isExcluded)
    {
        public ushort Code { get; } = code;
        public string Display { get; } = display;
        public bool IsExcluded { get; set; } = isExcluded;
    }

}

internal static class EnumerableExtensions
{
    internal static SortedSet<T> ToSortedSet<T>(this IEnumerable<T> values) => new(values);
}
