using System.Globalization;
using AgentTrainer.Recorder.Core;

namespace AgentTrainer.Recorder;

internal enum RecordingState
{
    Idle,
    Starting,
    Recording,
    Stopping
}

internal sealed record RecordingRequest(
    CapturePlan Plan,
    Guid DestinationFolderID,
    SortedSet<ushort> ExcludedKeyCodes,
    double TrimStart,
    double TrimEnd,
    bool PreferHevc,
    RecordingHotkeyBinding Hotkey,
    bool StartedByGlobalHotkey);

internal sealed record RecordingStatusUpdate(
    RecordingState State,
    string Message,
    DateTimeOffset? StartedAt = null,
    RecordingItem? SavedItem = null);

internal sealed class RecordingCoordinator : IDisposable
{
    private readonly RecordingLibrary _library;
    private readonly RawInputService _rawInput;
    private readonly WindowsScreenRecorder _screenRecorder = new();
    private readonly SemaphoreSlim _lifecycle = new(1, 1);
    private ActiveSession? _active;
    private CancellationTokenSource? _startCancellation;
    private RecordingState _state;
    private bool _disposed;

    internal RecordingCoordinator(RecordingLibrary library, RawInputService rawInput)
    {
        _library = library;
        _rawInput = rawInput;
    }

    internal event Action<RecordingStatusUpdate>? StatusChanged;
    internal RecordingState State => _state;

    internal async Task StartAsync(RecordingRequest request)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        await _lifecycle.WaitAsync().ConfigureAwait(false);
        try
        {
            if (_state != RecordingState.Idle) throw new InvalidOperationException("Recording is already active.");
            SetState(RecordingState.Starting, "Starting capture…");
            var id = Guid.NewGuid();
            var stage = _library.CreateRecordingStage(id);
            var eventPath = Path.Combine(stage, "events.atrevents");
            var temporaryVideo = Path.Combine(stage, "capture.work.mp4");
            var finalVideo = Path.Combine(stage, "capture.mov");
            var thumbnail = Path.Combine(stage, "thumbnail.jpg");
            var writer = new InputEventWriter(eventPath);
            var sink = new RecordingInputSink(writer, request.ExcludedKeyCodes, request.Hotkey, request.StartedByGlobalHotkey);
            if (request.StartedByGlobalHotkey) sink.SuppressActiveHotkey(_rawInput.CurrentModifiers);
            var startedAt = DateTimeOffset.UtcNow;
            var session = new ActiveSession(id, stage, request, writer, sink, startedAt);
            _active = session;
            _rawInput.SampleReceived += sink.Accept;
            _startCancellation = new CancellationTokenSource();

            try
            {
                await _screenRecorder.StartAsync(request.Plan, temporaryVideo, finalVideo, thumbnail, request.PreferHevc,
                    nanos => sink.StartAtFirstFrame(nanos, _rawInput.CurrentPointer, _rawInput.CurrentModifiers),
                    _startCancellation.Token).ConfigureAwait(false);
                _startCancellation.Dispose();
                _startCancellation = null;
                SetState(RecordingState.Recording, $"Recording — {request.Hotkey.DisplayText} stops and saves", startedAt);
            }
            catch
            {
                _rawInput.SampleReceived -= sink.Accept;
                _ = writer.Finish();
                await _screenRecorder.CancelAsync().ConfigureAwait(false);
                _library.AbandonStage(stage);
                _active = null;
                _startCancellation?.Dispose();
                _startCancellation = null;
                SetState(RecordingState.Idle, "Recording start cancelled");
                throw;
            }
        }
        finally
        {
            _lifecycle.Release();
        }
    }

    internal void RequestCancelStart()
    {
        if (_state == RecordingState.Starting) _startCancellation?.Cancel();
    }

    internal void SuppressActiveHotkeyInput()
    {
        if (_active is { } session) session.Sink.SuppressActiveHotkey(_rawInput.CurrentModifiers);
    }

    internal async Task<RecordingItem?> StopAsync()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        await _lifecycle.WaitAsync().ConfigureAwait(false);
        try
        {
            if (_state == RecordingState.Starting)
            {
                RequestCancelStart();
                return null;
            }
            if (_state != RecordingState.Recording || _active is not { } session) return null;
            SetState(RecordingState.Stopping, "Finishing video and validating package…", session.StartedAt);
            _rawInput.SampleReceived -= session.Sink.Accept;
            Exception? eventError = null;
            var eventCount = 0;
            try { eventCount = session.Writer.Finish(); }
            catch (Exception error) { eventError = error; }

            try
            {
                var capture = await _screenRecorder.StopAsync().ConfigureAwait(false);
                if (eventError is not null) throw new InvalidDataException("The recorded input stream could not be finalized.", eventError);
                var hostStart = session.Sink.HostStart != 0 ? session.Sink.HostStart : capture.FirstFrameHostNanos;
                if (hostStart == 0) throw new InvalidDataException("No complete screen frame was received.");
                var inputDuration = session.Sink.LastEvent >= hostStart
                    ? (session.Sink.LastEvent - hostStart) / 1_000_000_000.0
                    : 0;
                var duration = Math.Max(capture.Duration, inputDuration);
                var trimStart = Math.Min(duration, Math.Max(0, session.Request.TrimStart));
                var trimEnd = Math.Max(trimStart, duration - Math.Max(0, session.Request.TrimEnd));
                if (trimEnd <= trimStart)
                {
                    trimStart = 0;
                    trimEnd = duration;
                }

                var manifest = new RecordingManifest
                {
                    Id = session.ID,
                    Name = $"Recording {session.StartedAt.ToLocalTime().ToString("g", CultureInfo.CurrentCulture)}",
                    CreatedAt = session.StartedAt,
                    HostStartNanos = hostStart,
                    Duration = duration,
                    Capture = session.Request.Plan.Spec,
                    GlobalRect = session.Request.Plan.GlobalRect,
                    PixelWidth = capture.Width,
                    PixelHeight = capture.Height,
                    DeliveredFPS = capture.DeliveredFPS,
                    EventCount = eventCount,
                    TrimStart = trimStart,
                    TrimEnd = trimEnd,
                    FolderID = session.Request.DestinationFolderID,
                    ThumbnailFile = capture.HasThumbnail ? "thumbnail.jpg" : null,
                    ExcludedKeyCodes = session.Request.ExcludedKeyCodes.Count == 0
                        ? null
                        : new SortedSet<ushort>(session.Request.ExcludedKeyCodes)
                };
                var item = _library.PublishRecording(session.StagePath, manifest);
                AppLog.Write("Recording", $"Saved {item.Id:D}: {eventCount} inputs, {capture.Width}x{capture.Height}, {duration:F3}s, {capture.Codec}.");
                _active = null;
                SetState(RecordingState.Idle, "Recording saved", savedItem: item);
                return item;
            }
            catch
            {
                await _screenRecorder.CancelAsync().ConfigureAwait(false);
                if (Directory.Exists(session.StagePath)) _library.AbandonStage(session.StagePath);
                _active = null;
                SetState(RecordingState.Idle, "Recording was not saved");
                throw;
            }
        }
        finally
        {
            _lifecycle.Release();
        }
    }

    private void SetState(RecordingState state, string message, DateTimeOffset? startedAt = null, RecordingItem? savedItem = null)
    {
        _state = state;
        StatusChanged?.Invoke(new RecordingStatusUpdate(state, message, startedAt, savedItem));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        RequestCancelStart();
        if (_active is { } session) _rawInput.SampleReceived -= session.Sink.Accept;
        try { _screenRecorder.Dispose(); }
        catch (Exception error) { AppLog.Write("Recording dispose", error.ToString()); }
        if (_active is { } active)
        {
            try { _ = active.Writer.Finish(); }
            catch (Exception error) { AppLog.Write("Input dispose", error.ToString()); }
            try { if (Directory.Exists(active.StagePath)) _library.AbandonStage(active.StagePath); }
            catch (Exception error) { AppLog.Write("Stage dispose", error.ToString()); }
        }
        _startCancellation?.Dispose();
        _lifecycle.Dispose();
        StatusChanged = null;
    }

    private sealed record ActiveSession(
        Guid ID,
        string StagePath,
        RecordingRequest Request,
        InputEventWriter Writer,
        RecordingInputSink Sink,
        DateTimeOffset StartedAt);
}
