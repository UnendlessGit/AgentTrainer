using AgentTrainer.Recorder.Core;
using ScreenRecorderLib;
using ScreenRecorder = ScreenRecorderLib.Recorder;

namespace AgentTrainer.Recorder;

internal sealed record WindowsCaptureResult(
    ulong FirstFrameHostNanos,
    ulong LastFrameHostNanos,
    int FrameCount,
    int Width,
    int Height,
    double Duration,
    double DeliveredFPS,
    string Codec,
    bool HasThumbnail);

internal sealed class WindowsScreenRecorder : IDisposable
{
    private readonly object _gate = new();
    private Attempt? _attempt;
    private CapturePlan? _plan;
    private string? _temporaryVideoPath;
    private string? _finalVideoPath;
    private string? _thumbnailPath;
    private string _codec = "";
    private ulong _stopRequestedNanos;
    private bool _disposed;

    internal bool IsActive
    {
        get { lock (_gate) return _attempt is not null; }
    }

    internal async Task StartAsync(
        CapturePlan plan,
        string temporaryVideoPath,
        string finalVideoPath,
        string thumbnailPath,
        bool preferHevc,
        Action<ulong> onFirstFrame,
        CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        lock (_gate)
        {
            if (_attempt is not null) throw new InvalidOperationException("A screen recording is already active.");
            _plan = plan;
            _temporaryVideoPath = temporaryVideoPath;
            _finalVideoPath = finalVideoPath;
            _thumbnailPath = thumbnailPath;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(temporaryVideoPath)!);
        var codecs = preferHevc ? new[] { "HEVC", "H.264" } : new[] { "H.264" };
        Exception? lastError = null;
        foreach (var codec in codecs)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (File.Exists(temporaryVideoPath)) File.Delete(temporaryVideoPath);
            var attempt = CreateAttempt(plan, temporaryVideoPath, thumbnailPath, codec, onFirstFrame);
            lock (_gate) _attempt = attempt;
            try
            {
                attempt.Recorder.Record(temporaryVideoPath);
                using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
                timeout.CancelAfter(TimeSpan.FromSeconds(12));
                await attempt.FirstFrame.Task.WaitAsync(timeout.Token).ConfigureAwait(false);
                lock (_gate) _codec = codec;
                return;
            }
            catch (Exception error) when (codec == "HEVC" && attempt.FirstFrameNanos == 0 && !cancellationToken.IsCancellationRequested)
            {
                lastError = error;
                AppLog.Write("Capture", $"HEVC startup failed; retrying H.264. {error}");
                await EndFailedAttemptAsync(attempt).ConfigureAwait(false);
                attempt.Dispose();
                lock (_gate) { if (ReferenceEquals(_attempt, attempt)) _attempt = null; }
            }
            catch
            {
                await EndFailedAttemptAsync(attempt).ConfigureAwait(false);
                attempt.Dispose();
                lock (_gate) { if (ReferenceEquals(_attempt, attempt)) _attempt = null; }
                throw;
            }
        }
        throw new InvalidOperationException("Windows could not start a compatible video encoder.", lastError);
    }

    internal async Task<WindowsCaptureResult> StopAsync(CancellationToken cancellationToken = default)
    {
        Attempt attempt;
        CapturePlan plan;
        string temporaryVideo;
        string finalVideo;
        string thumbnail;
        string codec;
        lock (_gate)
        {
            attempt = _attempt ?? throw new InvalidOperationException("No screen recording is active.");
            plan = _plan ?? throw new InvalidOperationException("The active capture plan is unavailable.");
            temporaryVideo = _temporaryVideoPath ?? throw new InvalidOperationException("The active video path is unavailable.");
            finalVideo = _finalVideoPath ?? throw new InvalidOperationException("The final video path is unavailable.");
            thumbnail = _thumbnailPath ?? throw new InvalidOperationException("The thumbnail path is unavailable.");
            codec = _codec;
            _stopRequestedNanos = NativeMethods.HostNanos();
        }

        try
        {
            attempt.Recorder.Stop();
            var completion = await attempt.Completion.Task.WaitAsync(TimeSpan.FromSeconds(30), cancellationToken).ConfigureAwait(false);
            if (!string.IsNullOrEmpty(completion.Error)) throw new InvalidOperationException(completion.Error);
            if (attempt.FirstFrameNanos == 0 || attempt.FrameCount == 0)
                throw new InvalidDataException("No complete screen frame was captured.");
            if (!File.Exists(temporaryVideo) || new FileInfo(temporaryVideo).Length == 0)
                throw new InvalidDataException("The Windows video encoder produced an empty recording.");

            if (File.Exists(finalVideo)) File.Delete(finalVideo);
            File.Move(temporaryVideo, finalVideo);
            try { await attempt.SnapshotTask.WaitAsync(TimeSpan.FromSeconds(2), cancellationToken).ConfigureAwait(false); }
            catch (TimeoutException) { }
            var lastFrame = Math.Max(attempt.LastFrameNanos, attempt.FirstFrameNanos);
            var frameSpan = (lastFrame - attempt.FirstFrameNanos) / 1_000_000_000.0;
            var wallSpan = (_stopRequestedNanos - attempt.FirstFrameNanos) / 1_000_000_000.0;
            var duration = Math.Max(frameSpan + 1.0 / Math.Max(1, plan.Spec.RequestedFPS), wallSpan);
            var deliveredFPS = attempt.FrameCount <= 1 || frameSpan <= 0
                ? plan.Spec.RequestedFPS
                : (attempt.FrameCount - 1) / frameSpan;
            var hasThumbnail = File.Exists(thumbnail) && new FileInfo(thumbnail).Length > 0;
            return new WindowsCaptureResult(attempt.FirstFrameNanos, lastFrame, attempt.FrameCount,
                plan.PixelWidth, plan.PixelHeight, duration, Math.Min(1000, deliveredFPS), codec, hasThumbnail);
        }
        finally
        {
            attempt.Dispose();
            lock (_gate)
            {
                if (ReferenceEquals(_attempt, attempt)) _attempt = null;
                _plan = null;
                _temporaryVideoPath = null;
                _finalVideoPath = null;
                _thumbnailPath = null;
                _codec = "";
            }
        }
    }

    internal async Task CancelAsync()
    {
        Attempt? attempt;
        lock (_gate) attempt = _attempt;
        if (attempt is null) return;
        await EndFailedAttemptAsync(attempt).ConfigureAwait(false);
        attempt.Dispose();
        lock (_gate)
        {
            if (ReferenceEquals(_attempt, attempt)) _attempt = null;
            _plan = null;
        }
    }

    private static Attempt CreateAttempt(CapturePlan plan, string videoPath, string thumbnailPath, string codec, Action<ulong> onFirstFrame)
    {
        var options = RecorderOptions.Default;
        options.SourceOptions.RecordingSources.Add(plan.RecordingSource);
        options.OutputOptions.RecorderMode = RecorderMode.Video;
        options.OutputOptions.OutputFrameSize = new ScreenSize(plan.PixelWidth, plan.PixelHeight);
        options.OutputOptions.Stretch = StretchMode.Fill;
        options.AudioOptions.IsAudioEnabled = false;
        options.MouseOptions.IsMousePointerEnabled = plan.Spec.ShowsCursor;
        options.MouseOptions.IsMouseClicksDetected = false;
        options.SnapshotOptions.SnapshotFormat = ImageFormat.JPEG;
        options.SnapshotOptions.SnapshotsWithVideo = false;
        options.LogOptions.IsLogEnabled = false;
        options.VideoEncoderOptions = new VideoEncoderOptions
        {
            Encoder = codec == "HEVC"
                ? new H265VideoEncoder { BitrateMode = H265BitrateControlMode.Quality }
                : new H264VideoEncoder { BitrateMode = H264BitrateControlMode.Quality, EncoderProfile = H264Profile.High },
            Framerate = checked((int)Math.Round(plan.Spec.RequestedFPS)),
            Bitrate = RecommendedBitrate(plan.PixelWidth, plan.PixelHeight, plan.Spec.RequestedFPS, codec),
            Quality = 78,
            IsFixedFramerate = false,
            IsThrottlingDisabled = false,
            IsHardwareEncodingEnabled = true,
            IsLowLatencyEnabled = true,
            IsMp4FastStartEnabled = false,
            IsFragmentedMp4Enabled = false
        };

        var recorder = ScreenRecorder.CreateRecorder(options);
        var attempt = new Attempt(recorder, videoPath);
        recorder.OnFrameRecorded += (_, args) =>
        {
            var now = NativeMethods.HostNanos();
            if (Interlocked.CompareExchange(ref attempt.FirstFrameTicks, checked((long)now), 0) == 0)
            {
                try
                {
                    onFirstFrame(now);
                    attempt.FirstFrame.TrySetResult(now);
                }
                catch (Exception error)
                {
                    attempt.FirstFrame.TrySetException(error);
                }
            }
            Interlocked.Exchange(ref attempt.LastFrameTicks, checked((long)now));
            var frameCount = Interlocked.Increment(ref attempt.Frames);
            if (frameCount == 12 && Interlocked.Exchange(ref attempt.SnapshotRequested, 1) == 0)
            {
                attempt.SnapshotTask = Task.Run(() =>
                {
                    try { _ = recorder.TakeSnapshot(thumbnailPath); }
                    catch (Exception error) { AppLog.Write("Thumbnail", error.ToString()); }
                });
            }
            _ = args;
        };
        recorder.OnRecordingFailed += (_, args) =>
        {
            var error = string.IsNullOrWhiteSpace(args.Error) ? "Windows screen recording failed." : args.Error;
            attempt.FirstFrame.TrySetException(new InvalidOperationException(error));
            attempt.Completion.TrySetResult(new Completion(error));
        };
        recorder.OnRecordingComplete += (_, _) => attempt.Completion.TrySetResult(new Completion(null));
        return attempt;
    }

    private static async Task EndFailedAttemptAsync(Attempt attempt)
    {
        try
        {
            var shouldWait = attempt.Completion.Task.IsCompleted;
            if (attempt.Recorder.Status is RecorderStatus.Recording or RecorderStatus.Paused or RecorderStatus.Finishing)
            {
                if (attempt.Recorder.Status is RecorderStatus.Recording or RecorderStatus.Paused) attempt.Recorder.Stop();
                shouldWait = true;
            }
            if (shouldWait) _ = await attempt.Completion.Task.WaitAsync(TimeSpan.FromSeconds(8)).ConfigureAwait(false);
        }
        catch (Exception error)
        {
            AppLog.Write("Capture cleanup", error.ToString());
        }
        if (File.Exists(attempt.VideoPath))
        {
            try { File.Delete(attempt.VideoPath); }
            catch (IOException) { }
        }
    }

    private static int RecommendedBitrate(int width, int height, double framesPerSecond, string codec)
    {
        var bitsPerPixelFrame = codec == "HEVC" ? 0.09 : 0.14;
        var value = width * (double)height * framesPerSecond * bitsPerPixelFrame;
        return checked((int)Math.Clamp(value, 4_000_000, 80_000_000));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        try { CancelAsync().GetAwaiter().GetResult(); }
        catch (Exception error) { AppLog.Write("Capture dispose", error.ToString()); }
    }

    private sealed record Completion(string? Error);

    private sealed class Attempt(ScreenRecorder recorder, string videoPath) : IDisposable
    {
        internal ScreenRecorder Recorder { get; } = recorder;
        internal string VideoPath { get; } = videoPath;
        internal Task SnapshotTask { get; set; } = Task.CompletedTask;
        internal TaskCompletionSource<ulong> FirstFrame { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        internal TaskCompletionSource<Completion> Completion { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        internal long FirstFrameTicks;
        internal long LastFrameTicks;
        internal int Frames;
        internal int SnapshotRequested;
        internal ulong FirstFrameNanos => checked((ulong)Math.Max(0, Interlocked.Read(ref FirstFrameTicks)));
        internal ulong LastFrameNanos => checked((ulong)Math.Max(0, Interlocked.Read(ref LastFrameTicks)));
        internal int FrameCount => Math.Max(0, Volatile.Read(ref Frames));
        public void Dispose() => Recorder.Dispose();
    }
}
