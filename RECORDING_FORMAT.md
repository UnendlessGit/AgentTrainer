# AgentTrainer recording interchange contract

This document is the platform-neutral contract shared by AgentTrainer for
macOS and AgentTrainer Recorder for Windows. A recorder is compatible only when
it produces this directory and these semantics; a matching filename alone is
not sufficient.

For file sharing, the Windows **Export for Mac** action wraps exactly one
`.atrrecord` directory in a standard ZIP named `<recording>.atrrecord.zip`.
The ZIP is transport only: Mac Import safely extracts it to private temporary
storage, rejects absolute/parent-relative paths and links, validates the native
package below, and deletes the extraction. Training always consumes the native
directory; video and input data are never transcoded or rewritten.

## Portable directory

Each recording is one directory named `<UUID>.atrrecord`:

```text
<UUID>.atrrecord/
├── manifest.json
├── capture.mov
├── events.atrevents
└── thumbnail.jpg        # optional
```

`capture.mov` is an ISO Base Media/QuickTime-family video containing one H.264
or HEVC video track and no required audio track. The filename is fixed even
when the Windows Media Foundation writer uses the MP4-compatible member of
that container family. AgentTrainer opens the stream by content and decodes
both codecs through VideoToolbox. Recording code must not transcode on export.

`manifest.json` uses UTF-8, camel-case keys, ISO-8601 UTC dates without
fractional seconds, and the schema-2 `RecordingManifest` in
`Sources/AgentTrainer/Core/Domain.swift`. Unknown extra fields must not be
required for decoding. `folderID` is library-local and is replaced by the
destination library during import.

## Input event stream

`events.atrevents` is always little-endian:

```text
offset  size  value
0       8     ASCII "ATREVT01"
8       4     UInt32 format version (1)
12      ...   zero or more 72-byte records
```

Each record is:

```text
offset  size  type      field
0       8     UInt64    timestampNanos
8       1     UInt8     kind (1 move, 2 button, 3 scroll, 4 key, 5 flags)
9       1     UInt8     isDown (0 or 1)
10      1     UInt8     mouse button (left 0, right 1, middle 2, then 3...7)
11      1     UInt8     reserved, zero
12      2     UInt16    Apple virtual key code
14      2     UInt16    reserved, zero
16      8     UInt64    Quartz-compatible modifier flags
24      8     Float64   global pointer X
32      8     Float64   global pointer Y
40      8     Float64   raw pointer delta X
48      8     Float64   raw pointer delta Y
56      8     Float64   horizontal point-scroll delta
64      8     Float64   vertical point-scroll delta
```

Timestamps are nanoseconds in any monotonic, per-machine host clock. They must
be finite in the UInt64 domain and nondecreasing. `hostStartNanos` is the
timestamp of the first usable encoded screen frame. Events before that boundary
are discarded. The first record is a zero-delta cursor snapshot at exactly
`hostStartNanos`, establishing absolute pointer and modifier state.

Windows input is translated to Apple virtual key codes before it reaches disk.
This is deliberate: the fixed 128-key policy layout is already defined in that
space. Windows Ctrl maps to macOS Control, Alt to Option, and the Windows key to
Command. Shift uses the dedicated Shift action. Modifier transitions are kind
5 `flags` records; they are not ordinary key records.

Quartz-compatible modifier masks used on both platforms are:

```text
Caps Lock  0x00010000
Shift      0x00020000
Control    0x00040000
Option     0x00080000
Command    0x00100000
```

Redundant operating-system key-repeat downs are removed. A press and release
inside one action interval are both retained so the training cache can preserve
the pulse. Recording hotkey keys, their buffered modifier transitions, and
their trailing releases never enter the stream. A blacklisted modifier removes
its flag from every event as well as excluding its physical key path.

## Coordinate and action semantics

The manifest's `globalRect` and every absolute pointer coordinate share the
same top-left-origin desktop coordinate space used by the source platform's
global pointer API. The dataset normalizes `(x, y)` against `globalRect`.
Windows runs per-monitor-DPI-aware and records physical desktop pixels, so its
video dimensions and pointer rectangle remain aligned at mixed display scales.

Raw mouse deltas are signed HID-style counts: right/down are positive. Training
divides their accumulated value by the fixed game-camera scale of 80 and clips
to `-1...1`. Scroll values use point-like units; one conventional Windows wheel
detent is normalized to 3 points before persistence, matching the scale used by
ordinary macOS wheel events. Training divides accumulated scroll by 20.

## Recording lifecycle

1. Freeze capture source, region, FPS, cursor visibility, trim values, folder,
   key blacklist, and recording-hotkey suppression.
2. Start the input listener, but keep the event clock closed.
3. Start GPU screen capture and hardware video encoding.
4. On the first encoded frame, write the initial pointer snapshot and open the
   event clock.
5. On stop, first stop and drain input, then finish the video writer.
6. Compute duration as the maximum of video duration and final input time.
7. Apply non-destructive trims in the manifest only. If trims consume the whole
   recording, retain the complete duration.
8. Write the manifest last and publish the `.atrrecord` directory atomically.

An incomplete directory has no manifest and is never a library item. Import
copies into a private staging directory, validates the manifest, complete event
stream, video track, and safe leaf filenames, assigns a fresh recording ID and
destination folder, then atomically publishes the directory. Source data is
never modified.
