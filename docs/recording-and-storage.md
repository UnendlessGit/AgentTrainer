# Recording and storage

Back to [architecture overview](../ARCHITECTURE.md).

Capture, synchronized input, recording packages, and library storage in AgentTrainer 2.52. Dataset preprocessing is described in [Model and training](model-and-training.md#vision-and-dataset-construction).

## Recording pipeline

ScreenCaptureKit supplies screen buffers and presentation timestamps on a monotonic host clock. The first complete frame establishes the recording clock and seeds already-held physical controls. Input events use the same clock, so later dataset queries can align them with video without deriving timing from nominal frame numbers.

Complete/started frames are accepted; idle notifications can reuse the most recent usable image; blank, suspended, and stopped frames are dropped. Video finalization ends the HEVC session on the shared clock, including when a static screen produced only idle notifications near the end.

Saved recording capture uses bi-planar video-range 4:2:0 at the configured even dimensions. The 8-bit path passes the captured surface directly to a hardware-required HEVC Main encoder. Lower Color Bit settings use pooled accelerated luma/chroma quantization before encoding. Legal-range-aware tables preserve neutral chroma. Compression maps to a bitrate based on resolution and requested FPS. Live inference capture uses BGRA because those frames are preprocessed rather than video-encoded.

The input tap records key, modifier, mouse-button, relative/absolute movement, and scroll samples. A periodic reconciliation pass reads `.hidSystemState` to recover transitions missed by the tap. Revision checks discard snapshots taken before newer physical-state updates or first-frame seeding. Recording shutdown emits balanced releases for logical controls still held.

Recording blacklists and global-shortcut filters apply to ordinary and reconciled events. Shortcut handling may buffer modifier transitions while preserving timestamps. HUD state is reduced separately and published with revisions, allowing immediate physical-state display while persisted events are filtered. Caps Lock is represented as a tap; Fn/Globe uses physical key edges.

Source discovery refreshes on a two-second poll while Record/Run is visible, the app is active, and capture/runtime startup or execution is idle. App activation and application/display changes trigger the same guarded refresh path. A revision discards stale overlapping discovery results.

### Recording packages and events

A native `.atrrecord` directory contains a JSON manifest, video, input events, and a thumbnail. Recording manifest schema 4 contains optional generic context assignments. Each assignment stores field/value IDs for identity and field/value names for portable display. A context metadata edit rewrites the manifest rather than the video/event files.

`events.atrevents` starts with the eight bytes `ATREVT01` and a little-endian UInt32 version of 1. Each subsequent record occupies 72 bytes:

| Offset | Type | Meaning |
| --- | --- | --- |
| 0 | UInt64 | Monotonic timestamp in nanoseconds |
| 8 | UInt8 | Event kind |
| 9 | UInt8 | Down/up flag |
| 10 | UInt8 | Mouse-button number |
| 11 | UInt8 | Reserved |
| 12 | UInt16 | Key code |
| 14 | UInt16 | Reserved |
| 16 | UInt64 | Modifier flags |
| 24, 32 | Double, Double | Cursor X/Y |
| 40, 48 | Double, Double | Relative movement X/Y |
| 56, 64 | Double, Double | Scroll X/Y |

Numeric fields are written little-endian. The reader validates headers, record sizes, event kinds, timestamp order, and finite values. Dataset reading uses mapped random access, avoiding an expanded Swift object array for large mouse-heavy recordings. The writer buffers output and clamps an out-of-order append timestamp to the preceding timestamp to handle callback races.

Manifests validate timing, dimensions, trims, and leaf-only artifact names. Legacy clock repair retains the original manifest and derives corrections from the actual video; it does not transcode source media. New manifests save successfully encoded frame counts. Library display estimates legacy counts from duration/FPS and uses package sizes collected during library scans.

### Transfer and organization

Import accepts a package, a folder of packages, or an exported root with `Recordings`, `recording-folders.json`, and optional `context-fields.json`. It validates the input batch, stages copied artifacts, maps folders/context definitions, and assigns new package IDs where necessary. Video and event bytes remain unchanged while manifest metadata can change. Publication includes rollback of committed packages and indexes on failure.

Export copies selected packages and referenced folder/context definitions, verifies the copies, and rejects a non-empty destination. Context name snapshots allow a standalone package to describe its labels without sidecars.

Recording presets contain source, destination, capture/visual settings, input blacklist, trims, and context selections. Library bulk moves/metadata edits use grouped rollback; deletion stages package directories before cleanup. Folder totals include all recordings in a folder independently of search filtering.

## Workspace storage and relocation

The default workspace is `~/Library/Application Support/AgentTrainer`. Training data (`Recordings`, `Caches`, folder/context catalogs) and model data (`Profiles` and compatibility markers) have independently configurable roots. `AGENTTRAINER_WORKSPACE_ROOT` supplies an isolated root for tests/experiments without persisting the user's root settings.

Relocation validates writable local destinations, rejects nested/app-bundle destinations and implicit merges, copies and verifies data, persists the new root, and then cleans the old copy. Failed cleanup can leave a duplicate. Switching to an existing managed library changes roots without merging libraries.
