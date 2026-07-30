# AgentTrainer Development Guide

This is the durable engineering reference for AgentTrainer 1.9.5. Keep it focused on contracts that a future change must preserve; release history belongs in Git.

## Platform and dependencies

- Swift 6.2 package
- Apple silicon, macOS 15+
- SwiftUI and AppKit
- ScreenCaptureKit and AVFoundation
- Metal and VideoToolbox
- MLX Swift pinned exactly by `Package.resolved`

The product is local-first. Do not add telemetry, cloud training, or data upload as an incidental dependency.

## Source map

- `App/AppModel.swift` — main-actor orchestration and UI state
- `Capture/` — capture source discovery, ScreenCaptureKit streaming, HEVC writing, and input monitoring
- `Storage/WorkspaceStore.swift` — actor-isolated libraries, migrations, relocation, and artifact transactions
- `Storage/InputEventFile.swift` — fixed-width binary event writer and validated mapped reader
- `Preprocessing/VisionPreprocessor.swift` — Metal resize, color conversion, packing, and MLX expansion
- `Training/DatasetCache.swift` — causal frame/action pairing and memory-mapped datasets
- `Training/PolicyNetwork.swift` — Policy v4 and resumable AdamW
- `Training/TrainingEngine.swift` — split construction, compiled steps, validation, checkpoints, and version publication
- `Agent/AgentRuntime.swift` — capture, inference scheduling, prediction latching, and teardown
- `Agent/InputInjector.swift` — final output firewall and synthetic HID events
- `Replay/InputReenactor.swift` — guarded demonstration replay
- `Core/GitHubReleaseUpdater.swift` — release discovery, verification, staging, and rollback installer
- `UI/` — views, themes, HUD, diagnostics, and CNN visualization
- `Tests/AgentTrainerTests/DomainTests.swift` — contract, corruption, concurrency, storage, MLX, and rendering coverage

## Concurrency ownership

`AppModel` is `@MainActor`. UI-visible state changes belong there.

`WorkspaceStore` and `DatasetCacheBuilder` are actors. File transactions and managed root changes must remain serialized through them.

Capture, runtime, training, and injection use dedicated locks and queues because framework callbacks are not actor-isolated. Preserve these rules:

- Never hold a runtime lock across an `await`.
- Stop scheduling actions, drain the action queue, then release physical controls.
- Post key/button releases while holding the same injector lock used by execution. A stop/restart or permission off/on transition must not overtake a release.
- Keep callbacks to UI state outside low-level locks.
- Treat start and stop as re-entrant operations; duplicate requests must coalesce or return.
- A background task may publish only bounded snapshots to the main actor.

## Capture and recording contracts

Capture uses a monotonic host-time clock shared with input events. A recording is publishable only after a complete screen frame establishes the clock.

`events.atrevents` has a 12-byte header (`ATREVT01` plus little-endian version 1) followed by 72-byte records. Readers must reject:

- unknown versions or event kinds
- truncated files
- decreasing timestamps
- non-finite control values

Large event streams stay memory-mapped. Do not replace summary or dataset paths with eager arrays.

Recording manifests validate schema, finite timing, dimensions, trims, and leaf-only artifact names. Legacy repair may correct malformed metadata after reading the real video duration, but it must retain the original manifest and never rewrite source video or input.

Capture status matters: complete/started frames are usable, idle frames may reuse the last good frame, and blank/suspended/stopped frames are dropped.

## Vision and dataset contracts

Packed observations are `UInt8`:

- grayscale: Y
- color: Y plus Cb/Cr using the selected 4:2:0, 4:2:2, or 4:4:4 layout

MLX expands color to dense RGB-like channels. Policy input concatenates the current frame, signed difference from the previous perception frame, and generated X/Y coordinate planes.

`TrainingDataContract.schemaVersion` is 7. Dataset caches are disposable and must be invalidated when causal pairing or target meaning changes.

One action row has 146 values:

- `0..<2` absolute cursor
- `2..<4` relative mouse
- `4..<12` buttons
- `12..<14` scroll
- `14..<142` key codes
- `142` Shift
- `143..<146` Control, Option, Command

Shift belongs to Keyboard. Control, Option, and Command use only their dedicated outputs; their duplicate ordinary key-code slots must be zeroed in targets, loss, history, and runtime.

Sub-tick taps must survive action sampling. Observation mappings retain current and preceding perception frames so motion is causal. Validation context must not reach into training rows or cross recording boundaries.

## Model and training contracts

`ModelContract.schemaVersion` is 4 and the weight format is `AgentTrainer.Policy.v4`.

A runnable version is immutable. Runtime vision, architecture, precision, history, channel semantics, cursor visibility, and demonstrated keys come from the version manifest, not mutable editor fields.

Current Policy v4 combines:

- convolutional spatial features
- current-frame difference channels and coordinate planes
- attention or legacy flattened visual pooling
- GRU/LSTM action history
- per-head bounded activations

Training uses class/transition weighting, focal binary loss, anti-shortcut history masking, deterministic salience-balanced order, compiled MLX updates, bounded held-out evaluation, and adaptive cosine scheduling. Do not change those semantics without updating tests and the relevant schema.

An exact checkpoint contains weights, both AdamW moment sets, optimizer metadata, scheduler state, training state, model schema, and MLX random state. Missing or mismatched moment pairs/random state are corruption, not a silent fresh start.

Checkpoint and version activation are transactions. A profile must never point at one brain while its checkpoint resumes another. Weights-only activation intentionally removes an unrelated resumable checkpoint so later training begins a fresh optimizer from the selected brain.

Completed training publishes an immutable version. With honest held-out data, the lowest acceptable validation brain is activated; periodic autosaves are bounded and never prune the active or protected brain.

## Runtime safety contracts

The final executable action is the intersection of:

1. immutable capabilities learned by the saved version
2. current profile channel disables
3. profile key/button/modifier restrictions
4. demonstrated key codes
5. live cursor and keyboard permissions
6. the allowed control region unless full-Mac control is enabled

Predictions must contain 146 finite values. Held keys/buttons may persist between fresh predictions, but cursor deltas and scroll are transient. The prediction latch consumes transient outputs once.

All synthetic events carry `agentTrainerSyntheticTag`; input safety monitors ignore that tag. Panic and normal stop both drain action work and release held state.

## Storage contracts

Default root:

```text
~/Library/Application Support/AgentTrainer
```

Training data (`Recordings`, `Caches`, folder index) and model data (`Profiles`, versions, checkpoints, contract markers) can be relocated independently.

Relocation must:

1. validate a local writable destination
2. reject nested or application-bundle destinations
3. refuse implicit merges
4. copy and verify before switching
5. persist the new root before cleaning the old copy

A cleanup failure may leave a duplicate, never destroy the only verified copy. Artifact filenames loaded from manifests are leaf names only. Cache deletion counts only successful removals.

## Updater contracts

Release checks accept stable semantic versions only. Downloads require HTTPS and approved GitHub hosts, enforce declared and observed size limits, and verify the exact DMG SHA-256 line.

Before installation, verify:

- disk image validity
- app bundle identifier
- expected version
- strict code signature
- designated requirement equality with the installed app

The helper must acknowledge startup before the app quits. An unacknowledged helper is terminated. Replacement uses an incoming app and rollback backup on the target volume.

## UI rules

The app owns a solid top bar and sidebar for consistent macOS 15+ rendering. Expensive or continuously updating content should not remain mounted when hidden.

Theme controls are bounded and global. Motion honors Reduce Motion, can be disabled, and pauses when the app is inactive. Charts keep bounded histories and downsample while preserving extrema.

User-facing technical values must be derived from shared contract helpers such as `NeuralInputSizing` and `ModelSizing`, not duplicated formulas.

## Validation and release

Before release:

1. Run `./test.sh`.
2. Run `git diff --check`.
3. Confirm `CFBundleShortVersionString` and `CFBundleVersion`.
4. Quit AgentTrainer and run `./build.sh`.
5. Verify `/Applications/AgentTrainer.app` launches and reports the intended version.
6. Exercise permission refresh, recording start/stop, training pause/resume, agent panic, and updater UI.
7. Verify the DMG and `SHA256SUMS.txt` in `outputs`.
8. Confirm no machine-specific paths, signing identities, generated files, or user data entered the source archive.

`build.sh` creates a fully signed staging bundle, transactionally installs it into `/Applications/AgentTrainer.app` by default, and then packages the DMG/source/checksums. Override the install path with `APP_PATH`; override signing with `CODE_SIGN_IDENTITY` and optionally `CODE_SIGN_KEYCHAIN`.
