# AgentTrainer Development Guide

This is the durable engineering reference for AgentTrainer 2.50. Keep it focused on contracts that a future change must preserve; release history belongs in Git.

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
- `Training/PolicyNetwork.swift` — Policy v7, predictive objectives, augmentation, and resumable AdamW
- `Training/TrainingEngine.swift` — split construction, compiled steps, validation, checkpoints, and version publication
- `Core/TemporalInputs.swift` — versioned timing vocabulary and shared causal control reducer
- `Core/MetalArrayBufferPool.swift` — pooled Metal shared-memory inputs handed directly to MLX
- `Agent/AgentRuntime.swift` — capture, inference scheduling, prediction latching, and teardown
- `Agent/InputInjector.swift` — final output firewall and synthetic HID events
- `Replay/InputReenactor.swift` — guarded demonstration replay
- `Core/GitHubReleaseUpdater.swift` — release discovery, verification, staging, and rollback installer
- `UI/ContextLabelsView.swift` — reusable context pickers, Library bulk editor, and field/value manager
- `UI/` — views, themes, HUD, diagnostics, and CNN visualization
- `Tests/AgentTrainerTests/DomainTests.swift` — contract, corruption, concurrency, storage, MLX, and rendering coverage
- `Tests/AgentTrainerTests/TrainingPerformanceTests.swift` — opt-in release hardware benchmark and trajectory checks

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

Recording transfer is producer-neutral. Import accepts a native `.atrrecord` directory, a folder containing packages, or an exported root containing `Recordings`, `recording-folders.json`, and optional `context-fields.json`. Validate every manifest, event file, video track, displayed dimension, first-frame clock, and context definition before changing the library. Stage the full batch, preserve video/event bytes, map folder and context metadata deterministically, regenerate colliding package IDs, remap conflicting context IDs by canonical name, and roll back both indexes plus every committed package if publication fails. Export writes the same portable root, includes only referenced folders and context fields, copies packages unchanged, verifies the copy, and never overwrites a non-empty destination. Recording manifests snapshot context names as well as stable IDs, so one standalone package remains self-describing without either sidecar.

Capture status matters: complete/started frames are usable, idle frames may reuse the last good frame, and blank/suspended/stopped frames are dropped.

New saved video requests a bi-planar video-range 4:2:0 ScreenCaptureKit surface at the selected even Record resolution and passes the default 8-bit path directly to the required hardware HEVC Main encoder. Lower Color Bit settings use pooled accelerated, legal-range-aware luma/chroma quantization; neutral chroma remains exactly neutral even at one bit. The user-facing compression amount maps monotonically to a resolution/FPS-aware bitrate while codec, GOP, and profile details remain internal. Live model capture remains BGRA because it is consumed directly rather than encoded. Existing/imported recording artifacts remain byte-preserving and are never migrated implicitly.

`RecordingVisualSettings` owns output width/height, ratio-lock state, Color Bit, and compression. Recording presets must snapshot these along with the pre-existing source, destination, FPS, cursor, region, trim, input blacklist, and generic context assignments; presets saved before 2.2/2.3 decode visual/context additions to safe defaults.

Current `RecordingManifest` schema is 4. Schema 4 adds optional generic context assignments. Each assignment has a field ID/name and value ID/name, and a manifest may select at most one value per field. IDs carry training identity; names are portable display snapshots. Context names are trimmed, bounded, free of control characters, and unique without locale-dependent comparison. A context edit changes only `manifest.json`; it never rewrites video or input.

Current recording manifests persist the exact successfully encoded frame count. The field is optional for compatibility; Library presentation estimates a missing legacy count from duration and delivered FPS. Package size is the logical recursive byte total collected once during the workspace scan, never a per-row filesystem walk.

Capture-source discovery is a live list, not a one-shot bootstrap result. Refresh it on app activation, display topology changes, application launch/termination, and through a bounded two-second poll while a source picker is visible. Publish only changed, stably sorted snapshots, discard stale overlapping results, and do not disturb an active capture.

The input event tap is backed by a low-rate physical-state reconciliation pass. Reconciliation reads the HID-system hardware table, never the foreground session's mutable accumulated event state: another app's event sources must not synthesize a release for a physically held key. A poll is also revision-guarded, so a snapshot taken before a concurrent first-frame recording seed is discarded instead of overwriting newer physical truth. If macOS disables the tap or drops a transition, the next valid pass emits only the missing key, modifier, or mouse-button edges. Recording shutdown appends balanced releases for every logical control still held. The first complete screen frame seeds controls that were already held during asynchronous startup. Shortcut and blacklist filters apply to both ordinary and reconciled input.

Live HUD state and persisted recording state are intentionally separate. The HUD reduces blacklist-sanitized physical samples immediately, so a held modifier or mouse button is visible on its down edge; monotonic state revisions prevent cross-thread UI delivery from restoring an older state. The event file may briefly buffer modifier transitions to determine whether they belong to a global shortcut, but it preserves their original timestamps and separately held modifiers. Global shortcut suppression ends with the trigger/modifier release lifecycle; it must never consume an unrelated input merely because it occurs inside an expiry window. Caps Lock is normalized to a tap and Fn/Globe retains physical key edges so both remain in the ordinary keyboard action space.

Saved HEVC sessions end on the shared host clock even when ScreenCaptureKit reports only idle frames. Dataset construction materializes the last decoded frame at every configured perception interval until a newer frame becomes causal; a static screen must not collapse temporal spacing or observation count.

## Vision and dataset contracts

Packed observations are `UInt8`:

- grayscale: Y
- color: Y plus Cb/Cr using the selected 4:2:0, 4:2:2, or 4:4:4 layout

MLX expands each packed image independently to dense RGB-like channels at its native configured size. The visual encoder appends generated X/Y coordinate planes internally.

Temporal vision is current-only or includes a historical sequence. Timing is always present. The visual/control portion contains:

1. one current frame at the exact configured `PreprocessingSpec`
2. `pastFrameCount` real causal frames at `pastFrameSpec`, ordered oldest to newest
3. one complete 146-value control row paired with each past frame

`TemporalVisionConfiguration` owns past-frame count, spacing in Perception FPS intervals, and linear downscale. Its displayed seconds are nominal values derived from configured Perception FPS; dropped or delayed live perceptions can increase wall-clock separation. The reduced specification must copy color mode, chroma layout, bit detail, and resize policy from the current frame. Width and height use the same scale, with only unavoidable integer-pixel rounding. Never resize past images back to current resolution and never synthesize difference, optical-flow, or motion channels.

The default is four past frames, two perception intervals apart, with a 2× linear downscale. Bounds are part of the public safety/performance contract: 0–32 frames, 1–240 intervals, 1×–8× downscale, and at most 512 MB of retained packed runtime context. Zero frames omits the recurrent visual history and past-image/control/timing buffers; current decision timing remains available. The retained-context limit includes packed pixels, Float32 controls/timing, and cached visual embeddings.

`TrainingDataContract.schemaVersion` is 15. This changes causal frame selection, trailing control pairing, timing inputs, and supervision endpoints; older caches rebuild. Model schema 7 requires retraining, with incompatible brains/checkpoints archived rather than claiming a warm start or exact v6 resume. Display-only context renames still refresh the tiny cache manifest without repacking pixels. A cache has:

- `current-observations.bin` — one full-resolution packed image per perception tick
- `past-observations.bin` — one reduced packed image per tick, omitted for current-only
- `observation-indices.bin` — current plus configured past indices for every action row; missing history uses `UInt32.max`
- `frame-actions.bin` — complete trailing-interval controls, omitted for current-only
- `actions.bin` — forward action targets; the previous target remains separate for transition weighting
- `decision-timing.bin` — 2,190 Float32 timing values per action decision
- `frame-timing.bin` — the same vocabulary per historical observation, omitted for current-only
- `observation-times.bin` — little-endian UInt64 query time and actual source PTS per observation, relative to recording origin
- `decision-times.bin` — little-endian UInt64 query time and positive target interval duration per action row

All files have validated sizes and bounded arithmetic. Timing rows must be finite; segment-owned clock metadata is monotonic and causal, including source PTS ≤ observation query ≤ action query. Decode begins at movie time zero, retaining the last causal source PTS at a trim rather than rebasing the first decoded frame. Targets cover `[t, min(t + actionInterval, trimEnd))`; inputs contain only events strictly before `t`. Priming preserves pre-trim hold onset history. Zero-duration endpoint targets are excluded.

The detailed timing layout, source audit, ablation procedure, and measurements are in [TEMPORAL_DESIGN.md](TEMPORAL_DESIGN.md).

The cache manifest snapshots a deterministic `ModelContextSchema`. Fields and values are UUID-sorted for tensor stability and include only values assigned by at least one selected recording. Each `CacheSegment` stores that recording's assignments once; batch gathering expands them to a multi-hot Float row. Each value owns one dimension, at most one value per field is active, and an unassigned field is an all-zero segment. Context-free caches have no context descriptor or Metal allocation.

Each `CacheSegment` owns both an action-row start and an observation start. Legacy recordings may have observation context before supervised targets. Cache validation derives past indices from the observation start, never the first supervised row. All recordings use equal sample emphasis; mask weights remain shared by padding and future-prediction objectives.

Future action rows/previous rows and future observation maps are gathered from existing cache data at training time. Every horizon is masked at segment boundaries. Future-state targets use a configurable stride and deduplicate repeated observation indices before visual encoding. No future label or image may cross a recording boundary.

For a current-only profile, each observation-index row contains only its current index; `past-observations.bin` and `frame-actions.bin` remain zero-byte compatibility placeholders, and their manifest byte stride is zero. No synthetic zero-sized Metal buffer is created.

One action row has 146 values:

- `0..<2` absolute cursor
- `2..<4` relative mouse
- `4..<12` buttons
- `12..<14` scroll
- `14..<142` key codes
- `142` Shift
- `143..<146` Control, Option, Command

Shift belongs to Keyboard. Control, Option, and Command use only their dedicated outputs; their duplicate ordinary key-code slots must be zeroed in targets, loss, frame-aligned controls, and runtime.

Every frame-control row includes normalized absolute cursor position, raw relative movement, held and sub-tick mouse-button presses, scroll, held and sub-tick key presses, Shift, Control, Option, and Command. Sub-tick taps must survive both action and perception-interval sampling. Cache loading verifies exact configured spacing, strictly causal indices, monotonic current frames, segment-leading sentinels, file sizes, and bounded arithmetic. Validation context must not reach into training frames or cross recording boundaries.

## Model and training contracts

`ModelContract.schemaVersion` is 7 and the weight format is `AgentTrainer.Policy.v7`.

All training uses behavioral cloning. Older feature settings decode as unknown keys and are omitted when saved. `ModelOutputCapabilities` snapshots immutable key/button/movement/scroll limits. Native v7 publication derives binary capabilities from positive supervised classes in the training split, so loss-masked heads cannot post random events. Imported limits remain immutable; this is not an action distribution. Missing legacy limits fail closed. Run always uses the established deterministic activations and the shared output firewall.

Legacy recording `supervisionStart` remains a data compatibility contract: pre-boundary images and controls are observation context, never action labels. Dataset schema 15 rebuilds packed caches; incompatible v6 brains/checkpoints are archived and must be retrained. Migration from v6 preserves profile architecture and temporal settings. There are no special recording purposes, per-recording loss weights, or purpose-specific validation splits.

A runnable version is immutable. Runtime current vision, temporal vision, architecture, precision, channel semantics, cursor visibility, and demonstrated keys come from the version manifest, not mutable editor fields.

Dropout is stochastic training regularization, not part of the learned tensor layout. It is normalized out of `LearnedBrainContract`: editing only dropout must preserve the active brain and load its weights into the next run. Because the stochastic objective changed, the exact optimizer/checkpoint signature still includes dropout and begins a fresh optimizer sequence rather than claiming an invalid exact resume.

Current Policy v7 combines:

- one shared convolutional encoder for the real current and past images
- independent X/Y coordinate planes at each image's native resolution
- attention or legacy flattened visual pooling
- a shared Float32 timing projection (2,190 → 64, no bias) plus LayerNorm, cast to configured model precision
- a GRU/LSTM sequence over each past visual embedding, its 146 controls, and its 64 timing values
- fusion of current visual embedding, final temporal state, and current 64-value timing embedding
- optional raw categorical context concatenated immediately before the fusion stack
- per-head bounded activations
- a fixed four-horizon future-action head and an action-conditioned latent-state-delta head; both are training-only and add no runtime action latency

`ModelVersionManifest.contextSchema` is the immutable display and tensor vocabulary for a context-trained brain. It is nil for every pre-2.3/context-free brain. Empty context adds no tensor, parameter, input descriptor, or compiled-step argument, preserving the exact existing Policy-v7 weight layout. A warm start is allowed only when ordered field/value IDs match; display-name changes alone are layout-compatible. The first fusion layer is the only learned shape widened by context values.

Attention pooling shares its projection across current and past image sizes. Legacy flattened pooling needs distinct current and past projections because its spatial dimensions differ. One per-sample keep mask drops the historical controls and all current/historical control-derived timing (including cursor) for half of training samples while visual clocks, validity, and images remain visible; this prevents held controls from becoming an easy shortcut without deleting temporal visual evidence. Inference always receives the full paired sequence.

Training uses class/transition weighting, focal binary loss, anti-shortcut frame-control masking, deterministic salience-balanced order, compiled MLX updates, bounded held-out evaluation, and adaptive cosine scheduling. Configurable photometric augmentation samples one transform per supervised row and applies it identically to only that row's current, past, and future frames. It never moves pixels. Augmented training expands deduplicated observations to row shape before the CNN and uses the diversity-oriented sampler; validation and augmentation-off training retain deterministic visual reuse. Future-action loss weights near horizons more strongly. Future-state loss predicts a stopped-gradient latent change from fused context plus the demonstrated current action; it never generates pixels. Conservative default auxiliary weights are gated by a matched-update causal held-out test at 1% immediate-loss drift. Do not change these semantics without updating tests and the relevant schema.

The performance path preserves those semantics:

- VideoToolbox decodes directly into native video-range YUV. One CVMetalTexture mapping and Metal command buffer produce both configured output resolutions, while decode, resize/color packing, and cache writes overlap through a bounded ordered pipeline of up to eight observations within a 64 MB cap.
- When one decoded frame remains causal across many configured perception ticks, Metal packs it once and the cache writer repeats those exact packed bytes. Observation cadence, indices, controls, and every action label remain unchanged.
- Dataset files are required memory mappings with adaptive VM advice. Epochs randomize small collections of temporally local lanes rather than every row globally, so each optimizer batch remains diverse while macOS can read ahead the contiguous mapped pages inside each lane. A batch gathers packed vision, frame controls, and target/previous-target rows once into pooled Metal shared memory; MLX expands packed `UInt8` vision inside the compiled graph.
- With augmentation off, current images are encoded once per distinct temporal sequence. Overlapping causal windows additionally encode each distinct reduced-resolution past observation once, then gather those deterministic embeddings to their exact frame slots. With augmentation on, rows are expanded before encoding so independent sample transforms cannot be collapsed. No label, loss term, update, or temporal control is removed, duplicated, averaged, or reused as another label.
- Above the measured workload crossover, the coordinate contribution to the first convolution is evaluated once per batch and the established GroupNorm layout uses MLX's fused Metal layer-normalization kernel. Small tensors retain the lower-overhead kernels.
- Validation reuses deterministic CNN embeddings, gathers them to action-row shape, then computes timing-conditioned recurrence and head logits independently for every held-out label. Sharing images must never share final predictions across distinct timing rows.
- CPU gathering of the next batch overlaps the current Metal graph. Every optimizer step still evaluates the complete model, AdamW moments, scheduler, and MLX random state before advancing the checkpoint cursor.
- Live diagnostics publish a rolling pipelined-step time, mapped-input time, peak-throughput retention, MLX active/cache/peak memory, and macOS thermal pressure. Metrics are rate-limited and chart histories are presentation-bounded so observing a run does not grow UI work with optimizer-step count.
- The release benchmark separately executes the complete Policy-v7 stochastic objective and runs the same sustained-window retention/memory/thermal checks. It gates the required per-sample augmentation path at 3× the augmentation-off reused core, then gates both predictive heads together at 2× that augmented immediate path. The immediate-only reference/optimized paths still isolate encoder equivalence and throughput, while the two-stage gate prevents one semantic cost from hiding a regression in the other.

Do not queue stateful optimizer steps merely to improve a benchmark. Exact resume means a saved checkpoint must continue its own versioned sampling contract, weights, moments, scheduler, and random state without skipping or replaying an update.

Performance work may change graph operation order, grouped sample order, and low-bit numerical trajectories. “No quality loss” is evaluated by matched-update training loss, held-out validation loss, and per-head validation behavior—not byte-identical weights or predictions. The hardware benchmark gates relative validation-loss drift at 1%, while focused tests compare the optimized objective and gradients to the reference graph. Never obtain speed by silently reducing vision resolution, precision, model capacity, labels, loss terms, or update count.

For a longer power/thermal comparison of the optimized and reference 16-frame paths, run `AGENTTRAINER_BENCHMARK_STRESS_STEPS=2048 AGENTTRAINER_BENCHMARK_STRESS_BASELINE_STEPS=1024 ./benchmark.sh`. Compare first/last-window retention as well as thermal state and cache memory; a coarse nominal macOS thermal state does not prove that sustained GPU clocks remained at their initial boost level.

An exact checkpoint contains weights, both AdamW moment sets, optimizer metadata, scheduler state, training state, model schema, and MLX random state. A resumable immutable version must name and contain all three optimizer, state, and random artifacts; missing or mismatched moment pairs/random state are corruption, not a silent fresh start.

Checkpoint and version activation are transactions. A profile must never point at one brain while its checkpoint resumes another. Weights-only activation intentionally removes an unrelated resumable checkpoint so later training begins a fresh optimizer from the selected brain.

Completed ordinary training publishes the lowest acceptable held-out-validation brain. Periodic autosaves are bounded and never prune the active or protected brain.

## Runtime safety contracts

The final executable action is the intersection of:

1. immutable capabilities learned by the saved version
2. current profile channel disables
3. profile key/button/modifier restrictions
4. demonstrated key codes
5. live cursor and keyboard permissions
6. the allowed control region unless full-Mac control is enabled

Every mutable profile channel is a disable-only runtime firewall over the immutable saved channels—including mouse movement, buttons, scroll, keyboard, and modifiers. A current editor switch may never enable a head the saved brain did not learn or continue executing a head the user disabled.

Predictions must contain 146 finite values. Held keys/buttons may persist between fresh predictions, but cursor deltas and scroll are transient. The prediction latch consumes transient outputs once.

At runtime, every successfully encoded perception stores its reduced packed image and evaluated embedding with controls/timing reduced from events actually posted under the injector lock. Predictions that permissions, restrictions, rounding, or clipping prevented from executing never enter this history. Forced releases do. Exact spaced frames use a fixed-capacity circular buffer; unavailable startup slots duplicate the current reduced image with zero controls and invalid timing.

CNN encoding runs at Perception FPS; a lightweight timing-conditioned head runs at Action FPS. Both share one serialized MLX queue. Newest-frame work coalesces to one pending image and yields after each encode; action work admits at most one pending head/execution. Revisions discard obsolete context/permission results both after inference and before posting. Stop drains action work/releases, drains capture/inference, then drains late action callbacks before allowing restart. An independent action-timer watchdog stops stale predictions even if MLX work is pending. Idle perceptions retain the true original source-frame timestamp.

Two independent injector interval reducers keep action and perception rates correct. Historical query ages are recomputed at each action tick; source-frame age is retained separately. Runtime diagnostics and visualization/saliency graphs receive the same timing vocabulary. Head latency and source-frame age are exposed separately from visual encoding latency. At 60 Hz execution remains approximately 16.7 ms; nanosecond input provenance is not an arbitrary sub-tick output scheduler.

Run reads context fields and values only from the active immutable version, never the mutable workspace catalog. Selecting another brain clears every Run selection to blank. A live context change increments a runtime revision, clears temporal/prediction state, drains and releases old-context held controls on the action queue, and discards inference already in flight under the previous revision before admitting a new-context prediction.

All synthetic events carry `agentTrainerSyntheticTag`; input safety monitors ignore that tag. Panic and normal stop both drain action work and release held state.

## Storage contracts

Default root:

```text
~/Library/Application Support/AgentTrainer
```

Training data (`Recordings`, `Caches`, folder index, context-field catalog) and model data (`Profiles`, versions, checkpoints, contract markers) can be relocated independently.

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

Library recording rows and inspectors expose encoded frames plus logical package MB/GB. Folder headers aggregate every recording in that folder—not only search results—and show total recordings, frames, logical bytes, and full recorded seconds. Newly created manifests use exact frame counts; legacy estimates must remain compatible and non-destructive.

Recording presets are user-owned, editable snapshots of Record-page settings. Applying a preset must tolerate an unavailable source or deleted destination without inventing an identifier. Library range selection follows the visible folder/search order; bulk manifest moves roll back as a set, and bulk deletion stages all recording directories before cleanup. Global shortcuts may use keyboard keys or any macOS mouse-button number, including additional side buttons, and must remain excluded from recording and human-input safety hooks.

Context UI uses one shared generic field/value catalog. Record freezes selections at startup. Library single and multi-selection pickers may set or clear any field; mixed bulk values stay unchanged until explicitly replaced. Catalog renames update affected recording snapshots transactionally. Deleting a field/value requires confirmation and transactionally removes matching recording assignments, while immutable model-version schemas remain unchanged. Run renders a context card only when the active brain's schema is non-empty and starts every trained field blank.

The scripts explicitly select SwiftPM's `native` backend. Newer Swift toolchains may default to a different backend and output layout; do not silently change backend, executable discovery, or Metal-library placement independently. `test.sh` supplies a disposable `AGENTTRAINER_WORKSPACE_ROOT` for the real training/checkpoint integration test.

## Validation and release

Before release:

1. Run `./test.sh`.
2. Run `./benchmark-temporal.sh` for the approved isolated 20-epoch ablation (completed arms are evaluation-only), then run `./benchmark.sh` and compare original/optimized throughput plus matched-update training and validation loss on the same idle Mac. Check the full auxiliary-objective overhead/memory line and every sustained window for stable throughput, bounded MLX cache, and nominal thermal pressure.
3. Run `git diff --check`.
4. Confirm `CFBundleShortVersionString` and `CFBundleVersion`.
5. Quit AgentTrainer and run `./build.sh`.
6. Verify `/Applications/AgentTrainer.app` launches and reports the intended version.
7. Exercise permission refresh, recording start/stop, human takeover, training pause/resume, agent panic, and updater UI.
8. Verify the DMG and `SHA256SUMS.txt` in `outputs`.
9. Confirm no machine-specific paths, signing identities, generated files, or user data entered the source archive.

`build.sh` creates a fully signed staging bundle, transactionally installs it into `/Applications/AgentTrainer.app` by default, and then packages the DMG/source/checksums. Override the install path with `APP_PATH`; override signing with `CODE_SIGN_IDENTITY` and optionally `CODE_SIGN_KEYCHAIN`.

`scripts/signing.sh` resolves exact usable certificate fingerprints, rejects revoked/ambiguous identities, and pins the certificate and designated requirement outside the checkout in `~/Library/Application Support/AgentTrainer/BuildSigning`. Never fall back to another certificate or ad-hoc signing when the pinned key is unavailable. Before replacement and after DMG stripping/signing, require the same designated requirement as the installed app and saved pin. Explicit certificate renewal must pass that same check. Keep these machine-local pins and private keys out of source/distribution archives. Run `zsh scripts/test-signing.sh` for focused signing regressions (also included in `test.sh`).
