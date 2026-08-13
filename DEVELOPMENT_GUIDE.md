# AgentTrainer Development Guide

This is the durable engineering reference for AgentTrainer 2.5.0. Keep it focused on contracts that a future change must preserve; release history belongs in Git.

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
- `Training/PolicyNetwork.swift` — Policy v6 and resumable AdamW
- `Training/TrainingEngine.swift` — split construction, compiled steps, validation, checkpoints, and version publication
- `Training/ReinforcementTrainer.swift` — bounded exploration, causal credit, compiled online updates, metrics, and RL snapshots
- `Core/MetalArrayBufferPool.swift` — pooled Metal shared-memory inputs handed directly to MLX
- `Agent/AgentRuntime.swift` — capture, inference scheduling, prediction latching, and teardown
- `Agent/InputInjector.swift` — final output firewall and synthetic HID events
- `Replay/InputReenactor.swift` — guarded demonstration replay
- `Core/GitHubReleaseUpdater.swift` — release discovery, verification, staging, and rollback installer
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

Recording transfer is producer-neutral. Import accepts a native `.atrrecord` directory, a folder containing packages, or an exported root containing `Recordings` and `recording-folders.json`. Validate every manifest, event file, video track, displayed dimension, and first-frame clock before changing the library. Stage the full batch, preserve video/event bytes, map folder metadata deterministically, regenerate colliding package IDs, and roll back the folder index plus every committed package if publication fails. Export writes the same portable root, includes only referenced folders, copies packages unchanged, verifies the copy, and never overwrites a non-empty destination.

Capture status matters: complete/started frames are usable, idle frames may reuse the last good frame, and blank/suspended/stopped frames are dropped.

New saved video requests a bi-planar video-range 4:2:0 ScreenCaptureKit surface and passes it directly to the required hardware HEVC Main encoder. The resolution/FPS-aware bitrate, six-second GOP, temporal compression, and frame reordering are a high-quality lossy storage contract; do not restore the former near-lossless-sized bitrate or force a full-resolution BGRA conversion without measurement. Live model capture remains BGRA because it is consumed directly rather than encoded. Existing/imported recording artifacts remain byte-preserving and are never migrated implicitly.

Current recording manifests persist the exact successfully encoded frame count. The field is optional for compatibility; Library presentation estimates a missing legacy count from duration and delivered FPS. Package size is the logical recursive byte total collected once during the workspace scan, never a per-row filesystem walk.

Capture-source discovery is a live list, not a one-shot bootstrap result. Refresh it on app activation, display topology changes, application launch/termination, and through a bounded two-second poll while a source picker is visible. Publish only changed, stably sorted snapshots, discard stale overlapping results, and do not disturb an active capture.

The input event tap is backed by a low-rate physical-state reconciliation pass. If macOS disables the tap or drops a transition, the next pass emits only the missing key, modifier, or mouse-button edges. Recording shutdown appends balanced releases for every logical control still held. The first complete screen frame seeds controls that were already held during asynchronous startup. Shortcut and blacklist filters apply to both ordinary and reconciled input.

Live HUD state and persisted recording state are intentionally separate. The HUD reduces blacklist-sanitized physical samples immediately, so a held modifier or mouse button is visible on its down edge; monotonic state revisions prevent cross-thread UI delivery from restoring an older state. The event file may briefly buffer modifier transitions to determine whether they belong to a global shortcut, but it preserves their original timestamps and separately held modifiers. Global shortcut suppression ends with the trigger/modifier release lifecycle; it must never consume an unrelated input merely because it occurs inside an expiry window. Caps Lock is normalized to a tap and Fn/Globe retains physical key edges so both remain in the ordinary keyboard action space.

Saved HEVC sessions end on the shared host clock even when ScreenCaptureKit reports only idle frames. Dataset construction materializes the last decoded frame at every configured perception interval until a newer frame becomes causal; a static screen must not collapse temporal spacing or observation count.

## Vision and dataset contracts

Packed observations are `UInt8`:

- grayscale: Y
- color: Y plus Cb/Cr using the selected 4:2:0, 4:2:2, or 4:4:4 layout

MLX expands each packed image independently to dense RGB-like channels at its native configured size. The visual encoder appends generated X/Y coordinate planes internally.

Packed 8-bit vision may normalize directly into BFloat16: exhaustive byte-value tests prove it is bit-identical to Float32 normalization followed by the established model-boundary BFloat16 cast. Do not generalize this shortcut to Float16; direct half-precision division changes ten of the 256 possible values by one ULP, so Float16 must continue through Float32 normalization unless a future quality cycle independently proves otherwise.

Temporal vision is either current-only or an explicit three-part input:

1. one current frame at the exact configured `PreprocessingSpec`
2. `pastFrameCount` real causal frames at `pastFrameSpec`, ordered oldest to newest
3. one complete 146-value control row paired with each past frame

`TemporalVisionConfiguration` owns past-frame count, spacing in Perception FPS intervals, and linear downscale. Its displayed seconds are nominal values derived from configured Perception FPS; dropped or delayed live perceptions can increase wall-clock separation. The reduced specification must copy color mode, chroma layout, bit detail, and resize policy from the current frame. Width and height use the same scale, with only unavoidable integer-pixel rounding. Never resize past images back to current resolution and never synthesize difference, optical-flow, or motion channels.

The default is four past frames, two perception intervals apart, with a 2× linear downscale. Bounds are part of the public safety/performance contract: 0–32 frames, 1–240 intervals, 1×–8× downscale, and at most 512 MB of compact cached embeddings plus controls. Packed historical images are not retained at runtime. Zero frames disables the temporal branch: runtime and training consume only current vision, and Policy v6 omits the recurrent/control-projection parameters and unused fusion columns.

`TrainingDataContract.schemaVersion` is 10. Dataset caches are disposable and must be invalidated when causal frame selection, frame-control pairing, static-frame cadence, current-only layout, or target meaning changes. A cache has:

- `current-observations.bin` — one full-resolution packed image per perception
- `past-observations.bin` — one reduced packed image of the same perception
- `observation-indices.bin` — current plus configured past indices for every action row; unavailable segment-leading frames use `UInt32.max`
- `frame-actions.bin` — one complete control row per perception interval
- `actions.bin` — action-rate targets; the immediately preceding target remains separate for transition weighting

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

`ModelContract.schemaVersion` is 6 and the weight format is `AgentTrainer.Policy.v6`. This is an intentional weight break. The migration archives older versions and checkpoints, resets the profile to the closest v6 preset, and keeps recordings available for cache reuse/retraining.

A runnable version is immutable. Runtime current vision, temporal vision, architecture, precision, channel semantics, cursor visibility, and demonstrated keys come from the version manifest, not mutable editor fields.

Dropout and `GeneralizationConfiguration` are stochastic training regularization, not part of the learned tensor layout. Editing only these values must preserve the active v6 brain and load its weights into the next run. Because the objective changed, the exact optimizer/checkpoint signature still includes them and begins a fresh optimizer sequence rather than claiming an invalid exact resume.

Current Policy v6 combines:

- one shared efficient visual encoder for real current and past images: a dense coordinate-aware stem, depthwise spatial stages, pointwise channel mixers, and a final compatible residual stage
- independent X/Y coordinate planes at each image's native resolution
- attention or legacy flattened visual pooling
- a learned projection that compresses each sparse 146-value frame-control row before a GRU/LSTM sequence over the paired past visual/control embeddings
- fusion of the current visual embedding with the final temporal state
- per-head bounded activations

Attention pooling shares its projection across current and past image sizes. Legacy flattened pooling remains decodable for profile repair/tests but is no longer offered by the current editor; it needs distinct projections when spatial dimensions differ.

`GeneralizationConfiguration` owns five training-only defenses. Structured YUV luminance/contrast/chroma variation and small neutral rectangles alter pixels without moving absolute pointer geometry. Per-value control-history dropout, rare binary flips, and bounded continuous noise simulate imperfect prior outputs. Temporal-token dropout trains startup/missing-history behavior. Small binary label smoothing reduces brittle confidence while validation retains exact targets. Visual-embedding and fusion dropout occur only after unique visual work is gathered back to action rows, preserving an independent stochastic path per label. Inference applies none of these perturbations.

Training uses class/transition weighting, focal binary loss, the v6 generalization pipeline, deterministic salience-balanced locality order, compiled MLX updates, bounded held-out evaluation, and adaptive cosine scheduling. Do not change those semantics without updating tests and the relevant schema.

The performance path preserves those semantics:

- VideoToolbox decodes directly into native video-range YUV. One CVMetalTexture mapping and Metal command buffer produce both configured output resolutions, while decode, resize/color packing, and cache writes overlap through a bounded ordered pipeline of up to eight observations within a 64 MB cap.
- Multiple recordings use up to four parallel decode/packing workers, bounded by active CPU count and a conservative half-of-unified-memory budget. Each worker produces a local shard; the builder merges shards in the user's original recording order and rebases only non-sentinel observation indices. Cache bytes, segment boundaries, and sample order remain deterministic.
- Before packing, a bounded metadata-only preflight validates each fixed-width input stream and opens the video track, duration, and dimensions. Unreadable packages are reported by recording name and identifier, left byte-for-byte untouched, and excluded from both packed data and the exact checkpoint signature. If every selected package is unreadable, training fails immediately with an actionable error.
- Projected cache plus peak shard working space is checked against the training-data volume before decode. Out-of-order packed shards use a strict two-worker-window look-ahead; a slow early recording must never permit temporary disk usage or pending merge state to grow with the full library size.
- When one decoded frame remains causal across many configured perception ticks, Metal packs it once and the cache writer repeats those exact packed bytes in large buffered writes. Observation cadence, indices, controls, and every action label remain unchanged. Temporary shards skip redundant durability barriers; the final cache files are synchronized before their atomic directory move.
- Dataset files are required memory mappings with adaptive VM advice. Epochs randomize small collections of temporally local lanes rather than every row globally, so each optimizer batch remains diverse while macOS can read ahead the contiguous mapped pages inside each lane. A batch gathers packed vision, frame controls, and target/previous-target rows once into pooled Metal shared memory; MLX expands packed `UInt8` vision inside the compiled graph.
- Whole-library grouping reads only each row's current-observation index. The cache validator guarantees that a global current-observation identity determines its complete causal sequence, so direct grouping must remain byte-for-byte order-equivalent to grouping through `visionBatchPlan`; the latter is still used for actual batches where its past-frame deduplication maps are consumed.
- Current images are encoded once per distinct temporal sequence. Overlapping causal windows additionally encode each distinct reduced-resolution past observation once, then gather those embeddings to their exact frame slots. Training-only pixel augmentation is sampled per unique encoded observation; visual-feature dropout and history corruption occur only after action-row gathering, preserving one independent post-encoder stochastic path per label. No label, loss term, update, or temporal control is removed, duplicated, averaged, or reused as another label.
- Above the measured workload crossover, the coordinate contribution to the first convolution is evaluated once per batch and the established GroupNorm layout uses MLX's fused Metal layer-normalization kernel. Small tensors retain the lower-overhead kernels.
- The six affine action heads retain their named Policy v6 tensors but concatenate those tensors into one Metal matrix multiplication. Reference tests compare both predictions and gradients against the six-call form.
- GRU/LSTM execution mirrors MLX Swift 0.31.3's gate equations but returns only the final state consumed by Policy v6, avoiding MLX's otherwise unused stack of every time-step output. Output and parameter-gradient parity tests are mandatory, and this local recurrence must be re-audited whenever the pinned MLX recurrent implementation changes.
- AdamW concatenates parameters, finite gradients, and both moment sets during the compiled update so identical elementwise work is issued as a few large kernels. Save/load splits moments back into the established checkpoint keys and shapes; the schedule, clipping rule, decay rule, and missing-gradient behavior are unchanged.
- Validation shares deterministic temporal work and head logits before gathering them back to every held-out label, then derives loss and predictions from that one compiled forward graph.
- CPU gathering overlaps a bounded queue of one to four strictly ordered Metal updates. Each compiled invocation consumes the lazy weights, AdamW moments, scheduler step, and MLX random state produced by the preceding invocation. The queue is unified-memory-aware, never crosses an epoch/autosave/run limit, stops growing on pause, and is fully evaluated before the checkpoint cursor advances.
- Live inference encodes the full current frame plus one current reduced frame. It returns that reduced embedding with the prediction, stores only the compact Float32 embedding and sanitized control row in the circular history, and feeds cached embeddings directly into later temporal steps. Visual-encoder work is therefore independent of `pastFrameCount`.
- Live diagnostics publish a rolling pipelined-step time, mapped-input time, peak-throughput retention, MLX active/cache/peak memory, and macOS thermal pressure. Metrics are rate-limited and chart histories are presentation-bounded so observing a run does not grow UI work with optimizer-step count.

Do not add asynchronous stateful work unless its production path is covered by ordered-update parity tests and bounded at every persistence boundary. Exact resume means a saved checkpoint must continue its own versioned sampling contract, weights, moments, scheduler, and random state without skipping or replaying an update.

Performance work may change graph operation order, grouped sample order, and low-bit numerical trajectories. “No quality loss” is evaluated by matched-update training loss, held-out validation loss, and per-head validation behavior—not byte-identical weights or predictions. The hardware benchmark gates relative validation-loss drift at 1%, while focused tests compare the optimized objective and gradients to the reference graph. Never obtain speed by silently reducing vision resolution, precision, model capacity, labels, loss terms, or update count.

For a longer power/thermal comparison of the optimized and reference 16-frame paths, run `AGENTTRAINER_BENCHMARK_STRESS_STEPS=2048 AGENTTRAINER_BENCHMARK_STRESS_BASELINE_STEPS=1024 ./benchmark.sh`. Compare first/last-window retention as well as thermal state and cache memory; a coarse nominal macOS thermal state does not prove that sustained GPU clocks remained at their initial boost level.

An exact checkpoint contains weights, both AdamW moment sets, optimizer metadata, scheduler state, training state, model schema, and MLX random state. Missing or mismatched moment pairs/random state are corruption, not a silent fresh start.

Checkpoint and version activation are transactions. A profile must never point at one brain while its checkpoint resumes another. Weights-only activation intentionally removes an unrelated resumable checkpoint so later training begins a fresh optimizer from the selected brain.

Completed training publishes an immutable version. With honest held-out data, the lowest acceptable validation brain is activated; periodic autosaves are bounded and never prune the active or protected brain.

## Online reinforcement learning contracts

`ReinforcementLearningContract.schemaVersion` is versioned independently from Policy v6 and demonstration caches. RL deliberately uses the same Policy v6 parameter layout: an existing active version is loaded without alteration, while a brand-new AI receives neutral action-head weights and a sparse binary prior before its first inference. Enabling or disabling RL is not a learned-brain contract change.

Every manual or automatic provider emits a `ReinforcementSignal` containing a monotonic action-time timestamp, a finite signed value, and a source identifier. Values are clamped to ±100 at the runtime boundary. Hotkeys, mouse buttons, Run-page controls, exact modifier-scroll steps, and future screen/OCR/game-state detectors all enter this one interface. Providers own detection only; the runtime owns credit assignment, optimization, safety, persistence, and metrics.

Runtime inference, transition recording, feedback application, and MLX optimization share one serial inference queue. A feedback timestamp selects only transitions at or before the signal inside the configured causal window. The newest decision receives full credit and older decisions receive exponential decay. By default, inactive binary controls and negligible relative/scroll outputs are masked out; explicitly enabling inaction learning changes that behavior. Transition storage is capped by configured frames, elapsed credit time, and a 256 MB memory ceiling. The pending signal queue is capped at 256 entries and preserves the bounded signed total when high-rate feedback must coalesce.

The online objective uses the stored behavior logits and executed, safety-sanitized action. It applies a clipped likelihood ratio, behavior-policy anchor, binary entropy bonus, fixed action-time exploration distribution, finite checks, and global gradient clipping. The compiled step must force evaluation of its objective, model parameters, and optimizer state together before metrics or a snapshot can observe it. No online update may race inference or input injection.

RL persistence is separate from the exact demonstration checkpoint. Each published RL brain contains Policy v6 weights, RL AdamW state, its optimizer identity, exploration random state, cumulative counters/reward, and training time. Publication is an immutable transaction; stale asynchronous autosaves cannot replace a newer sequence, and the active profile is updated only after every required artifact verifies. Activating an RL brain removes the former demonstration checkpoint transactionally so a later Training Data run warm-starts from current RL weights rather than resuming stale pre-RL parameters. Returning to RL restores its optimizer only when the saved optimizer identity matches the current stability/exploration configuration; otherwise only optimizer state restarts and brain weights remain intact.

A brand-new AI may emit only keyboard keys explicitly enabled in its RL keyboard firewall. An existing brain uses the union of demonstrated and RL-enabled keys. Profile restrictions, live cursor/keyboard permissions, configured control regions, and input-injection safety still take precedence. Periodic snapshots require at least one successful update; after any successful session learning, stopping publishes a final named brain even if the latest state was also covered by an autosave.

## Runtime safety contracts

The final executable action is the intersection of:

1. immutable capabilities learned by the saved version
2. current profile channel disables
3. profile key/button/modifier restrictions
4. demonstrated or explicitly RL-enabled key codes
5. live cursor and keyboard permissions
6. the allowed control region unless full-Mac control is enabled

Predictions must contain 146 finite values. Held keys/buttons may persist between fresh predictions, but cursor deltas and scroll are transient. The prediction latch consumes transient outputs once.

At runtime, every successfully inferred perception stores its compact reduced-frame visual embedding with the sanitized controls predicted for that interval. Exact spaced entries are selected from a fixed-capacity circular buffer. Unavailable startup slots use zero embeddings and controls; training's whole-token dropout explicitly covers that state. Sampling/appending remains O(1), and historical images are never re-encoded.

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

Library recording rows and inspectors expose encoded frames plus logical package MB/GB. Folder headers aggregate every recording in that folder—not only search results—and show total recordings, frames, logical bytes, and full recorded seconds. Newly created manifests use exact frame counts; legacy estimates must remain compatible and non-destructive.

Recording presets are user-owned, editable snapshots of Record-page settings. Applying a preset must tolerate an unavailable source or deleted destination without inventing an identifier. Library range selection follows the visible folder/search order; bulk manifest moves roll back as a set, and bulk deletion stages all recording directories before cleanup. Global shortcuts may use keyboard keys or any macOS mouse-button number, including additional side buttons, and must remain excluded from recording and human-input safety hooks.

## Validation and release

Before release:

1. Run `./test.sh`.
2. Run `./benchmark.sh` and compare original/optimized throughput plus matched-update training and validation loss on the same idle Mac. Check every sustained window for stable throughput, bounded MLX cache, and nominal thermal pressure.
3. Run `git diff --check`.
4. Confirm `CFBundleShortVersionString` and `CFBundleVersion`.
5. Quit AgentTrainer and run `./build.sh`.
6. Verify `/Applications/AgentTrainer.app` launches and reports the intended version.
7. Exercise permission refresh, recording start/stop, training pause/resume, agent panic, and updater UI.
8. Verify the DMG and `SHA256SUMS.txt` in `outputs`.
9. Confirm no machine-specific paths, signing identities, generated files, or user data entered the source archive.

`build.sh` creates a fully signed staging bundle, transactionally installs it into `/Applications/AgentTrainer.app` by default, and then packages the DMG/source/checksums. Override the install path with `APP_PATH`; override signing with `CODE_SIGN_IDENTITY` and optionally `CODE_SIGN_KEYCHAIN`. Restricted builders without the macOS disk-image helper may set `SKIP_DMG=1` to emit a resource-safe signed `.app.zip`; public releases must still use and verify the DMG path.
