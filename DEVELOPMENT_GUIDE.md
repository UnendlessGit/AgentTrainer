# AgentTrainer development guide

This document is the durable handoff for future AgentTrainer development chats. Read it before changing capture, input, training, storage, or runtime behavior. The implementation is original; MimicClone was consulted only for lightweight UX cues such as collection-oriented browsing and choosing absolute cursor versus game-camera behavior at runtime.

For the complete Run-tab cursor/keyboard output-permission contract and the verified procedure for updating an already-authorized app without recreating its bundle, also read `IN_PLACE_UPDATE_GUIDE.md`.

## Product invariants

- The app is local-only for user data: no accounts, telemetry, cloud training, uploads, or Python runtime. Its sole network path is the public GitHub Releases update check/installer documented below.
- Native stack: Swift, SwiftUI, AppKit, ScreenCaptureKit, AVFoundation/VideoToolbox, Metal, and MLX Swift.
- Deployment floor: Apple silicon running macOS Sequoia 15.0. The source and opaque custom chrome support Sequoia 15, Tahoe 26, and macOS 27. Hotkeys use native registration when Carbon is importable and compile an AppKit fallback otherwise, avoiding a hard future-SDK dependency without adding Accessibility to today's recording-only workflow. The 2026-07-17 baseline was built with Swift 6.3.3/Xcode 26.6/SDK 26.5 and exercised on macOS 27 beta. Xcode 27 was not installed on that build Mac; also compile with it before a public macOS 27 release whenever that toolchain is available. MLX Swift is pinned exactly to stable 0.31.3.
- Live vision must use the profile's exact preprocessing width, height, bit depth, color mode, chroma subsampling, and resize policy.
- Dataset and live frames share `VisionPreprocessor` and the same packed UInt8 Y/Cb/Cr representation.
- Recording and agent streams exclude AgentTrainer windows, the menu extra, PIP, and input HUD through the ScreenCaptureKit content filter.
- Global shortcuts never become training data and never count as human interruption input.
- Stop and panic are transactional: prevent new output first, drain queued work, release held inputs, neutralize relative mouse state, then report stopped.
- Each immutable brain stores the keys present in its raw training events. Runtime keyboard and modifier output is intersected with that set after all other restrictions.
- A training pause is immediately runnable and exactly resumable, including optimizer and random state.
- One AI may train while a different AI runs. A single profile cannot train and run simultaneously.
- Mouse training is mode-independent. Both normalized position and raw delta heads learn from the same demonstration; Run chooses the execution head.
- Run's default Auto mode diagnoses the selected recordings and resolves to Absolute Cursor or Game Camera before the injector starts. Never pass Auto through to `InputInjector` unresolved.
- Game-camera delta semantics are resolution-independent: dataset values are raw HID delta / 80 and runtime reverses the same scale before applying user sensitivity. A delta is an additive action, not held state, and may be executed only once per published prediction.
- A frame's training target is the demonstrated action interval immediately after that frame. Inputs at or before the trim start prime held state and pointer position, but their accumulated movement/scroll is cleared before the first target.
- Policy input is the current packed frame expanded to dense planes, its signed difference from the exact preceding perception frame, and explicit X/Y coordinates. Training and runtime must advance the predecessor only when a new perception is processed.
- Ground-truth action history is never allowed to become the sole solution: every training sample independently masks the complete history branch with 50% probability—even when ordinary feature Dropout is zero—while inference receives normal history.
- Runtime recurrent history contains the action semantics that could actually execute: continuous outputs are bounded, binary controls are thresholded, transient outputs are zero on stale ticks, and disabled, restricted, duplicate, or never-demonstrated keyboard paths remain zero. Raw policy probabilities must not become an unguarded hidden-state channel.
- Crystal V4 and Crystal V4 Fine-tuned + glass are user-protected profiles: autosave pruning, version deletion, learning reset, and profile deletion must leave compatible artifacts intact. Protection becomes sticky when either is saved, so later renames remain safe. A model-schema migration must inspect each version manifest independently, preserve compatible artifacts, and archive incompatible or unreadable artifacts without deleting them; protected profile records and all source recordings remain intact.
- Duplication is a deep profile clone: versions, active version, training/timing summary, and Checkpoint must be copied into the new profile namespace. A duplicate is never deletion-protected, even when its source is protected.
- Exact vision or `ArchitectureSpec` edits on a trained profile require explicit destructive confirmation. `WorkspaceStore.resetLearning` clears Versions/Checkpoint/active progress while retaining source recordings. Protected AIs may not cross this boundary; duplicate them first.
- Persistent diagnostics must never contain high-rate per-frame logging. Errors and important lifecycle boundaries go to OSLog and bounded JSONL storage.
- Training-data and model-library locations are independently persisted. Relocation is forbidden during any active workflow, never merges two populated libraries, and never deletes the source until an empty destination has been copied, verified, selected, and prepared.

## Source map

- `App/AgentTrainerApp.swift`: main window, commands, menu-bar status/control, termination coordination.
- `App/AppModel.swift`: main-actor workflow coordinator and persisted workflow settings.
- `Core/Domain.swift`: storage/domain contracts and compatibility defaults.
- `Capture/CaptureService.swift`: persistent ScreenCaptureKit streams and capture-source geometry.
- `Capture/HEVCWriter.swift`: hardware-required realtime HEVC writer.
- `Capture/InputCaptureService.swift`: listen-only input event tap, shortcut suppression, recording key filtering, and run-loop lifecycle.
- `Preprocessing/VisionPreprocessor.swift`: shared BGRA/native bi-planar YUV Metal packing kernel and MLX packed-tensor expansion. Recorded H.264/HEVC requests VideoToolbox's video-range NV12 surfaces; the kernel honors their tagged Rec.709/601/2020 matrix so cache construction avoids a full-resolution BGRA round trip while live BGRA frames retain the same output contract.
- `Training/DatasetCache.swift`: reusable memory-mapped observation/action cache and action accumulation.
- `Training/PolicyNetwork.swift`: convolutional encoder, recurrent history encoder, output heads, loss, and resumable AdamW.
- `Training/TrainingEngine.swift`: detached GPU training workflow, exact checkpoints, validation, bounded metrics, and runnable versions.
- `Agent/AgentRuntime.swift`: live capture, perception scheduling, compiled inference, independent action timer, focus safety, and teardown.
- `Agent/InputInjector.swift`: tagged synthetic events, output restrictions, runtime mouse mode, held-state transitions, and hard disable/release.
- `Replay/InputReenactor.swift`: guarded recording replay with generation-token cancellation.
- `Storage/WorkspaceStore.swift`: actor-isolated workspace operations, independent library locations, verified relocation, and atomic JSON replacement.
- `Storage/InputEventFile.swift`: fixed-width binary event format plus full and summary readers.
- `Core/AppLog.swift`: OSLog bridge, bounded persistent JSONL entries, support-report copy, and crash-report discovery.
- `Core/MLXMemoryLifecycle.swift`: process-wide unified-memory/cache limits plus lifecycle-boundary cache reclamation.
- `UI/LibraryView.swift`: expandable recording folders, video inspector, streamed input/mouse diagnostics, and fixed-height timeline preview.
- `UI/VisualKeyboard.swift`: reusable keyboard used by Library and live AI input HUD.
- `UI/InputHUD.swift`: capture-excluded recording/AI input keyboard and mouse state; exact-vision PIP is enabled only for AI runtime.
- `UI/CNNVisualization.swift`: bounded CNN heatmap/channel rendering and newest-only background coalescing for the capture-excluded HUD.
- `UI/RootView.swift`: deterministic solid top bar/sidebar chrome, native traffic-light integration, navigation, and global status/safety controls.
- `UI/UIAppearance.swift`: balanced palettes plus bounded persisted corner, surface, accent, sidebar-width, and interface-motion controls.
- `UI/Pages.swift`: remaining feature pages and parameter editors.
- `IN_PLACE_UPDATE_GUIDE.md`: run-only output firewall design, live-toggle race guarantees, regression checks, macOS TCC/code-identity rules, and incremental in-place app replacement.

## Persistent workspace

Runtime data never moves with the source checkout. By default all data lives in `~/Library/Application Support/AgentTrainer`; Settings may independently point the training-data and model-library roots at local folders or mounted external disks. The fixed support root always owns logs, while each selectable root owns only its managed entries:

```text
Application Support/AgentTrainer/                 # fixed support root
└── Logs/
    ├── app.jsonl
    └── app.previous.jsonl

<selected training-data root>/                    # default: support root
├── Recordings/<uuid>.atrrecord/
│   ├── manifest.json
│   ├── capture.mov
│   ├── events.atrevents
│   └── thumbnail.jpg
├── Caches/<sha256>.atrcache/
│   ├── manifest.json
│   ├── observations.bin
│   ├── observation-indices.bin
│   └── actions.bin
└── recording-folders.json

<selected model-library root>/                    # default: support root
├── Profiles/<profile uuid>/
│   ├── profile.json
│   ├── Checkpoint/
│   │   ├── weights.safetensors
│   │   ├── best.weights.safetensors              # after validation
│   │   ├── optimizer.safetensors
│   │   ├── random.safetensors
│   │   └── state.json
│   └── Versions/<version uuid>/...
└── model-contract.json
```

The two selected root paths are persisted separately. `WorkspaceStore.prepare` validates that each location is available and writable before creating or reading managed directories; a disconnected `/Volumes/...` location surfaces an actionable error instead of silently recreating a different library. Moving to an empty root copies into a private staging directory, verifies logical byte/file summaries item by item, moves staged items into place, commits the new root, and only then removes the old items. A failure can leave duplicate data but cannot leave the only copy half-migrated. A populated target is an explicit library switch and never a merge.

For launch/UI smoke tests only, set `AGENTTRAINER_WORKSPACE_ROOT` to an absolute disposable directory. `WorkspaceStore.shared` then ignores persisted external-library paths and keeps support data, recordings, caches, models, and logs beneath that root. This prevents a schema-migration test from touching the user's real library; production launches leave the variable unset.

`events.atrevents` begins with `ATREVT01`, a UInt32 version, then 72-byte little-endian records. Readers reject unknown kinds, truncated records, non-finite values, and decreasing timestamps before use. Keep old manifests decodable by using optional fields or explicit custom decoding when a required field is introduced. The cache key includes the complete recording manifests, preprocessing, rates, and history length; trims and recording exclusions therefore invalidate derived caches automatically.

Strict recording validation must run only after `repairInvalidRecordingManifests`. The recovery pass enumerates `.atrrecord` directories directly so malformed-but-decodable legacy duration/trim metadata cannot hide a recording before repair. It derives the real video duration, clamps safe metadata, retains `manifest.pre-1.8.1-recovery.json`, and must never alter or remove the video or event stream. Training/model storage destinations inside an `.app` bundle are rejected because bundle replacement during an update would replace that directory.

## Recording pipeline

1. `AppModel.startRecording` freezes the selected `CaptureSpec`, global capture rectangle, trim settings, destination folder, and recording key blacklist.
2. `InputCaptureService` starts as a listen-only `.cgSessionEventTap`. It drops AgentTrainer synthetic events, complete configured shortcut chords, and blacklisted keys. Excluded physical modifier keys also remove the corresponding modifier flag.
   The start call does not return until the dedicated tap run loop is ready. Redundant `keyboardEventAutorepeat` key-down events are dropped because held state already represents them.
3. `CaptureService` starts ScreenCaptureKit and hardware HEVC. The first usable `started` or `complete` frame's host-clock PTS becomes `hostStartNanos`; inputs before it are discarded. `idle` means the source is unchanged, not failed, while blank/suspended/stopped statuses remain unusable. At the first-frame boundary, recording writes a zero-delta cursor snapshot before opening the input clock so every dataset has a correct initial absolute position. Legacy datasets prime position from their earliest pointer-bearing event, and mouse-button/scroll events also refresh position.
4. Stop disables and joins the exact event-tap session's run loop, then stops/drains ScreenCaptureKit and finalizes HEVC. Each session owns its own completion group; a timed-out older thread may never signal or clear a later session.
5. `trimStart` is seconds removed from the beginning. `trimEnd` is the absolute retained endpoint (`duration - requested tail trim`). Video remains intact; dataset/replay apply the range.
6. The menu-bar extra indicates recording status. A capture-excluded floating input HUD shows the human keyboard/mouse state without vision; the vision PIP remains AI-run-only.

Recording startup is revision-tokened. Stop, panic, quit, and the global Record shortcut may cancel an in-flight ScreenCaptureKit start; cancelled startup drains capture/input and deletes only its incomplete recording directory. The recording key blacklist and Record-shortcut modifier mask are frozen before the first await so UI edits cannot alter a session halfway through startup.

## Vision and action contract

Packed observation layout:

- grayscale: `Y`, width × height bytes;
- 4:2:0: `Y + Cb(w/2,h/2) + Cr(w/2,h/2)`;
- 4:2:2: `Y + Cb(w/2,h) + Cr(w/2,h)`;
- 4:4:4: `Y + Cb(w,h) + Cr(w,h)`.

Values are quantized to the requested 1–8 bit levels and expanded directly in MLX. `ActionLayout` has 146 floats:

- 0–1 normalized absolute mouse;
- 2–3 raw mouse delta divided by the fixed `GameCameraContract.deltaScale` (80), clipped to -1...1;
- 4–11 eight held mouse buttons;
- 12–13 scroll;
- 14–141 simultaneous held keys 0–127;
- 142 Shift (owned by the Keyboard channel in training-data schema 7+).
- 143–145 Control, Option, Command (owned by the Modifiers channel).

macOS also emits Command, Option, and Control as ordinary key codes 54/55, 58/61, and 59/62. These six duplicate slots remain in the fixed 146-value tensor for model compatibility but are excluded from binary-output accounting, training targets and history, keyboard loss, CNN focus, runtime history, and injection. The dedicated 143–145 outputs are their only active path, so disabling Modifiers hard-blocks them for every model generation. At Run startup, the current profile's disabled Modifiers setting intersects the immutable brain manifest; it can turn an old brain's modifier head off but can never turn an untrained head on. Shift key codes 56/60 and output 142 intentionally remain owned by Keyboard.

The policy always contains both mouse heads. `ActionChannels.mouseMovement` is the semantic training toggle and normalizes legacy absolute/relative profile values. `MouseControlMode` exists only in Run settings. Auto is resolved from `InputEventReader.MouseDiagnostics`, which distinguishes locked-cursor game-camera recordings from moving absolute cursors and reports coordinates outside the frozen recording rectangle. This prevents normalized absolute coordinates from being misused as one-sided movement by a game that continually locks its cursor.

Policy v4 concatenates the current dense Y or Y/Cb/Cr planes, the signed `current - previous` planes, and normalized X/Y channels before the first convolution. Four default GroupNorm/SiLU stages use a stride-four stem and reach a 63-pixel receptive field with a cumulative stride of 32. The remaining spatial grid is flattened into the visual projection instead of globally averaged, so location and layout survive into the action heads. The exact output geometry is shared by `AgentPolicy`, `CNNGeometry`, and `ModelSizing`. The deterministic coordinate grid is deliberately underscore-prefixed so MLX module reflection does not treat it as a trainable/saved tensor or allocate AdamW moments for it; regression tests compare the complete reflected GRU and LSTM parameter trees with `ModelSizing`.

Live CNN inspection is presentation-only and does not change the learned-brain contract. `AgentPolicy.visualActivations` exposes normalized post-SiLU spatial stages without adding modules or parameters. Activation Overlay compiles one lazy graph per selectable layer and reduces only the chosen stage; Feature Channels ranks on the GPU and transfers at most 16 spatial maps; Action Saliency computes positive Grad-CAM influence for a selected action head from the exact unsampled final convolution. Spatial copies are capped to a 96-pixel longest side. The normal compiled prediction function remains the only evaluated graph while inspection is disabled.

The AI Models editor derives its live **Input size check** through `NeuralInputSizing`, which must remain synchronized with `VisionPreprocessor`, `AgentPolicy`, `AgentRuntime`, `ActionLayout`, and `ModelSizing`. The default view compares complete per-decision input values with learned parameters and uses conservative usability bands: at most 0.75 inputs per parameter is Comfortable, at most 2 is Balanced, at most 5 is High, and anything larger is Too high. This ratio is intentionally only a simple warning guide—not a validity limit or training-quality prediction—because convolutional parameters are shared across image positions.

The collapsed technical details distinguish packed source values from the dense values processed by the network: chroma subsampling reduces packed cache/live-frame values, MLX expands color to full-resolution Y/Cb/Cr, signed temporal differences add one more dense set of visual planes, the policy appends two full-resolution coordinate channels, and the recurrent branch receives `max(1, historyLength) × 146` values. A zero history uses one all-zero row to preserve a valid recurrent shape. Bit depth changes quantization levels and meaningful source bits, not value count; resize policy changes framing; enabled control channels change loss/output behavior; architecture widths change parameters; and batch size repeats the complete per-decision input. Perception FPS determines the maximum new network-input rate and may not exceed Action FPS. Action FPS determines the history time span and may maintain held state against a reused perception; transient relative-mouse and scroll slots are zero on action ticks without a new prediction. The displayed precision payload is nominal input-only storage and must not be described as total MLX or unified-memory use.

Model contract schema 4 is the temporal, normalized spatial Policy v4 input and intentionally cannot load older weights. Training-data schema 7 retains exact preceding-perception indices and sub-tick control pulses while moving Shift into the Keyboard channel; Command, Option, and Control remain in the Modifiers channel. Old immutable brains keep their original modifier routing, while a new or fine-tuned schema-7 brain uses the new routing. The 1.8.2 model-artifact audit never trusts only the library-wide marker: `WorkspaceStore.removeObsoleteModelArtifacts` decodes every version manifest, retains current-schema versions byte-for-byte, and writes a schema marker into known-compatible checkpoints. Incompatible, unreadable, or uncertain artifacts move into each profile's `Archived Model Artifacts` recovery directory instead of being deleted. Only profiles with no compatible version or checkpoint are deactivated and mapped to the new presets; every profile and source recording remains intact. The separate audit marker makes this repair run once even on libraries already stamped by 1.8/1.8.1. Never broaden this migration to source recordings.

## Training lifecycle and performance

- `TrainingEngine` owns one detached training task. `AppModel.trainingProfileID` identifies its immutable captured profile while UI selection may move to another AI.
- Each training run gets its own `MLXRandom.RandomState` via `withRandomState`. This prevents inference model initialization from changing dropout order and preserves exact simultaneous train/run behavior.
- Forward, backward, AdamW, module state, optimizer state, and run-local random state are one compiled MLX graph.
- AdamW uses a persisted adaptive linear warmup of one epoch, bounded to 10–500 steps, followed by inverse-square-root decay; global gradient norm is clipped to 1. This reaches the configured learning rate promptly on small datasets without removing the stabilizing ramp on large runs. Optimizer metadata preserves the chosen value, and checkpoints from the fixed-schedule trainer intentionally default to their historical 500 steps.
- Each training task gets a dedicated MLX GPU stream, allowing Metal to schedule it independently from simultaneous inference without changing math, precision, sample order, or optimizer state.
- Each step calls `asyncEval` for loss, model, and optimizer outputs, gathers the next mapped CPU batch while Metal executes, then performs a full `eval` barrier. This overlaps CPU and GPU work without changing update order or exact-resume state.
- Cache construction traverses validated memory-mapped event records and uses multi-megabyte buffered sequential writes. Each perception image is stored once; each action row stores two little-endian UInt32 indices for its current and exact preceding perception. This removes duplicated observation payload when Action FPS exceeds Perception FPS without changing samples.
- Packed observation gathering writes directly into pre-sized current/previous `Data` buffers with bulk `memcpy`; it never grows an append buffer row by row.
- Epoch order is deterministic and salience-balanced: every training row still appears exactly once, while binary transitions and active relative-mouse/scroll rows are round-robined across fixed-size batches before ordinary rows fill the remaining slots. The sampling-strategy version is checkpointed. A legacy checkpoint paused mid-epoch finishes with its historical uniform order and upgrades only at the next epoch boundary.
- Fixed-shape batch temporaries live inside an autorelease pool. `MLXMemoryLifecycle` applies the same process-wide policy before both inference and training: cache is bounded to 6% of physical memory with a 2 GiB ceiling, and the MLX allocation limit retains at least 15% and normally at least 2 GiB for macOS. Unused MLX cache is explicitly reclaimed after model/compiled-graph release at every run, failed-start, and training boundary. Clearing cache is safe during simultaneous work because active tensors remain allocated. Profile validation uses a conservative parameters/optimizer/activation working-set estimate before training starts.
- Dataset construction writes completed Metal output bytes directly into a bounded sequential writer. Do not reintroduce an intermediate `Data` allocation per perception frame or request BGRA from `AVAssetReaderTrackOutput`; both multiply memory bandwidth on full-resolution recordings. Packing progress is weighted by usable trimmed duration and updates within each recording.
- UI metrics publish at most four times per second and report rolling throughput. Loss history drops old data in large chunks rather than shifting on every step. Published chart suffixes are detached copies, preventing Swift array copy-on-write from copying the complete checkpoint history on the next optimizer append.
- Training charts render only the selected suffix, retain extrema during bounded downsampling, and draw smooth Canvas curves. Observe the snapshot values, not only `values.count`: published suffix lengths remain fixed at their cap while new losses shift in. Active charts morph between resampled snapshots for 160 ms, then become completely idle until the next metrics publication. The Training page is unmounted on another tab; while the app is inactive, its chart keeps a stable snapshot instead of accepting render updates.
- `CheckpointState.elapsed` is cumulative optimizer wall time. `CheckpointState.experienceSeconds` increments by `batchSampleCount / actionFPS`, measuring how much real demonstration time optimizer batches have consumed across epochs. Both values flow into immutable versions and the cheap profile progress summary; older profiles decode with optional fields and receive a stable `globalStep * batchSize / actionFPS` estimate until their next exact checkpoint.
- `CachedDataset` uses mapped files and bulk copies. Segment lookup is binary rather than repeatedly scanning recording segments.
- Binary controls use training-split positive weights up to 1024×, transition weights, and a zero loss mask for unseen/blocked keyboard and modifier outputs. Active relative-mouse and scroll targets use weighted Smooth L1; absolute cursor uses Smooth L1. This prevents the all-zero keyboard shortcut from dominating BCE while retaining robust continuous losses.
- Whole recordings are held out where possible. A segment cannot enter validation if doing so removes the only training example of an enabled, permitted binary output; blocked/disabled controls cannot erase validation availability or consume its representative budget. With only one recording, the temporal boundary is extended until validation's full recurrent history and both perception indices are disjoint from training; irregular/static frame delivery is checked through actual observation indices rather than an FPS estimate. If no honest held-out tail remains, all samples train and validation is disabled. Evaluation uses one fixed class-aware subset (rare positives first, then transitions, active deltas, and even timeline coverage), not the first contiguous batches.
- The latest exact checkpoint and lowest-validation-loss weights are separate. Epoch-end validation updates `best.weights.safetensors`; completed training activates a weights-only Best Brain while the latest optimizer checkpoint remains the continuation source. Explicitly activating a weights-only brain clears a stale checkpoint so later training safely fine-tunes that brain with a fresh optimizer. Before the first update, those selected weights are evaluated on the current run's held-out set and captured as the comparable baseline; a saved score from different recordings, targets, split, or loss semantics is never compared directly. The validation-strategy version is checkpointed independently, so a contract upgrade recalibrates the comparison while retaining compatible optimizer state; if the current split has no honest validation rows, stale validation/best metadata is cleared.
- Maximum Steps and Autosave Steps are `TrainingRunSettings` persisted in workflow settings, not profile/model identity.
- Epochs is a repeatable training-block size, not a lifetime ceiling. `CheckpointState.epochGoal` persists the current goal: an interrupted block resumes toward the same goal, and a new start after reaching it adds `TrainingConfiguration.epochs`. Maximum Steps is added to the restored global step as a per-session budget rather than compared as an absolute lifetime step.
- Autosave Steps means updates performed during the current run: the next save is `restoredGlobalStep + autosaveSteps`, not the next global modulo boundary. Epoch boundaries save the exact checkpoint but do not publish extra visible autosaves. Only periodic/manual/final lifecycle points publish versions.
- If an exact checkpoint signature no longer matches but the active runnable version has the same current model schema, preprocessing, and architecture, training warm-starts from those immutable weights with a fresh optimizer and batch boundary. Never silently fall back to random weights solely because the training-data contract changed.
- The newest ten autosave versions are retained per AI. Completed versions and the active version are preserved; both protected AIs are excluded from pruning. `AIProfile.trainingProgress` stores the cheap step/epoch/version-count/timing summary used by model dropdowns and list rows.
- MLX memory fields are allocator metrics on unified memory: active arrays, reusable cache, and process-lifetime peak. They are not dedicated VRAM measurements.

Exact checkpoint identity includes preprocessing, normalized channels, learning configuration/architecture, complete selected recording manifests, folder IDs, and output restrictions. Epoch count, maximum step budget, and autosave interval intentionally do not invalidate a checkpoint. Checkpoint state separately stores recording segment order because a set-equivalent recording selection can still map sample indices differently; an order mismatch warm-starts safely from the runnable brain instead of attaching a batch offset to different rows.

## Runtime and cleanup

Runtime startup loads an immutable model version, compiles prediction, starts the safety event tap, optionally focuses the target window, starts a persistent capture stream, then starts the independent action timer.

- Newest Frame mode coalesces to one pending `CVPixelBuffer`.
- Every Frame mode applies backpressure on the capture callback instead of enqueueing unbounded inference work.
- Prediction state reuse is intentional when action FPS exceeds perception FPS, but only idempotent state may be reused: keyboard/button holds and absolute position remain available, while relative-mouse and scroll deltas are consumed once and recorded as zero on later stale ticks. History stores thresholded controls after channel, restriction, demonstrated-key, and live-output-permission filtering, matching executed action semantics instead of feeding unconstrained logits back into the recurrent branch.
- Timing, architecture, channels, cursor visibility, and Auto mouse mode come from the immutable selected version rather than mutable editor/Record-tab fields. App-level startup is single-flight before its first await, reserves the profile against simultaneous training, and has a revision token. Stop during version lookup, diagnostics, or weight loading waits for that launch to unwind, so repeated Run requests cannot overwrite the only tracked runtime or leave an orphan injector. Runtime startup rechecks cancellation around safety-monitor, capture-stream, and action-timer installation; after joining teardown, its failure path removes any resource that started in the narrow interval after an earlier stop pass.
- Runtime feeds the exact previous successfully preprocessed perception into Policy v4. A ScreenCaptureKit `idle` sample reuses the last usable pixel surface for a new inference, producing a correct zero temporal difference on an unchanged screen. It never suppresses the safety watchdog: if MLX itself stops publishing, a prediction older than `max(350 ms, 3 / Perception FPS)` stops the runtime and releases every held output. Blank, suspended, and stopped frames are not reused.
- CNN diagnostics have an independent 0.5–15 FPS cap. Diagnostic inference replaces—not duplicates—the normal forward pass on due activation/channel frames. Grad-CAM reuses the exact final feature tensor and runs only the post-CNN gradient work. CPU image rendering is newest-only on a utility queue and cannot delay inference or actions.
- The exact-vision PIP converts packed Y/Cb/Cr bytes directly into its bitmap on the coalescing utility queue. Do not reintroduce an intermediate full-resolution Float array per preview frame.
- Action history is a fixed ring buffer, not an array shifted at every action.
- `InputInjector.enable()` begins a session. `execute` holds its lifecycle lock while posting, so `disableAndReleaseAll()` cannot race a late action.
- Run-only `RuntimeOutputPermissions` are independent of trained channels and profile restrictions. `InputInjector.updateOutputPermissions` uses the same lifecycle lock as `execute`; disabling keyboard output immediately releases held keys/modifiers, and disabling Game Camera movement posts a neutral zero-delta event.
- Runtime startup and permission updates are also ordered by `AgentRuntime`'s lock. A toggle changed while weights are loading must win over the older value captured at the start of the async launch.
- Game Camera uses a `.hidSystemState` event source, clips pathological output, optionally warps to the capture center before and after each delta, and multiplies predictions by the same fixed 80-pixel scale used by training. Nonzero raw deltas always use `.mouseMoved`, even while a button is held; button edges remain separate events and rounded zero deltas are not posted. Reenactment automatically uses the same locked-camera path when diagnostics detect it.
- Stop marks runtime stopped and cancels the action source, drains the action queue, then disables the injector, releases keys/buttons, associates the cursor, and posts a zero-delta move. It next disables/joins safety input. Only after physical input is neutral does it stop/drain capture and the potentially uninterruptible inference queue, release model/compiled graphs, clear reusable MLX cache, and report stopped. Concurrent Stop/panic/failure callers join the same teardown continuation and cannot return early.
- Capture startup failure, stop failure, and unexpected ScreenCaptureKit termination all converge on that same teardown path; stream delegate errors may never be ignored while the action timer remains live.
- Reenactment uses a generation token so a cancelled task cannot post after a newer session or after Stop.
- Every synthetic event carries `agentTrainerSyntheticTag` and is ignored by recording/safety taps.

The action queue must drain before release so an already-entered `execute` cannot post afterward. Do not move physical release behind capture or inference drainage: an MLX evaluation is not cancellable and may take seconds, during which a held key/button or relative cursor state must already be neutral. The UI must continue to show Starting/Stopping until the remaining drains and every coalesced Stop caller finish.

## UI state and responsiveness

- `AppModel` is the only main-actor workflow coordinator.
- Training and runtime have separate profile IDs and status strings.
- Agent launch state is explicit (`isStartingAgent` / `agentIsActiveOrStarting`). Run controls, global shortcuts, storage changes, recording, replay, and app termination must use the combined state rather than only `isRunning`.
- High-rate capture/action callbacks never publish directly per event to SwiftUI. Runtime metrics are limited to 10 Hz; input HUD state is limited to 30 Hz with immediate control-edge updates; training is limited to 4 Hz.
- CNN inspector settings are optional persisted workflow state, remain live during a run, and are independent of profiles, checkpoints, cache identity, and the exact-vision PIP. The CNN image is placed above the exact frame in the same capture-excluded HUD.
- Visual keyboards preserve row order but omit every never-used key and every empty row so sparse demonstrations stay compact.
- The library has no synthetic all-recordings node. Every recording belongs to an expandable persisted folder; legacy/orphaned items are normalized into `Recordings`.
- Library event inspection uses `InputEventReader.summarize`, which memory-maps and streams records without materializing the entire timeline.
- The AVKit inspector must not host dynamically expanding `DisclosureGroup` or lazy content. The timeline is an explicit button plus a fixed-height plain `ScrollView`; this avoids AppKit constraint-tree mutation crashes during display updates.
- Thumbnail decoding occurs lazily per visible card.
- Complex parameter explanations use accessible info buttons with native hover help and click popovers.
- Primary button padding lives inside `PrimaryButtonStyle`, so the entire painted button is clickable. Do not reintroduce outer-only padding around a plain button.
- Diagnostics is a top-level app section. JSONL logging is capped in memory, rotates at 8 MiB, and exposes the latest macOS `.ips`/`.crash` reports without sending anything off the Mac.
- AI profiles are newest-first. List rows read `trainingProgress`; the full saved-brain list remains unloaded until the user presses Load Saved Brains.
- AI Models mirrors the Library's folder hierarchy for training recordings. Selecting a folder includes its future recordings automatically; expanding it exposes individual recording toggles.
- `OLEDCard`, primary buttons, status pills, the canvas, and navigation derive from one `AppTheme` plus one sanitized `UIAppearanceTuning`. The global corner radius scales nested controls proportionally; surface contrast, accent fill, and sidebar width stay within accessibility-minded bounds. Daylight supplies a true light color scheme; Midnight, Graphite, and Ember are polished dark alternatives.
- Interface motion is driven by one persisted toggle and the `uiMotionEnabled` environment value. Only opacity, scale, stroke, and one-shot chart masks animate; the root disables transactions when motion is off, Reduce Motion is enabled, or the app is inactive.
- `RootView` owns a 48-point opaque top bar and a solid custom sidebar instead of `NavigationSplitView`, native toolbar material, or blur. This is intentional: Sequoia, Tahoe, and macOS 27 therefore render the same active colors without version-specific glass or inactive-sidebar tinting. `WindowChromeConfigurator` makes the native titlebar transparent, preserves the standard traffic lights, and lets only the dedicated top-bar drag handle move the window. Page content remains below the bar.
- `GlobalHotkeyMonitor` uses `RegisterEventHotKey` when Carbon is present because it works without Accessibility and consumes the chord. A `#if !canImport(Carbon)` AppKit global/local-monitor fallback keeps future SDKs compilable, with the documented limitation that global key monitoring then requires Accessibility and cannot consume the target app's event. Shortcut suppression remains the authority that keeps trailing chord events out of recordings and safety input.
- Full-Mac bounds are discovered once per injector session rather than at action FPS. Target-focus safety checks are capped at 10 Hz, retaining a 100 ms response while avoiding WindowServer/AppKit work on every action tick.

## Safety and permissions

- Screen Recording: record and live vision.
- Input Monitoring: physical input recording and human-interruption safety.
- Accessibility: synthetic keyboard/mouse output and reenactment.
- Panic always stops replay and agent, stops recording, and requests a safe training stop.
- Full-Mac control is opt-in; otherwise pointer coordinates are clamped to the selected capture/control region.
- Cursor-movement and keyboard output each have an independently persisted, run-only permission. They remain live while an agent runs; keyboard includes modifier keys, while cursor movement does not disable mouse buttons or scrolling.
- Target-window focus loss stops a window-bound runtime after startup grace.

## Build, test, and distribution

The launch-time updater reads `ATGitHubOwner` and `ATGitHubRepository` from `Info.plist`, queries GitHub's public latest-release endpoint once per process launch, and compares semantic versions against `CFBundleShortVersionString`. Network or API failures are logged and never interrupt local workflows. A release intended for in-app discovery must use a version tag such as `v1.4.0` and attach both an AgentTrainer DMG and the exact `SHA256SUMS.txt` generated by `build.sh`. The app streams the DMG with visible progress, verifies its SHA-256, mounts it read-only, validates the embedded app's bundle/version and exact designated code-signing requirement, copies it to a hidden sibling, and launches a minimal helper that waits for clean app termination before transactionally swapping and reopening it. The old bundle is restored if the swap or final signature check fails. The updater never receives a GitHub token or uploads user data, and it refuses read-only/non-writable launch locations.

```sh
./test.sh
./build.sh
```

Required release checks:

1. All unit/regression tests pass, including compiled equality, exact optimizer/RNG resume, sparse-control loss masks, temporal input, rare-control validation, corruption rejection, blacklist filtering, deduplicated cache layout, updater pipe drainage, and long-loop throughput/memory stability.
2. `file outputs/AgentTrainer.app/Contents/MacOS/AgentTrainer` reports arm64.
3. `codesign --verify --deep --strict outputs/AgentTrainer.app` succeeds.
4. Launch the final rebuilt bundle against a disposable workspace, inspect Record, Library, Models, Training, Run, Diagnostics, Settings, and the menu extra through native UI automation. Never use a real library for a migration smoke test unless destructive schema cleanup is explicitly intended.
5. Toggle the Library timeline repeatedly while AVKit is visible, delete a disposable recording from its list row, expand model recording folders, click an info popover, switch through all four themes, exercise every appearance slider/reset and motion toggle, and confirm no new AgentTrainer crash report appears.
6. Run/stop an agent with permissions and confirm the event tap disappears, buttons/keys release, and camera motion does not continue.
7. Change each storage category to an empty disposable folder, verify migration and old-root cleanup, switch to a pre-populated test library without merging, restore the default locations, and test the missing-external-volume launch error.
8. Mount the single generated DMG; verify AgentTrainer.app, the Applications symlink, its signature, and a size below 10,000,000 bytes. Release builds suppress test-coverage data where the generated Xcode scheme permits it. The staged DMG app may strip link/debug symbols and the duplicate Resources metallib only; it must retain the executable-adjacent `mlx.metallib`.
9. On every available major toolchain, compile once with Xcode 27 and run on macOS 27 in addition to the macOS 15 deployment-floor build checks. Absence of Xcode 27 on a development Mac must be reported, not silently described as tested.

`build.sh` refuses an accidental ad-hoc build. It first honors `CODE_SIGN_IDENTITY`/`CODE_SIGN_KEYCHAIN`, then uses this Mac's long-lived local signing identity. It also refuses to modify a running app, asserts a macOS 15.0 Mach-O/plist floor, rejects profile-instrumented executables, updates a pre-existing output bundle in place, and verifies the signed app and disk image. Exactly one size-optimized DMG is built, the build fails if it reaches 10,000,000 bytes, and obsolete versioned DMGs are removed after a successful build. Xcode builds the MLX Metal library; the distributable executable comes from `swift build -c release`, avoiding the coverage counters inserted by Xcode's generated Swift-package scheme. Keeping `local.agenttrainer.mac`, the same certificate, and the same installed path gives rebuilt binaries one stable designated requirement for Screen Recording, Input Monitoring, and Accessibility. Moving from the historical ad-hoc signature requires one final TCC grant; later builds retain it. `ALLOW_ADHOC_SIGNING=1` is an explicit disposable-build escape hatch. A public frictionless download still requires a Developer ID Application certificate plus Apple notarization; this is independent of App Store distribution.

For a minimal Swift-only in-place update, follow `IN_PLACE_UPDATE_GUIDE.md` and replace only the executable. For a complete release that must also refresh the plist, Metal library, and DMGs, the hardened `build.sh` preserves an existing `outputs/AgentTrainer.app` directory while replacing its known contents and re-signing it with the same certificate. Always confirm that `outputs/AgentTrainer.app` is the copy the user actually launches.
