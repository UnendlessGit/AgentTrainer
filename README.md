# AgentTrainer 2.2.0

AgentTrainer is a local-first Apple-silicon macOS app for recording demonstrations, training imitation policies with MLX, and running those policies with explicit safety controls.

Recordings, models, settings, and diagnostics stay on the Mac. The only built-in network operation is a check of the public AgentTrainer GitHub Releases feed; the app does not upload training data or telemetry.

## Requirements

- Apple silicon
- macOS 15 or later
- Screen Recording and Input Monitoring permission for recording
- Accessibility permission only when running an AI or reenacting input
- Xcode with the macOS SDK to build from source

The package pins MLX Swift exactly in `Package.resolved`.

## Workflow

1. Open AgentTrainer and grant the permissions needed for the operation you want.
2. Create a recording folder, choose a display, window, or region, optionally save those settings as a reusable preset, then record a demonstration.
   Existing native `.atrrecord` packages or a portable recording export can also be imported from the Library without transcoding.
3. Create an AI profile, select recording folders, and configure its vision, controls, and network.
4. Train the profile. Pause publishes a runnable brain and retains an exact checkpoint.
5. In Run, choose cursor mode and output permissions, then start the AI.
6. Use the configured panic shortcut at any time to stop hooks and release held controls.

Training and another already-trained AI may run at the same time when they use different profiles.

The capture-source picker refreshes while Record or Run is visible, and reacts immediately when apps launch or quit, displays change, or AgentTrainer becomes active again. A closed window is removed and the picker falls back to another valid source of the same kind.

## Policy v6

AgentTrainer 2.1 intentionally introduces a new brain format. Existing recordings remain reusable, while older weights and checkpoints are moved into the recovery archive instead of being attached to a different tensor layout.

Policy v6, introduced in AgentTrainer 2.1, is designed around both useful capacity and bounded local compute:

- a coordinate-aware dense stem is followed by depthwise spatial filters, pointwise channel mixing, and a same-width residual stage
- learned spatial keypoints retain layout while global mean/max context protects against narrow attention failures
- the sparse 146-value control history is compressed before the GRU/LSTM, reducing temporal work and limiting direct action-history shortcuts
- current-only profiles omit the recurrent network and its unused fusion parameters entirely
- live temporal inference encodes each reduced frame once, caches its compact visual embedding, and reuses that embedding when the frame enters later causal windows

Training has independent, configurable anti-memorization controls for label-preserving vision variation, small random neutral occlusions, control-history dropout/noise, whole temporal-token dropout, and binary label smoothing. These perturbations are disabled for validation and live inference. Recording-disjoint validation where possible, purged causal validation otherwise, per-head metrics, and best-held-out-brain activation remain the primary generalization checks.

## Faster training in 2.2

AgentTrainer 2.2 removes host and kernel overhead without changing the learned model or training objective. Compatible Policy v6 brains, exact optimizer checkpoints, recordings, and packed caches remain reusable.

- Multiple recordings are decoded concurrently with a CPU- and unified-memory-bounded worker count. Their cache shards are merged deterministically in the original recording order, and static-frame bytes are repeated in bulk.
- Batch planning keys repeated labels by their unique current perception instead of rebuilding and hashing complete temporal arrays for every row.
- The six existing action heads share one Metal matrix multiplication while retaining the same named weights and biases.
- AdamW keeps its two moment sets in contiguous MLX vectors during training, reducing many small GPU kernels; checkpoint files retain their established per-parameter tensor names and shapes.
- Up to four strictly ordered compiled MLX updates are chained before one host synchronization when unified memory permits. Queues never cross an epoch, autosave, pause, or configured run boundary.

Every row, current and past frame, target, loss term, optimizer update, schedule value, and random-state transition remains present. The release benchmark compares matched-update loss and validation behavior; no speed setting silently lowers resolution, precision, capacity, or data coverage.

## Recording efficiency

New recordings use the Apple-silicon hardware HEVC encoder with a direct 8-bit 4:2:0 capture surface, a high-quality resolution/FPS-aware bitrate, and efficient temporal compression. This is intentionally visually transparent lossy storage rather than near-lossless capture: dimensions, timing, and input synchronization are preserved while typical files are several times smaller than the former high-bitrate preset. Existing and imported recordings are never silently transcoded or rewritten.

The Library shows exact encoded-frame and logical package-size totals for new recordings. Legacy recordings retain an FPS/duration frame estimate. Each folder shows its complete recording count, frame count, size, and recorded seconds, independent of the current search filter.

The first training-data build decodes compressed video directly to native YUV, creates current and past training resolutions together in one Metal submission, overlaps decode/packing/writes through a bounded pipeline, and packs a held static frame once before repeating its exact bytes at every required perception interval. When several recordings are selected, bounded workers decode and pack them in parallel before an ordered byte-for-byte merge. Renaming or moving a recording no longer invalidates that expensive cache. Subsequent compatible training runs reuse it.

## Temporal vision

Every decision uses one exact-resolution current frame. Temporal memory can be disabled by setting Past frames to `0`; otherwise the current frame is joined by a configurable causal sequence of real past frames. Past frames:

- are reduced by a configurable linear factor while retaining the current frame's aspect, color mode, chroma layout, bit detail, and resize policy
- are sampled at a configurable number of Perception FPS intervals, with the editor showing both nominal seconds per step and total lookback
- remain ordinary images; AgentTrainer does not synthesize motion or difference channels
- carry the complete controls from their own perception interval: cursor position and raw movement, mouse buttons, scroll, keyboard, Shift, Control, Option, and Command

The default context is four past frames, two perception intervals apart, at half width and half height. The current frame always remains at the exact configured model resolution. Training reads the real causal images; live inference reuses the compact embeddings created when those images were first seen. Changing temporal vision creates a new model contract, and older incompatible brains are archived while recordings remain available for retraining. Changing training-only regularization does not change learned tensor shapes, so existing Policy v6 brain weights are retained; the next run starts a fresh optimizer sequence for the new stochastic objective.

## Safety

Runtime output is constrained in several independent layers:

- A physical-input monitor can stop the AI as soon as the user intervenes.
- The panic action stops capture and action queues before releasing every held key and button.
- A model can emit only keys represented in its training data.
- Per-profile restrictions and live cursor/keyboard permissions are applied again at execution.
- Full-Mac control is opt-in; otherwise output stays inside the configured control region.
- Locked-camera movement is transient and is never replayed from a stale prediction.

These are safeguards, not a security boundary. Supervise new brains in a low-risk environment.

## Build and test

Run the complete test suite:

```sh
./test.sh
```

Run the opt-in release benchmark for a compact fixture, the default vision/model/batch hardware profile, and a large 16-frame Float32 temporal stress profile (with stochastic layers disabled for repeatability):

```sh
./benchmark.sh
```

The benchmark reports end-to-end optimizer-step throughput, ordered multi-step pipeline throughput, held-out evaluation throughput, inference throughput, and learning-quality deltas against the pre-optimization graph. Default and temporal-stress runs also report sustained throughput windows, MLX memory, and thermal pressure. It compares training and validation loss after matched update counts; low-bit numerical trajectories may differ while the objective and every training label remain unchanged. Results are hardware-specific, so compare repeated warm runs on the same idle Mac.

Build, sign, package, and install:

```sh
./build.sh
```

The default installed application is:

```text
/Applications/AgentTrainer.app
```

### App icon

The editable icon source is [`Resources/AppIcon.icon`](Resources/AppIcon.icon),
an Icon Composer document. Open it in Xcode's Icon Composer, make and save your
changes, then run `./build.sh`; the build compiles its native Default, Dark, and
Mono variants for the app bundle and the icon shown inside AgentTrainer.

The build assembles and verifies a complete signed bundle before transactionally replacing that path. AgentTrainer must be quit first. Distribution artifacts are written to the ignored `outputs` folder:

- `AgentTrainer-2.2.0.dmg`
- `AgentTrainer-Source.zip`
- `SHA256SUMS.txt`

The build reuses the installed app's signing identity when available. You can select one explicitly:

```sh
CODE_SIGN_IDENTITY="Developer ID Application: Example" ./build.sh
```

Set `CODE_SIGN_KEYCHAIN` when the identity is outside the normal keychain search list. Use `APP_PATH` to install into another Applications folder. `ALLOW_ADHOC_SIGNING=1` is intended only for disposable development builds because an ad-hoc identity changes between builds and macOS permissions may need to be granted again.

## Data and privacy

The default support root is:

```text
~/Library/Application Support/AgentTrainer
```

It contains:

- `Recordings` — size-optimized hardware HEVC video, synchronized binary input, thumbnails, and manifests
- `Caches` — rebuildable packed datasets
- `Profiles` — profile metadata, immutable runnable versions, and exact checkpoints
- `Logs` — bounded local diagnostics

The Library can export any selected recording subset as a self-contained `Recordings` folder plus its folder metadata. Import validates every video, manifest, and synchronized input stream before changing the managed library; duplicate recording identifiers are regenerated without rewriting media or event contents.

Settings can relocate training data and models independently. Moves use copy, verification, switch, and cleanup steps; selecting an existing AgentTrainer library switches to it without merging. Do not manually edit manifests or move only part of a managed library.

## Updates

The updater accepts only a newer stable semantic version from the configured GitHub repository. It requires HTTPS, bounded downloads, an exact SHA-256 entry, a valid disk image, the same bundle identifier, the expected version, and the same designated signing requirement as the installed app. Installation keeps a rollback copy until the replacement verifies.

Engineering contracts and release checks are in [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md).
