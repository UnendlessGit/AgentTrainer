# AgentTrainer 2.52

AgentTrainer is a local-first Apple-silicon macOS app for recording demonstrations, training behavioral cloning policies with MLX, and running those policies with explicit safety controls.

Recordings, models, settings, and diagnostics stay on the Mac. Network use is limited to the public GitHub Releases check. AgentTrainer does not upload training data or telemetry.

**2.52 is a diagnostic candidate with positive live Reborn feedback.** The retained-history brain was preferred over the matching visual-movement comparator for both keys and camera control; trial completion and broader generalization remain unproven. The latest published release is still 2.50. The previous 2.51 comparison was rejected in actual play. See [movement-control findings](MOVEMENT_CONTROL.md) for verified results and remaining limitations.

Download the published signed macOS installer from the [latest GitHub Release](https://github.com/UnendlessGit/AgentTrainer/releases/latest). Version 2.52 preserves the app's signing identity across rebuilds and updates. On this Mac, updates signed with the same identity should retain existing privacy permissions. A new Mac still needs permissions granted once; building from source requires your own signing key.

## Requirements

- Apple silicon
- macOS 15 or later
- Screen Recording and Input Monitoring permission for recording
- Accessibility permission only when running an AI or reenacting input
- Xcode with the macOS SDK to build from source

The package pins MLX Swift exactly in `Package.resolved`.

## Workflow

1. Open AgentTrainer and grant the permissions needed for the operation you want.
2. Create a recording folder, choose a display, window, or region, set FPS, cursor, output resolution, ratio lock, Color Bit, compression, and any custom Context / Labels, optionally save the complete setup as a reusable preset, then record a demonstration.
   Existing native `.atrrecord` packages or a portable recording export can also be imported from the Library without transcoding.
3. Create an AI profile, select recording folders, and configure its vision, controls, and network.
4. Train the profile from recordings. Pause publishes a runnable brain and retains an exact checkpoint.
5. In Run, choose cursor mode and output permissions, then start the AI.
6. Use the configured panic shortcut at any time to stop hooks and release held controls.

Training and another already-trained AI may run at the same time when they use different profiles.

The capture-source picker refreshes while Record or Run is visible, and reacts immediately when apps launch or quit, displays change, or AgentTrainer becomes active again. A closed window is removed and the picker falls back to another valid source of the same kind.

## Compatibility

Version 2.52 introduces Policy v9 (model schema 9). Existing recordings and schema-15 training caches remain compatible. Weight-format metadata is checked even when an older brain has matching tensor shapes. Incompatible brains and checkpoints are preserved in the managed archive; profiles keep their architecture and temporal settings, and require retraining from their recordings. There is no exact v7/v8-to-v9 optimizer resume or weight conversion. BF16/FP16 training now preserves small updates using Float32 master weights, including exact residuals in checkpoints. Exact pause/resume remains available within v9.

Older DAgger/reinforcement-learning settings remain inactive. Saved supervision boundaries still allow earlier frames and held-control history as context while excluding earlier actions from labels. Training uses whole-recording held-out splits and equal recording emphasis.

## Recording efficiency

New recordings use the Apple-silicon hardware HEVC encoder with a 4:2:0 capture surface, an exact configurable even output resolution, selectable 1–8 bit color detail, and a simple continuous compression control. The default 8-bit path remains zero-copy; lower Color Bit values use pooled accelerated legal-range quantization that preserves neutral chroma before hardware encoding. Dimensions, timing, and input synchronization remain exact, and every setting—including ratio lock and Context / Labels—round-trips through recording presets. Existing and imported recordings are never silently transcoded or rewritten.

The Library shows exact encoded-frame and logical package-size totals for new recordings. Legacy recordings retain an FPS/duration frame estimate. Each folder shows its complete recording count, frame count, size, and recorded seconds, independent of the current search filter.

The first training-data build decodes compressed video directly to native YUV, creates current and past training resolutions together in one Metal submission, overlaps decode/packing/writes through a bounded pipeline, and packs a held static frame once before repeating its exact bytes at every required perception interval. Renaming or moving a recording no longer invalidates that expensive cache. Subsequent compatible training runs reuse it.

## Context / Labels

Context is a generic user-defined categorical system. AgentTrainer does not ship game, task, camera, vehicle, or other domain-specific labels. Create any number of fields and values in Record or Library, then select zero or one value from each field for a recording.

- Record freezes the current selections when capture starts. Recording presets retain them.
- Library shows labels on recording rows and supports adding, changing, or clearing a field for one recording or a multi-selection. Bulk manifest edits roll back together on failure and never rewrite video or synchronized input.
- Renaming a field or value updates recording snapshots without rebuilding packed pixels. Changing the selected value invalidates only the relevant rebuildable dataset/checkpoint identity.
- Training snapshots only fields and values actually observed in the AI's selected recordings. Each value occupies one stable categorical input; a missing field is an all-zero segment rather than an invented hidden category.
- Run loads the immutable vocabulary from the selected brain. Every trained field appears blank so the current operating context can be chosen, and changes take effect on the next perception after stale temporal/action state is released. A brain trained without context shows no context controls.

Context-free Policy-v9 brains omit the categorical tensor. Adding or changing trained categorical IDs changes the learned contract; renaming display labels does not.

## Temporal vision

Every decision uses one exact-resolution current frame. Temporal memory can be disabled by setting Past frames to `0`; otherwise the current frame is joined by a configurable causal sequence of real past frames. Past frames:

- are reduced by a configurable linear factor while retaining the current frame's aspect, color mode, chroma layout, bit detail, and resize policy
- are sampled at a configurable number of Perception FPS intervals, with the editor showing both nominal seconds per step and total lookback
- remain ordinary images; AgentTrainer does not synthesize motion or difference channels
- carry the complete executed controls from the trailing interval ending at their observation query: cursor position and raw movement, mouse buttons, scroll, keyboard, Shift, Control, Option, and Command

The default context is four past frames, two perception intervals apart, at half width and half height. The current frame always remains at the exact configured model resolution. Changing temporal vision creates a new model contract, and older incompatible brains are archived while recordings remain available for retraining. Changing dropout does not change learned tensor shapes, so existing brain weights are retained; the next training run starts a fresh optimizer sequence for the new regularization setting.

## Explicit temporal inputs

Policy v9 receives elapsed time on every action decision, even when Past frames is `0`. Each supported key, modifier, and mouse button has held state, continuous hold age, press/release recency, last completed hold duration, interval edge counts, and held-time fraction. Mouse movement and scrolling include displacement, sample counts, first/last sample timing, observed spans, and real-time rates.

Current and historical inputs also carry observation intervals, source-frame age, historical age/separation, and validity indicators. Unknown startup hold durations and missing history remain explicitly unknown. Nanosecond clocks are subtracted before conversion to Float32; short linear and longer logarithmic encodings preserve both taps and long holds. Event spans describe observed samples, not an inferred duration of physical device motion.

The CNN runs at Perception FPS. A separate lightweight policy runs at Action FPS with cached visual embeddings, fresh timing, and feedback from the events actually posted after restrictions, clipping, and rounding. Two decisions sharing one image can therefore produce different actions. At 60 Action FPS the configured execution resolution is approximately 16.7 ms; this does not add arbitrary sub-tick HID scheduling.

The earlier Policy-v7 matched Blade Ball experiment used all 145 recordings, whole-recording held-out data, 60/60 FPS, 128×128 vision, focal gamma 0, and exactly 20 epochs per arm. Timing improved immediate held-out loss by 5.2% and relative mouse error by 13.8%. A controlled identical-image duration task reached 95.3% accuracy. Some real-data edge-timing and scroll errors worsened; offline results do not establish better gameplay. Method, full measurements, and input contracts are in [TEMPORAL_DESIGN.md](TEMPORAL_DESIGN.md).

## Mouse movement and stopping

The movement-memory setting is saved with each trained brain. **Bounded movement history** retains recent executed movement through a limited, time-normalized history path. **Visual timing only** predicts mouse movement without feeding previous cursor, movement, or scroll values into its motor path. Both ground camera direction and stopping in real visual history and observation clocks. Complete key/button/control timing remains available through a bounded correction to visually grounded binary actions, so a generated hold cannot freely dictate the next hold. The same real recorded actions separately supervise the visual base during training.

Relative movement has separate learned per-axis activity decisions. They allow an exact stop rather than repeatedly posting a small regression residual. The bounded path cannot change these decisions, its own attention weights, or the independent base movement prediction. This limits direct feedback amplification; it does not guarantee stability of an arbitrary game or eliminate the need for representative demonstrations.

Validation now reports mouse activity recall and idle false moves at the actual half-pixel rounding threshold for default 1× sensitivity. The opt-in `./benchmark-mouse.sh` compares two 20-epoch candidates trained only on real recordings and exercises production event posting into a private sink, including feedback from the commands actually posted. Recorded video cannot respond to those commands, so this check supplements a live-game comparison. Architecture, reproduction details, and measured limitations are in [MOVEMENT_CONTROL.md](MOVEMENT_CONTROL.md).

## Generalization and predictive learning

The AI editor independently configures three training-only features. None delays or averages the immediate runtime action:

- **Visual Augmentation** varies luminance, contrast, saturation, hue, and gamma within conservative bounds. Spatial transforms are intentionally excluded because they would change cursor/action meaning. Every supervised sample receives an independent transform; only that sample's current, past, and future frames share it, preserving temporal consistency without correlating the batch.
- **Future Action Prediction** predicts up to four configurable action horizons. Near horizons receive more weight, teaching intent and smoother continuation while the immediate head remains the sole runtime output.
- **Future State Prediction** predicts an action-conditioned change in the learned visual state. It uses stopped-gradient latent targets and never generates future images; repeated future observations are deduplicated before encoding.

The auxiliary heads have a fixed Policy-v9 layout, so these settings can warm-start compatible v9 brain weights. The release benchmark checks the full objective, including visual-base supervision, augmentation, and both predictive heads, against its unchanged quality and overhead gates.

## Safety

Runtime output is constrained in several independent layers:

- A physical-input monitor can stop the AI as soon as the user intervenes.
- The panic action stops capture and action queues before releasing every held key and button.
- A model can emit only keys and mouse buttons represented in its supervised training data. Untrained binary heads stay disabled.
- Per-profile restrictions and live cursor/keyboard permissions are applied again at execution.
- Full-Mac control is opt-in; otherwise output stays inside the configured control region.
- Locked-camera movement is transient and is never replayed from a stale prediction.

Physical input recovery reads macOS's HID-system hardware state rather than the foreground session's mutable accumulated event state. A game or another app therefore cannot turn a physically held key, modifier, or mouse button into a synthetic release in the recording. Reconciliation snapshots are revision-guarded as well, so an in-flight poll cannot overwrite a newer first-frame recording seed; genuine missed releases are still repaired by the next fresh poll and every recording is balanced at shutdown.

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

The benchmark reports end-to-end optimizer-step throughput, held-out evaluation throughput, inference throughput, cached action-head p50/p95 latency, and learning-quality deltas against the pre-optimization graph. It also executes the complete Policy-v9 objective—including visual-base supervision, independent per-sample augmentation and both predictive heads—and reports its overhead, memory, and thermal state. Default and temporal-stress runs report sustained windows for both the optimized core path and the full Policy-v9 objective. Low-bit numerical trajectories may differ while the objective and every training label remain unchanged. Results are hardware-specific, so compare repeated warm runs on the same idle Mac.

Run the optional local Blade Ball timing experiment (requires the existing 145-recording folder):

```sh
./benchmark-temporal.sh
```

It writes isolated artifacts under ignored `work/temporal-evaluation-v8`, never modifies the original recordings or existing profiles, disables the step cap/repetition, and stops each arm after 20 epochs. Running it again evaluates completed arms without adding epochs.

Run the local real-Reborn mouse-feedback comparison (requires the 39-recording fixture and preserved v7 reference):

```sh
./benchmark-mouse.sh
```

It uses ignored `work/mouse-comparison-v9`, trains each candidate for exactly 20 epochs, and evaluates completed candidates without further training. No global input is posted. See [MOVEMENT_CONTROL.md](MOVEMENT_CONTROL.md) for the feedback method and its limits.

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

- `AgentTrainer-2.52.dmg`
- `AgentTrainer-Source.zip`
- `SHA256SUMS.txt`

The build selects an exact, usable certificate fingerprint, excluding revoked certificates. It reuses the installed app's signing identity and saves a machine-local certificate/permission-identity pin in `~/Library/Application Support/AgentTrainer/BuildSigning`. This survives clean builds and repository updates. Every installed and packaged bundle must retain that same designated requirement; a missing key or identity change stops the build before replacement instead of silently resetting permissions. Keep using `./build.sh` and the same installed app path for future updates.

You can select a certificate explicitly (an exact SHA-1 fingerprint is preferred when names are duplicated):

```sh
CODE_SIGN_IDENTITY="Developer ID Application: Example" ./build.sh
```

Set `CODE_SIGN_KEYCHAIN` when the identity is outside the normal keychain search list; unlock it normally or provide `CODE_SIGN_KEYCHAIN_PASSWORD` for a protected build keychain. Use `APP_PATH` to install into another Applications folder. `ALLOW_ADHOC_SIGNING=1` is intended only for disposable development builds without an existing stable permission identity. It cannot override an installed identity or a saved pin. Certificate renewal can be selected explicitly, but must preserve the designated requirement. The pin contains public certificate/requirement information, never private keys, and is excluded from distribution artifacts.

## Data and privacy

The default support root is:

```text
~/Library/Application Support/AgentTrainer
```

It contains:

- `Recordings` — size-optimized hardware HEVC video, synchronized binary input, thumbnails, manifests, and portable context snapshots
- `Caches` — rebuildable packed datasets
- `Profiles` — profile metadata, immutable runnable versions, exact behavioral cloning checkpoints
- `Logs` — bounded local diagnostics

The training-data root also contains `context-fields.json`, the reusable custom field/value catalog.

The Library can export any selected recording subset as a self-contained `Recordings` folder plus its folder and context-field metadata. Import validates every video, manifest, synchronized input stream, and context definition before changing the managed library; duplicate recording or context identifiers are remapped without rewriting media or event contents.

Settings can relocate training data and models independently. Moves use copy, verification, switch, and cleanup steps; selecting an existing AgentTrainer library switches to it without merging. Do not manually edit manifests or move only part of a managed library.

## Updates

The updater accepts only a newer stable semantic version from the configured GitHub repository. It requires HTTPS, bounded downloads, an exact SHA-256 entry, a valid disk image, the same bundle identifier, the expected version, and the same designated signing requirement as the installed app. Installation keeps a rollback copy until the replacement verifies.

Engineering contracts and release checks are in [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md).
