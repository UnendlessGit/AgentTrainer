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
2. Create a recording folder, choose a display, window, or region, set FPS, cursor, output resolution, ratio lock, Color Bit, compression, and lead-up, optionally save the complete setup as a reusable preset, then record a demonstration.
   Existing native `.atrrecord` packages or a portable recording export can also be imported from the Library without transcoding.
3. Create an AI profile, select recording folders, and configure its vision, controls, and network.
4. Train the profile. Pause publishes a runnable brain and retains an exact checkpoint.
5. In Run, choose cursor mode and output permissions, then start the AI.
6. When the AI makes a mistake, press the normal Record shortcut while Run is active. Run suspends, the UI and capture-excluded HUD switch to DAgger Recording, the causal lead-up is retained at normal Record quality, and your physical correction is saved as DAgger experience for that AI. Press Record again to resume Run immediately while the correction finishes publishing.
7. Resume training. Existing compatible weights are warm-started; correction-aware best-brain selection requires improved correction fit without a meaningful ordinary-behavior regression.
8. Use the configured panic shortcut at any time to stop hooks and release held controls.

Training and another already-trained AI may run at the same time when they use different profiles.

The capture-source picker refreshes while Record or Run is visible, and reacts immediately when apps launch or quit, displays change, or AgentTrainer becomes active again. A closed window is removed and the picker falls back to another valid source of the same kind.

## DAgger corrections

DAgger is explicit. The Record shortcut creates an ordinary demonstration when no AI is running and toggles correction capture during Run. Ordinary physical takeover still follows the Run safety setting and never starts a recording by itself.

At the trigger, AgentTrainer stops inference and new AI output before preparing the recorder, drains work already in flight, releases held synthetic controls, and switches the visible lifecycle from Run to DAgger Recording. A correction package contains:

- a continuous causal lead-up and correction video using the Record page's source, FPS, cursor, resolution, Color Bit, and compression rather than the AI's model-vision resolution
- the AI or human controls actually associated with those causal frames
- expert action targets beginning only at the explicit trigger, so the mistake is context rather than a label to imitate
- a bounded correction weight so short interventions are not drowned by ordinary recording volume

Lead-up is universal and lives on the Record page; correction weight remains AI-specific. Stopping DAgger clears stale predictions and temporal state before Run resumes, so no pre-correction action can leak into the restarted policy. DAgger packages are ordinary native recordings in the Library, are automatically attached to the running AI, and remain exportable. Training always keeps corrections and at least one ordinary demonstration in the training split. A fixed correction probe and an ordinary validation/rehearsal guard select the runnable brain together; the exact key capabilities and runtime semantics belonging to the chosen weights are published with them.

Correction buffers and video admission are bounded. A package with more than 5% visual-frame loss is discarded instead of learning expert events against stale held video; smaller drop counts are recorded and shown. Quit, storage moves, folder deletion, and profile deletion wait for pending correction publication.

## Recording efficiency

New recordings use the Apple-silicon hardware HEVC encoder with a 4:2:0 capture surface, an exact configurable even output resolution, selectable 1–8 bit color detail, and a simple continuous compression control. The default 8-bit path remains zero-copy; lower Color Bit values use pooled accelerated legal-range quantization that preserves neutral chroma before hardware encoding. Dimensions, timing, and input synchronization remain exact, and every setting—including ratio lock and universal DAgger lead-up—round-trips through recording presets. Existing and imported recordings are never silently transcoded or rewritten.

The Library shows exact encoded-frame and logical package-size totals for new recordings. Legacy recordings retain an FPS/duration frame estimate. Each folder shows its complete recording count, frame count, size, and recorded seconds, independent of the current search filter.

The first training-data build decodes compressed video directly to native YUV, creates current and past training resolutions together in one Metal submission, overlaps decode/packing/writes through a bounded pipeline, and packs a held static frame once before repeating its exact bytes at every required perception interval. Renaming or moving a recording no longer invalidates that expensive cache. Subsequent compatible training runs reuse it.

## Temporal vision

Every decision uses one exact-resolution current frame. Temporal memory can be disabled by setting Past frames to `0`; otherwise the current frame is joined by a configurable causal sequence of real past frames. Past frames:

- are reduced by a configurable linear factor while retaining the current frame's aspect, color mode, chroma layout, bit detail, and resize policy
- are sampled at a configurable number of Perception FPS intervals, with the editor showing both nominal seconds per step and total lookback
- remain ordinary images; AgentTrainer does not synthesize motion or difference channels
- carry the complete controls from their own perception interval: cursor position and raw movement, mouse buttons, scroll, keyboard, Shift, Control, Option, and Command

The default context is four past frames, two perception intervals apart, at half width and half height. The current frame always remains at the exact configured model resolution. Changing temporal vision creates a new model contract, and older incompatible brains are archived while recordings remain available for retraining. Changing dropout does not change learned tensor shapes, so existing brain weights are retained; the next training run starts a fresh optimizer sequence for the new regularization setting.

## Generalization and predictive learning

The AI editor independently configures three training-only features. None delays or averages the immediate runtime action:

- **Visual Augmentation** varies luminance, contrast, saturation, hue, and gamma within conservative bounds. Spatial transforms are intentionally excluded because they would change cursor/action meaning. Every supervised sample receives an independent transform; only that sample's current, past, and future frames share it, preserving temporal consistency without correlating the batch.
- **Future Action Prediction** predicts up to four configurable action horizons. Near horizons receive more weight, teaching intent and smoother continuation while the immediate head remains the sole runtime output.
- **Future State Prediction** predicts an action-conditioned change in the learned visual state. It uses stopped-gradient latent targets and never generates future images; repeated future observations are deduplicated before encoding.

The auxiliary heads have a fixed Policy-v6 layout, so these settings can warm-start compatible brain weights. Conservative default loss weights passed a matched-update causal held-out test with no more than 1% immediate-policy loss drift while both auxiliary objectives improved.

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

The benchmark reports end-to-end optimizer-step throughput, held-out evaluation throughput, inference throughput, and learning-quality deltas against the pre-optimization graph. It also executes the complete Policy-v6 objective—including independent per-sample augmentation and both predictive heads—and reports its overhead, memory, and thermal state. Default and temporal-stress runs report sustained windows for both the optimized core path and the full Policy-v6 objective. Low-bit numerical trajectories may differ while the objective and every training label remain unchanged. Results are hardware-specific, so compare repeated warm runs on the same idle Mac.

Run the opt-in 1280×720 DAgger video-throughput gate with:

```sh
AGENTTRAINER_RUN_DAGGER_PERFORMANCE_TESTS=1 swift test -c release --filter TrainingPerformanceTests.testRecordQualityDAggerVideoPipelineSustainsDefaultWorkload
```

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
