# AgentTrainer

AgentTrainer is an Apple-silicon macOS app that records screen and input demonstrations, trains behavioral cloning and live reinforcement learning policies locally with MLX, and runs those policies through macOS keyboard and mouse events. This checkout identifies the app as **2.60, build 38**.

Recordings, datasets, model weights, and diagnostics are stored locally. The app's networking is used for checking and downloading public GitHub releases; it does not upload training data or telemetry. Published installers are available from [GitHub Releases](https://github.com/UnendlessGit/AgentTrainer/releases).

## Requirements

- Apple silicon and macOS 15 or later.
- Screen Recording and Input Monitoring permissions for recording demonstrations.
- Accessibility permission for running an AI or reenacting input.
- Xcode with the macOS SDK and Swift 6.2 for source builds. The package declares MLX Swift 0.31.3 as an exact dependency.

## Using the app

1. Record a display, window, or region, or import existing `.atrrecord` packages through the Library. Recording settings include FPS, cursor visibility, output resolution, aspect-ratio lock, Color Bit, compression, and optional Context / Labels. Presets save these settings for reuse.
2. Organize recordings into folders in the Library. The Library supports trimming, renaming, moving, deleting, importing, exporting, and editing context labels. It shows recording duration, encoded-frame counts, and logical package sizes; older recordings without a saved frame count use an estimate.
3. Create an AI profile and select its demonstration folders or recordings. The editor configures vision, temporal history, action channels, architecture, precision, and training settings.
4. Train the profile. Pause saves a runnable brain and a checkpoint containing the optimizer and random state. Training and an already-trained AI can run concurrently when they use different profiles.
5. In Run, select a trained brain, capture source, cursor mode, output permissions, and any context values. Panic Stop releases held controls; the optional human-input stop also ends a run when physical input is detected.

The Record and Run source lists refresh while their pickers are visible and the app is active and idle. Application/display changes also trigger refreshes under those conditions.

## Live reinforcement learning

AI Models has **Demonstration** and **Reinforcement Learning** sections with shared vision, architecture, timing, and channels. RL can start from an existing Policy v9 brain or a new model without recordings. Switching methods keeps compatible policy weights; a stale optimizer never overwrites the active brain.

1. In AI Models → Reinforcement Learning, choose a preset or Advanced settings. Use **Add Keys** to capture allowed keys, select mouse buttons, and enable movement/scrolling as needed. Start/end keys are reserved for AgentTrainer.
2. Choose a capture source and **Select Reward Region**. Drag one region containing both text lines, press **Enter** to confirm, or **Escape** to cancel. Window selections follow movement; resizing, scale changes, or source loss pause safely.
3. Place white text on black in that region: `Reward: 125.5` and `Episode Length: 60`. **Test Detection** shows native-resolution pixels and recognition results without sending input. These pixels are masked from the model before resizing.
4. After the Episode Start Key (default R), show a fresh `Reward: 0` with positive episode length. Hold the reset display long enough to be detected; the default settling delay is 0.1 seconds. Episode length is a fixed maximum in seconds, not a countdown. A changed length adjusts the current deadline. Keep final reward and `Episode Length: 0` visible until the next start key.
5. Select Run → **RL Training** and start. The default RL shortcut is **Control–Option–Command–L**, configurable in Settings; it starts/resumes training or stops and saves. Pause/stop releases inputs before saving. Resume starts a new environment episode while retaining learned state.

Cumulative rewards are the default: the counter resets to zero each episode and differences are counted once, including negative differences and terminal rewards. Immediate mode treats the displayed value as a reward per valid action interval, including repeated equal values; unread intervals are excluded. Prefer cumulative mode when the display or OCR updates more slowly than actions.

PPO updates during episodes from short rollouts. Capture, action inference, and learning have separate queues/model ownership. Native OCR targets Action FPS, but actual throughput depends on text, resolution, and hardware. The dashboard reports measured reward FPS, latency, losses, updates, backlog, and excluded old actions. Learner duty cycle and retained experience are bounded; a one-second detection loss pauses and saves. Live RL also pauses on physical input or target focus loss to avoid attributing another controller's actions to the AI.

The dashboard includes reward history and a rolling average. The capture-excluded HUD retains its existing vision/input views and adds episode, reward, remaining time, and status. Checkpoints preserve the critic, exploration, optimizer masters/moments, random state, normalization, pending experience, and statistics. Advanced settings expose the implemented PPO controls; architecture/vision changes retain their existing explicit brain-reset confirmation.

See [RL implementation and environment contract](docs/reinforcement-learning.md) for timing and checkpoints, and [2.60 validation](docs/release-2.60.md) for measured learning and live performance results. Improvements in controlled tests do not guarantee learning speed or success in an arbitrary game.

## Recording and labels

New recordings use hardware HEVC encoding with an even output resolution, selectable 1–8 bit color detail, and a continuous compression setting. The default 8-bit path sends the captured YUV surface directly to the encoder; lower Color Bit settings quantize luma/chroma first. Existing recordings are not automatically transcoded.

Context / Labels is a user-defined catalog of categorical fields and values. A recording can have zero or one value per field. Training uses the values found in the selected recordings; an unassigned field has an all-zero input segment. Each saved brain contains its own vocabulary. Run initially leaves those fields blank, and a live context change releases old-context controls and clears pending predictions/history. Brains without a context vocabulary show no context card.

Imports accept individual native packages, folders of packages, and portable exports containing recording-folder and context metadata. Video and input-event bytes are copied without transcoding; IDs and manifest metadata can be remapped to resolve collisions. Exports include the selected packages and their referenced folder/context definitions.

## Vision, learning, and control

Every decision uses a current image and explicit timing inputs. The default temporal configuration adds four past images, two Perception FPS intervals apart, at half width and height. Setting Past frames to zero uses current-only vision while retaining decision timing.

The current model, Policy v9, shares a visual encoder between current and historical images. Historical images are paired with controls and timing. A complete-context stream and a visually grounded stream share policy weights; a bounded correction lets control history inform key/button decisions without unrestricted copying of the previous held state. Mouse movement supports bounded reuse of executed movement or a mode without that movement prior. Separate per-axis activity confidence permits exact stops.

Demonstration training uses behavioral cloning from recorded actions. Optional photometric augmentation, future-action prediction, and future-state prediction add training objectives. The future heads do not delay the immediate runtime action. Offline loss and recorded-video evaluations describe prediction behavior, but do not establish live task completion or generalization.

The CNN runs at Perception FPS; a lighter policy head runs at Action FPS with cached visual embeddings and fresh timing. Its control history comes from events actually posted after permissions, restrictions, clipping, and rounding. Runtime output also reflects saved model capabilities, demonstrated keys, current channel settings, and the control region. Stale predictions stop execution, and relative movement/scroll are consumed once per prediction.

## Saved data and compatibility

The default support directory is `~/Library/Application Support/AgentTrainer`:

| Item | Contents |
| --- | --- |
| `Recordings` | `.atrrecord` packages with video, synchronized binary input, thumbnails, and manifests |
| `Caches` | Rebuildable packed training datasets |
| `Profiles` | Profile metadata, saved model versions, checkpoints, and recovery archives |
| `Logs` | Bounded local diagnostics |
| `recording-folders.json` | Recording-folder catalog |
| `context-fields.json` | User-defined context field/value catalog |

RL checkpoints are stored under each profile in `RLCheckpoint`, independently of demonstration checkpoints. Saved RL brains include their matching resume state. Settings can relocate training data and model data independently. Relocation copies and verifies data before switching roots; selecting an existing library switches to it without merging.

The current formats are recording manifest schema 4, event format `ATREVT01` version 1, dataset schema 15, and model schema 9 (`AgentTrainer.Policy.v9`). The current migration archives incompatible brains/checkpoints while retaining recordings and profile settings. Compatible schema-15 caches remain usable. Existing v7/v8 brains are not converted into v9 weights. Within a compatible training configuration, checkpoints support exact pause/resume. A dropout-only edit retains compatible weights but starts a fresh optimizer sequence.

## Build and signing

`./build.sh` builds, signs, installs, and packages the app. Its default installation path is `/Applications/AgentTrainer.app`; `APP_PATH` selects another location. The script exits if AgentTrainer is running. Its outputs are `outputs/AgentTrainer-2.60.dmg`, `outputs/AgentTrainer-Source.zip`, and `outputs/SHA256SUMS.txt`.

Certificate signing is configured by `scripts/signing.sh`. It resolves usable certificate fingerprints and saves the certificate/permission-identity pin outside the checkout in `~/Library/Application Support/AgentTrainer/BuildSigning`. For a pinned installation, a missing key or mismatched designated requirement stops replacement. Matching signing identity lets macOS recognize app updates and retain privacy grants; a new Mac still needs its own permissions.

`CODE_SIGN_IDENTITY` selects a certificate by name or fingerprint, and `CODE_SIGN_KEYCHAIN` selects a keychain. Signed builds do not imply notarization: the build script signs the app and DMG but contains no notarization submission. The editable app icon is `Resources/AppIcon.icon`, compiled by the build script with Xcode's asset compiler.

The updater checks release versions, download hosts and sizes, SHA-256, the disk image, bundle identity, and the candidate's signing requirement. Installation uses a rollback copy until the replacement verifies.

## Test and diagnostic tools

| Command | What it does |
| --- | --- |
| `./test.sh` | Runs signing regressions and the debug Swift test suite, using a disposable app workspace |
| `./benchmark.sh` | Runs opt-in release hardware benchmarks for compact, default, and temporal-stress profiles |
| `./benchmark-mouse.sh` | Runs the fixture-specific two-mode mouse/control-feedback experiment; its current test expects 46 real Reborn recordings |
| `./benchmark-temporal.sh` | Runs the fixture-specific timing ablation; its current test expects 145 Blade Ball recordings |
| `zsh scripts/test-signing.sh` | Runs isolated signing-identity regressions with mocked signing tools |

The game-specific experiments use separate workspaces under ignored `work/` and depend on local recordings that are not distributed with the source. Hardware benchmark results depend on the Mac and workload. The mouse-feedback experiment routes generated events to a private sink against recorded video, which cannot respond to those actions as a live game would.

The [architecture overview](ARCHITECTURE.md) maps the implementation, with detailed references for [recording and storage](docs/recording-and-storage.md), [model and training](docs/model-and-training.md), and [runtime](docs/runtime.md).
