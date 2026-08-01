# AgentTrainer 1.9.7

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
2. Create a recording folder, choose a display, window, or region, then record a demonstration.
3. Create an AI profile, select recording folders, and configure its vision, controls, and network.
4. Train the profile. Pause publishes a runnable brain and retains an exact checkpoint.
5. In Run, choose cursor mode and output permissions, then start the AI.
6. Use the configured panic shortcut at any time to stop hooks and release held controls.

Training and another already-trained AI may run at the same time when they use different profiles.

## Temporal vision

Every decision uses one exact-resolution current frame plus a configurable causal sequence of real past frames. Past frames:

- are reduced by a configurable linear factor while retaining the current frame's aspect, color mode, chroma layout, bit detail, and resize policy
- are sampled at a configurable number of Perception FPS intervals, with the editor showing both nominal seconds per step and total lookback
- remain ordinary images; AgentTrainer does not synthesize motion or difference channels
- carry the complete controls from their own perception interval: cursor position and raw movement, mouse buttons, scroll, keyboard, Shift, Control, Option, and Command

The default context is four past frames, two perception intervals apart, at half width and half height. The current frame always remains at the exact configured model resolution. Changing temporal vision creates a new model contract, and older incompatible brains are archived while recordings remain available for retraining.

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

Run the opt-in release benchmark for a compact fixture and the default vision/model/batch hardware profile (with stochastic layers disabled for repeatability):

```sh
./benchmark.sh
```

The benchmark reports end-to-end optimizer-step throughput, held-out evaluation throughput, inference throughput, and learning-quality deltas against the pre-optimization graph. It compares training and validation loss after matched update counts; low-bit numerical trajectories may differ while the objective and every training label remain unchanged. Results are hardware-specific, so compare repeated warm runs on the same idle Mac.

Build, sign, package, and install:

```sh
./build.sh
```

The default installed application is:

```text
/Applications/AgentTrainer.app
```

The build assembles and verifies a complete signed bundle before transactionally replacing that path. AgentTrainer must be quit first. Distribution artifacts are written to the ignored `outputs` folder:

- `AgentTrainer-1.9.7.dmg`
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

- `Recordings` — HEVC video, synchronized binary input, thumbnails, and manifests
- `Caches` — rebuildable packed datasets
- `Profiles` — profile metadata, immutable runnable versions, and exact checkpoints
- `Logs` — bounded local diagnostics

Settings can relocate training data and models independently. Moves use copy, verification, switch, and cleanup steps; selecting an existing AgentTrainer library switches to it without merging. Do not manually edit manifests or move only part of a managed library.

## Updates

The updater accepts only a newer stable semantic version from the configured GitHub repository. It requires HTTPS, bounded downloads, an exact SHA-256 entry, a valid disk image, the same bundle identifier, the expected version, and the same designated signing requirement as the installed app. Installation keeps a rollback copy until the replacement verifies.

Engineering contracts and release checks are in [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md).
