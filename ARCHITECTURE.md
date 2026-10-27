# AgentTrainer architecture

AgentTrainer 2.61 (build 39) is a local macOS demonstration-recording, behavioral-cloning, live reinforcement-learning, and input-execution app. This reference describes the implementation in this checkout.

## System flow

Demonstrations combine screen video with timestamped physical inputs. Dataset construction samples causal images, controls, and timing from those recordings. MLX trains a policy and publishes a saved brain. Run loads that brain, encodes live frames, predicts actions with fresh timing, and feeds executed keyboard/mouse events back into its control history.

## Technical references

| Document | Contents |
| --- | --- |
| [Recording and storage](docs/recording-and-storage.md) | Capture, HEVC encoding, binary input format, imports/exports, presets, and workspace relocation |
| [Model and training](docs/model-and-training.md) | Vision, cache files, action/timing layouts, Policy v9, losses, optimization, checkpoints, and compatibility |
| [Runtime](docs/runtime.md) | Perception/action scheduling, output filtering, executed-control feedback, stopping, UI, and updater |

## Reinforcement learning

The [RL reference](docs/reinforcement-learning.md) describes native reward OCR, the episode state machine, timestamped executed-action intervals, hybrid PPO, independent actor/learner ownership, checkpoints, and bounded compute. Policy v9 weights and dataset schema 15 are retained. RL state is a separate versioned sidecar, and training-mode switches use active-policy lineage rather than restoring an unrelated optimizer.

## Components and state

The Swift 6.2 package targets macOS 15 and uses SwiftUI/AppKit, ScreenCaptureKit, AVFoundation, VideoToolbox, Metal, and MLX Swift 0.31.3. Source paths below are relative to `Sources/AgentTrainer`.

| Source | Role |
| --- | --- |
| `App/AgentTrainerApp.swift` | App/window lifecycle, menus, menu-bar controls, and termination handling |
| `App/AppModel.swift` | Main-actor UI state, operation orchestration, permissions, and source monitoring |
| `Core/Domain.swift` | Profiles, recording/model manifests, settings, action layouts, compatibility, and sizing |
| `Capture/CaptureService.swift` | Display/window discovery, ScreenCaptureKit streams, frame classification, and recording callbacks |
| `Capture/HEVCWriter.swift` | Hardware HEVC video writing and optional color quantization |
| `Capture/InputCaptureService.swift` | Input event tap, physical-state reconciliation, shortcut filtering, and HUD state |
| `Storage/WorkspaceStore.swift` | Actor-isolated libraries, context catalog, transfers, relocation, and artifact publication |
| `Storage/InputEventFile.swift` | Buffered binary-event writing and validated mapped reading |
| `Preprocessing/VisionPreprocessor.swift` | Metal resize, color conversion, packing, and MLX expansion |
| `Core/TemporalInputs.swift` | Shared causal control timeline and explicit timing features |
| `Core/MetalArrayBufferPool.swift` | Reusable shared-memory Metal buffers passed to MLX |
| `Training/DatasetCache.swift` | Causal frame/action sampling, cache files, mapped access, and batch gathering |
| `Training/PolicyNetwork.swift` | Policy v9, losses, augmentation, and resumable AdamW |
| `Training/TrainingEngine.swift` | Training/validation splits, compiled steps, sampling, checkpoints, and model selection |
| `Agent/AgentRuntime.swift` | Capture/inference scheduling, temporal history, context changes, and stop lifecycle |
| `Agent/InputInjector.swift` | Final output filtering, synthetic HID events, and executed-control feedback |
| `Replay/InputReenactor.swift` | Timestamped demonstration input replay with permissions and stop handling |
| `Core/GitHubReleaseUpdater.swift` | Release discovery, download verification, staging, and rollback installation |
| `UI/`, `Hotkeys/` | Pages, appearance, charts, context editing, HUD, CNN visualization, and global shortcuts |

`AppModel` is `@MainActor`; `WorkspaceStore` and `DatasetCacheBuilder` are actors. Capture, training, runtime inference, and injection also use queues and locks for framework callbacks and mutable low-level state. The injector serializes posting, permission changes, and releases with the same lock, preventing an old release from overtaking a newly posted hold. Runtime startup/teardown coordinates overlapping requests and drains queued work before another session starts.

Training and Run have separate profile identities. The UI permits background training alongside a different trained profile, while rejecting training and execution of the same profile concurrently.

## Build and diagnostic implementation

`build.sh` combines Xcode's MLX Metal-library build with SwiftPM's native backend executable build. It compiles the Icon Composer assets, assembles and verifies a staging app, transactionally installs at `APP_PATH` (default `/Applications/AgentTrainer.app`), and produces a DMG, source ZIP, and checksums under `outputs/`. It exits when AgentTrainer is running. The DMG contains a stripped/re-signed executable and one runtime Metal-library copy.

`scripts/signing.sh` resolves usable certificate fingerprints, compares the installed designated requirement with a saved pin, verifies staged/installed/packaged bundles, and saves the certificate/requirement pin outside the checkout. Matching identity preserves continuity for macOS privacy grants. Keys and the machine-local pin are outside the source-archive input list. The updater performs its own identity comparison on downloaded apps.

`test.sh` creates a disposable workspace, runs signing regressions, prepares the MLX Metal library, and runs the debug Swift test suite. Tests cover formats, timing/causality, storage transactions, compatibility, model/loss behavior, optimizer resume, input permissions/releases, feedback, and UI rendering. Opt-in experiments and hardware benchmarks are separately gated by environment variables.

`benchmark.sh` compares reference/optimized training and inference paths for several workloads, reporting numerical/learning differences, objective overhead, sustained throughput, memory, thermal state, and cached-head latency. Those latencies exclude capture, CNN encoding, queueing, and HID delivery. Its pass/fail thresholds are implemented in `Tests/AgentTrainerTests/TrainingPerformanceTests.swift`.

`benchmark-mouse.sh` runs the visual/bounded movement comparison defined in `MouseCorrectionTests.swift`, using a frozen selection of 46 local Reborn recordings and 20 epochs per arm. `benchmark-temporal.sh` runs the timing ablation in `TemporalExperimentTests.swift`, expecting 145 local Blade Ball recordings and 20 epochs per arm. Completed arms are evaluated without additional training. These are fixture-specific tools with isolated workspaces, not app training limits; ordinary training uses the profile/run settings. Their source datasets and saved experiment artifacts are not part of the repository.
