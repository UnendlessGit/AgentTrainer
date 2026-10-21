# AgentTrainer 2.46

AgentTrainer now trains exclusively through behavioral cloning. Recording, Policy-v6 architectures, temporal vision, Context / Labels, visual augmentation, predictive objectives, inference, guarded replay, and existing safety controls remain available.

## Removed

- DAgger correction recording, lead-up capture streams and buffers, correction weights, specialized validation and best-brain selection, publication orchestration, HUD mode, controls, and metrics.
- Reinforcement-learning engines, PPO/GAE, actor-critic and rollout state, stochastic action sampling and filters, reward WebSocket hosting and protocol, configuration editors, lifecycle controls, hotkey, metrics, and dedicated checkpoint APIs.
- Feature-only tests, benchmarks, historical instructions, and the explicit Network framework dependency. MLXOptimizers is now declared only for the tests that use it.

## Compatibility

- Old profile, training, workflow, preset, and hotkey settings are ignored and omitted on the next save, including malformed values belonging to removed features.
- Native recordings retain their existing video and input format. Saved supervision boundaries still keep earlier observation context out of action targets. All recordings now use normal behavioral-cloning splits and equal sample emphasis.
- Compatible Policy-v6 weights remain loadable. Previously published online actors use deterministic Run activations; their original key, button, movement, and scroll limits survive as immutable output capabilities. Sampling and smoothing semantics are no longer executed.
- Dataset schema 14 rebuilds disposable caches. A compatible active brain warm-starts with fresh optimizer/random state on the first training run after upgrade; subsequent pause/resume retains exact checkpoints.
- Old private training checkpoints remain inactive user data. Explicit learning reset still removes those directories transactionally. Protected models retain their existing protection.

## Validation

The debug suite passes 160 tests: 159 passed and the opt-in hardware benchmark was skipped. The regression suite covers malformed legacy settings, recording boundaries, immutable output limits, hardware HEVC recording, input synchronization and recovery, native import/export, storage transactions, model architectures, auxiliary gradients, and a real video-to-training-to-checkpoint-resume-to-inference round trip. The test runner uses a disposable workspace so training validation cannot publish into the user library.

All three release benchmark profiles pass the existing quality, throughput, and memory gates. Over 256 default-profile steps, full-objective throughput retention was 100.5%; over 192 temporal-stress steps it was 101.8%. Every measured thermal state was nominal.

The stress benchmark now releases Cocoa/Metal temporaries after each step, matching TrainingEngine’s existing lifetime boundary. Its quality, throughput, and memory thresholds are unchanged.

Build, test, and benchmark scripts explicitly select the established SwiftPM native backend, preserving their executable and Metal-library layout with newer toolchains. The application version is 2.46, build 33.
