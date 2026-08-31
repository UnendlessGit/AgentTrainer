# AgentTrainer 2.4.0

AgentTrainer 2.4 adds isolated online reinforcement learning alongside the established training-data and DAgger workflows.

## Highlights

- A dedicated **Reinforcement Learning** tab now lives inside AI Models → Training Configuration. It includes bounded episode and rollout controls plus organized advanced PPO, critic, exploration, reward, connection, reconnect, normalization, and persistence settings.
- RL uses the existing Policy-v6 actor with clipped PPO and generalized advantage estimation. Its value critic, optimizer, sampling state, reward statistics, and checkpoint remain separate, so demonstration training and RL can warm-start each other without changing the supervised objective.
- The RL action space is a strict allowlist. Users may select exact keyboard keys, exact mouse buttons, absolute cursor movement, Game Camera movement, and scrolling; quick-add presets cover WASD, arrows, and number-row keys. Existing channel, restriction, Run permission, and control-region firewalls still apply.
- Rewards arrive over a configurable `ws://` or `wss://` connection. A bounded accumulator handles frequent reward messages without an unbounded queue, validates finite numeric JSON, supports remote episode completion, and reconnects with exponential backoff.
- Episode boundaries send `episode_finished`, `episode_starting`, and `episode_started` messages. Disconnects, stale rewards, PPO updates, context changes, and episode resets release synthetic controls and clear stale temporal/action state before output resumes.
- RL start and stop are global-keybind-only (default Control–Option–Command–L). The session reuses Run's capture source, region, frame mode, mouse execution, Game Camera, context, safety, preview, and live output permissions.
- Stopping RL transactionally publishes an ordinary runnable Policy-v6 actor. Normal Run does not create a reward connection and works with the reward server offline.
- RL rollout memory is preflighted against a fixed 1 GiB ceiling and physical memory. Exact actor-critic checkpoints are bounded to one transactional profile directory, and periodic runnable snapshots use the existing autosave retention policy.

## Verification

- 179 automated tests executed: 176 passed and the 3 opt-in hardware benchmarks were skipped in the normal suite.
- New coverage exercises allowlist intersections, blocked controls, deterministic hybrid sampling, GAE and terminal boundaries, reward validation/aggregation, high-rate loopback WebSocket delivery, outbound lifecycle ordering, PPO end-to-end visual gradients, exact actor-critic restore, both actor warm-start directions, immutable RL metadata, transactional publication/reset, and legacy hotkey decoding.
- The loopback integration test sustained 120 JSON reward messages per second and verified ordered `episode_finished`, `episode_starting`, and `episode_started` output.
- The default-profile RL gate completed a 64-transition, 8-step PPO update at 99.8 samples/s in 0.642 seconds with finite policy/value/entropy metrics, 36.8 MB active MLX memory after the update, and nominal thermal state.
- Compact, default, and 16-frame temporal-stress release benchmarks passed matched-quality, throughput, bounded-memory, sustained-load, and thermal gates. Default full-objective throughput retained 98.6% across 256 steps; temporal-stress full-objective throughput retained 107.9% across 192 steps; every thermal report was nominal.
- The 1280×720 Record-quality DAgger gate encoded 60 frames at 151.6 fps with zero drops.

AgentTrainer remains local-first, requires Apple silicon and macOS 15 or later, and ships as a hardware-HEVC macOS application.
