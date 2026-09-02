# AgentTrainer 2.4.5

AgentTrainer 2.4.5 is a production-focused reinforcement-learning reliability, runtime-parity, safety, and UX release.

## Highlights

- Normal Run now executes RL brains with their actual saved policy semantics. Allowed keys, modifiers, and mouse buttons are sampled from the Bernoulli distribution at the immutable training temperature instead of being converted into permanent holds by the demonstration runtime's fixed `0.5` threshold. Demonstration-trained brains keep their established deterministic behavior.
- RL mouse control now uses category-specific conservative exploration rather than one oversized continuous deviation. Absolute cursor actions start from the real pointer and are smoothed and step-limited; Game Camera actions are smoothed, dead-zoned, and magnitude-limited; scroll has independent exploration and smoothing. The same immutable action transform is used during collection and Run, while Run removes continuous exploration noise.
- AgentTrainer now hosts the reward WebSocket. One universal endpoint applies to every AI and is configured in Settings. It is loopback-only by default, admits one rewarder, limits messages to 64 KiB, waits for the rewarder before capture/actions start, and has an explicit warning-bearing trusted-LAN option.
- Settings lists every incoming field/type and outgoing lifecycle type. The type, reward, done, episode-finished, heartbeat, episode-starting/started/finished, and PPO-update-started/finished names are independently editable, validated, collision-checked, persisted, and covered end to end.
- Rewarder disconnect, reward/heartbeat timeout, and invalid-message-flood safety are configurable. Controls release immediately in every case. Default behavior transactionally checkpoints, publishes any newly updated runnable actor, and stops after a configurable reconnect grace; optional pause mode keeps listening safely for a reconnect or heartbeat.
- Application quit, panic, capture loss, stale inference, focus loss, physical input, and relevant Run-permission changes share the same ordered stop path. Every final stop refreshes the exact actor/critic/optimizer/random checkpoint.

## RL quality and lifecycle

- `Maximum Episodes` now counts episodes in the current start-to-stop run, while lifetime history remains cumulative. Runnable publication and periodic checkpoint cadence are session-relative as well.
- Short terminal rollouts are no longer silently deleted at episode boundaries. Terminal separators remain intact for GAE, and a maximum-episode or safe/final stop performs a finite PPO update even when only one to three transitions are available.
- Initial listener/rewarder wait, capture startup, environment reset delay, disconnect/staleness pauses, and PPO time no longer consume the episode duration budget.
- Lifecycle ordering is explicit: controls release, `episode_finished` is delivered, PPO update messages surround the update, then `episode_starting` and `episode_started` bracket reset/start. Rewards sent during a declared update are ignored and counted; a done received during the update is preserved.
- RL startup has an atomic final install guard, preventing a cancelled launch from reviving a late optimizer, listener, capture stream, input monitor, or injector.
- Live output permissions participate in action-space preflight. Changing a permission that would invalidate an active on-policy distribution saves and stops instead of training against an action that could not execute.
- Legacy 2.4.0 per-AI transport values migrate once into the universal settings and are removed from profile transport ownership. Existing Policy-v6 actors remain valid warm starts; exact pre-2.4.5 RL checkpoints restart critic/optimizer state under the corrected action-distribution contract.

## Controls and UX

- The RL key picker now listens for both key-down and modifier-flag changes and supports the full usable macOS virtual-key range, including Space, Return, Tab, Escape, function/navigation/keypad keys, arrows, Caps Lock, Shift, Control, Option, Command, and Fn/Globe.
- The key-name catalog now presents meaningful names for the complete standard 0–127 range instead of generic numeric chips for common special keys.
- Mouse-button capture no longer consumes the SwiftUI event needed to click **Done adding buttons**. A newly observed finishing click is removed from the allowlist so using Done does not accidentally grant Left Mouse.
- Run shows current-run and lifetime episode counters, hosted-server state, invalid messages, rewards ignored during PPO, and the active RL sampling/stabilization mode.
- Global safety, focus-loss behavior, frame mode, preview behavior, output permissions, Game Camera settings, and WebSocket settings now persist across launches.

## Verification

- The complete debug suite executed 187 tests: 184 passed and the 3 opt-in hardware benchmarks were skipped as designed.
- Hosted WebSocket integration tests cover 120 Hz reward input, ordered outbound lifecycle delivery, disconnect/reconnect, and fully customized incoming/outgoing names with a real Python rewarder client.
- Focused policy tests cover Bernoulli runtime sampling below the old threshold, per-category Gaussian likelihoods, cursor/camera bounds, session episode limits, short-rollout retention, and a finite single-transition final PPO update.
- The default-profile 64-transition PPO gate completed 8 optimizer steps at 99.4 samples/s in 0.644 seconds with finite policy/value/entropy metrics, about 36.2 MB active MLX memory after the update, and nominal thermal state.
- The default optimized-core sustained window retained 101.5% throughput over 256 steps; the full Policy-v6 objective retained 99.2%. The 16-frame Float32 temporal stress core retained 101.5% over 192 steps and its full objective retained 101.6%. Every reported thermal state was nominal and memory remained bounded by the existing gates.

AgentTrainer remains local-first, requires Apple silicon and macOS 15 or later, and uploads no training data or telemetry.
