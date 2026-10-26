# AgentTrainer 2.60 — build 38

Native Swift/MLX PPO with generalized advantage estimation adds live reinforcement learning alongside demonstration training. The shared Policy v9 encoder, temporal model, and action heads remain compatible. Critic/exploration and optimizer state live in RL sidecars; switching methods uses the active policy instead of a stale checkpoint or random initialization.

## Included

- Demonstration / Reinforcement Learning configuration, shared vision/architecture/controls, Beginner presets and Advanced PPO settings.
- Explicit allowed keys, eight mouse buttons, movement and scrolling; separate controller start/end keys, including the same key for both.
- Native-resolution reward/episode OCR with screen-region selection, Enter/Escape, live preview, and non-actuating detection tests. Rewards are masked from all model observations before resizing.
- Timestamped cumulative intervals and immediate-reward accounting, dynamic fixed episode durations, final zero-length rewards, reset acknowledgment, bounded final-reading drain, safe pause/resume and panic handling.
- Live updates during episodes, bounded microbatches/backlog, independent inference/learning model ownership, Float32 objectives/optimizer accumulation, gradient clipping and KL guards.
- Transactional policy/critic/optimizer/random/normalization/experience checkpoints, automatic saving, episode charts, dashboard metrics, compact HUD status, and a configurable Control–Option–Command–L shortcut.

## Release validation

The final isolated suite executed **224 tests with six opt-in data/hardware diagnostics skipped and zero failures**, including the enabled three-seed learning gate, in 140.1 seconds. The normal suite skips that longer gate as well. The remaining skipped diagnostics require separate real-recording datasets or benchmark flags. A further full regression run after the signed-app smoke test executed 224 tests with seven skips and zero failures in 21.2 seconds.

Coverage includes strict signed/decimal parsing, same-line or separate-line labels, missing/ambiguous fields, capture-time ordering, cumulative and immediate semantics, terminal accounting, elapsed-time returns, likelihoods and PPO derivatives, real shared-weight changes, exact next-update checkpoint continuation, context widening, precision casting, reward-mask leakage across resize/color paths, and negative/moving source coordinates.

The actual video-training regression continues demonstration → RL → demonstration → RL with stale checkpoints still present, verifies the selected weights and authorized controls, and preserves existing brains. Persistence tests exercise incomplete publications, failure of the final profile commit using an immutable destination, rollback, immutable-version activation, and recovery after an interrupted checkpoint/profile swap. Overlapping stop requests and a cancelled queued autosave complete without losing the checkpoint.

Certificate-signed screen fixtures exercised native Retina capture → Vision OCR → HID input → reward → PPO update. All **32 completed episodes** across the four production benchmarks matched the fixture's terminal rewards exactly, including its final penalty. Separate runs verified timeout episodes around 4.1 seconds, changing lengths from 10 to 3 seconds (episodes around 3.1 seconds), pause/resume with optimizer counters retained, panic release/save, and detection loss pausing/saving after one second. Selection tests confirmed Enter, Escape, restored visibility, and native detection. AI actions began after readiness, not during reset. The final installed-app smoke test completed two further 4.1-second timeout episodes, performed four learning updates, and stopped/saved through Panic Stop. Beginner/Advanced and dashboard views were visually inspected. Mixed-display/negative-coordinate behavior has synthetic coordinate coverage; the live fixture used a Retina display.

## Controlled learning

These deterministic visual tasks use actual PPO gradients and frozen initial-policy baselines. Keyboard score measures correct exclusive control selection for two visual cues; mouse score measures closeness to the two target positions. They establish learning on the controlled tasks, not success in arbitrary games.

| Task | Seed | Frozen baseline | Trained score |
| --- | ---: | ---: | ---: |
| keyboard | 7 | 0.5000 | 1.0000 |
| keyboard | 42 | 0.5000 | 1.0000 |
| keyboard | 101 | 0.0000 | 1.0000 |
| mouse | 7 | 0.7001 | 0.9917 |
| mouse | 42 | 0.6990 | 0.9936 |
| mouse | 101 | 0.7018 | 0.9969 |

## Measured live limits

Measured on **Apple M3 Max, 36 GiB, macOS 27.0.1**. Each production run lasted approximately 60 seconds, used Balanced PPO defaults (rollout 256, minibatch 32, four epochs, learning rate 0.0001, compute share 0.35), and completed eight episodes. Compact uses 96×64 grayscale, Small architecture, no past frames; default uses 640×360 color, Balanced architecture, and four half-size past frames. Both request 30 Perception FPS; the suffix is requested Action FPS.

| Profile / requested action FPS | Actions/s | Perceptions/s | Reward reads/s | Head / vision p95 ms | Sampled resident peak MB | Rollout updates / optimizer steps | Old actions excluded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compact30 | 26.7 | 23.3 | 23.3 | 4.0 / 4.4 | 377 | 7 / 24 | 66 |
| compact60 | 52.9 | 24.7 | 27.0 | 4.5 / 5.0 | 359 | 13 / 28 | 1094 |
| default30 | 26.4 | 23.1 | 22.6 | 9.0 / 10.8 | 885 | 4 / 2 | 323 |
| default60 | 53.0 | 24.8 | 26.9 | 8.6 / 10.8 | 715 | 10 / 15 | 1432 |

Action throughput includes reset and inter-episode pauses. p95 is the final rolling window of up to 512 compute measurements, rather than end-to-end input latency. Memory is process resident size sampled once per second. Reward reading includes reuse of unchanged pixels on fresh capture/idle evidence. Additional short-rollout stress runs updated during episodes at both requested action rates without the inference watchdog firing.

OCR did not sustain every requested 30/60 Hz action tick. Cumulative rewards therefore used capture-time intervals across intervening actions; immediate mode omits unread intervals. The larger profiles and higher action rates excluded more stale experience under the bounded learner budget. These are measured limits: a requested FPS is not a throughput guarantee, and large models may need lower rates, less visual history, smaller vision, or adjusted optimization settings. Terminal accounting remains distinct from conservative stale-policy/KL admission to optimization.

Window movement is followed; resize, scale change, source loss or invalid detection pauses instead of silently training against changed geometry. The external game state is not checkpointed; resuming starts a fresh episode. Terminal text must remain visible until the next start key, and cumulative counters must reset only at episode boundaries. See the [environment contract](reinforcement-learning.md).

## Signing and artifacts

Version is 2.60, build 38. The existing transactional build/install/package workflow is retained. Signing regressions and shell syntax checks pass. App and DMG use the existing valid certificate; installed and packaged app designated requirements are compared with the original installation. The dormant ad-hoc signing path is removed. Separate certificate/key exports, machine-local pins, user training data, and local validation artifacts are excluded from distribution archives. Signed bundles retain the public verification certificate required by their signature.

The release supplies the signed DMG, source archive, and SHA-256 checksums. Signed builds do not add notarization to the existing workflow. [Machine-readable aggregate results](validation-2.60.json) contain only fixture-derived measurements.
