# Reinforcement learning in AgentTrainer 2.61

AgentTrainer uses PPO with GAE for local live training. The choice favors conservative updates, mixed simultaneous key/button actions and mouse controls, and compatibility with the existing Policy v9 brain. PPO is on-policy and is not the most sample-efficient algorithm for every task. Demonstrations remain useful for establishing behavior before live exploration.

## Environment contract

Use one small region with white text on black, containing two labeled lines:

```text
Reward: -125.5
Episode Length: 60
```

Test zero and terminal values as well as a positive counter before training. If an isolated integer zero is unreliable with a particular font/scale, display `0.0`. Accurate recognition retries smaller copies when native text is incomplete; it retains the configured confidence threshold and never guesses missing digits.

Rewards may be signed, zero, or decimal. The decimal separator is a period. Episode length is a nonnegative fixed maximum in seconds, not a countdown. AgentTrainer tracks elapsed time from readiness; changing the length adjusts that same deadline. Zero ends the episode, including the reward in that image.

After the start key, reset the cumulative counter to zero and show a positive episode length. Keep this acknowledgment visible long enough for detection. Default settling is 0.1 seconds and reset timeout is 10 seconds. Keep terminal text visible until the next start key. The end key is optional and may equal the start key; AgentTrainer sends each at its own lifecycle boundary. Default inter-episode delay is 0.5 seconds.

Unexpected counter decreases are genuine negative rewards, not guessed resets. Signal an environment reset with the explicit episode protocol. Two text values cannot identify hidden resets or distinguish separate identical event messages; immediate mode instead defines the displayed number as the reward for each valid action interval.

## Capture and accounting

ScreenCaptureKit captures the reward crop at native point/pixel scale, separately from model vision, targeting Action FPS. Apple Vision first locates labels and then reads numeric crops. Unchanged pixels reuse the validated recognition on fresh capture/idle evidence. Numeric ambiguity or layout changes trigger accurate recognition. Both fields come from one source image.

Capture time determines reward ordering. Recognition completion time only measures processing latency. Cumulative differences span the executed actions between accepted captures, with duration-adjusted discounting and GAE and joint behavior likelihoods. Immediate gaps exclude unread intervals and break their training segment. Neither path inserts zero for OCR failure. One uncommitted interval retains rewards arriving after the last action release. A terminal reading closes it once before resetting.

The model never receives reward pixels: the Metal preprocessor masks the source region with a sampling guard band before creating current or historical observations. The same region can be masked during ordinary Run for an RL brain. App/HUD windows are excluded from display capture. Window regions use source-local coordinates; movement updates input geometry, while resize, scale change, disappearance, stale capture, or detection loss pauses safely.

## Learning and performance

PPO uses clipped likelihood ratios, a critic, GAE, optional advantage/reward normalization, entropy regularization, gradient clipping, and KL early stopping. Only explicit allowed controls participate in the distribution; shared channel restrictions still apply. Start/end keys are reserved. Mouse samples use the existing decoder, including movement intent, so their stored latent densities remain well-defined through clipping and exact-stop behavior.

The learner owns its model, compiled functions, random state, and optimizer on a serial queue and independent MLX stream. Inference owns another model. Evaluated CPU buffers cross the ownership boundary; lazy arrays and mutable weights do not. Historical embeddings are refreshed against the new revision before use. Original pixels/control/timing records are used for gradients.

Compiled microbatches accumulate the configured minibatch objective. The live learner yields according to its compute budget and gives overrunning inference extra time. At most one revision of behavior lag is accepted, and KL checks reject excessive policy drift. Recent collection is bounded; old untrained actions may be excluded when the learner cannot keep up, and the dashboard reports that exclusion. Terminal intervals and the uncommitted tail are protected; collection pauses if those alone exhaust the budget. This preserves responsive execution without promising a particular FPS for arbitrary architecture, history, or vision settings.

Default Balanced settings: learning rate 0.0001, rollout 256 actions, minibatch 32, four optimization epochs, discount 0.999, GAE lambda 0.95, clip 0.2, value coefficient 0.5, gradient limit 0.5, and target KL 0.02. Stable uses smaller updates/less exploration; Exploratory uses more exploration and a shorter rollout. Beginner sliders edit the same effective settings shown in Advanced mode.

RL uses a constant optimizer learning rate. Applied updates, optimizer steps, attempted rollouts, zero-step KL rejections, stale/buffer exclusions, and unread actions are distinct metrics. A zero-step rejection does not age the actor revision. Sampling and replay share the effective keyboard/cursor permissions, channels, allowed controls, and modifier restrictions. Clicks and scroll are contained even when movement is disabled; scroll events target the actual intended cursor location.

Discount and lambda are per nominal action, with elapsed-time adjustment. Changing FPS changes their time horizon: to preserve it, use `new = old^(oldFPS/newFPS)` for both coefficients. Joint likelihoods span all decisions in a cumulative OCR interval; longer intervals can reach clipping/KL limits sooner. Reward normalization currently uses RMS interval reward, so changing reward grouping also changes its scale. Short performance screens cannot establish learning equivalence across these settings.

## Persistence and compatibility

Policy format remains `AgentTrainer.Policy.v9`; demonstration cache schema remains 15. Profiles add optional RL configuration/progress and versions add an optional RL summary. Retired experimental RL/WebSocket settings remain ignored. No external reward transport is required.

Each saved RL version includes standalone policy weights, a trainer sidecar, critic/exploration weights, optimizer state, update/random/normalization state, and collected experience. `RLCheckpoint` is published transactionally with the active version reference. Loading an RL version restores its own checkpoint. If a process exits between a checkpoint swap and profile publication, the immutable active version supplies its last committed resume state. Failed publication leaves the previous active brain usable.

Checkpoint lineage prevents a stale demonstration or RL optimizer from overwriting the currently active policy. Compatible same-method resume restores its optimizer state; changes to the learned action distribution or another method's weights start a new optimizer sequence using the active policy. UI/capture/autosave preferences alone do not reset optimizer state. Context additions map existing projection columns and initialize new columns to zero. Architecture/vision incompatibilities retain the existing explicit reset workflow.

Pause, stop, panic, and quit release inputs first and then save. The external game is not serialized. Resume starts a new episode while preserving learned weights, optimizer progress, valid pending experience, and lifetime statistics. Default autosave is 60 seconds with ten periodic versions retained; manual/active/protected versions are protected.

Learning-semantics version 2 retains Policy v9. Older sidecars retain their active policy and compatible critic/exploration, but start fresh optimizer/normalization state and exclude old pending targets. New checkpoints retain the already-selected epoch order and resume partial minibatches without an extra shuffle. Experience limits include pixels, timing, controls, context, and action storage.

## Validation tools

`./test.sh` runs the isolated regression suite, including parser/reducer timing, compiled runtime state capture, likelihoods, PPO weight updates, exact optimizer continuation, source-coordinate/mask coverage, context widening, and RL publication/activation checks.

Set `AGENTTRAINER_RUN_RL_LEARNING_TESTS=1` to run controlled visual keyboard and mouse learning across seeds 7, 42, and 101. These compare trained policies against their frozen initial baselines. The tests exercise actual PPO updates, not a hand-written policy.

`scripts/build-rl-fixture.sh` creates a certificate-signed local game under ignored `work/rl-validation`. It uses A/D as actions, R to reset, E to end, and displays the screen reward protocol. Its `AGENTTRAINER_FIXTURE_SCENARIO` selects death, timeout, or dynamic-length behavior. The app's opt-in `AGENTTRAINER_RL_VALIDATION=1` harness requires an isolated `AGENTTRAINER_WORKSPACE_ROOT` containing `rl-validation` and an explicit report path. Normal launches do not run it. It does not reset permissions, change signing identity, or use the user's model library.

For the broader audit fixture, build with `AGENTTRAINER_FIXTURE_AUDIT=1` and follow the reproducible setup in [2.61 validation](release-2.61.md#reproducing-the-audit). `scripts/run-rl-audit.py` supports training, frozen evaluation, deployment smoke checks, and a scripted fixture-contract check. Frozen evaluation loads a fixed snapshot, invokes no learner, and publishes no checkpoint. Its deterministic and seeded stochastic episodes have separate results. The launcher enforces a cumulative 3,600-second wall-clock experiment budget under the isolated audit directory.

Controlled-task learning and a local screen fixture establish implementation behavior; they do not establish success in every game. Actual capture/action/reward throughput and resource limits must be measured on the target Mac.

## Primary references

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Apple Vision text recognition](https://developer.apple.com/documentation/vision/vnrecognizetextrequest)

Release-specific measurements are recorded in [2.61 validation](release-2.61.md); [2.60 validation](release-2.60.md) retains earlier results.
