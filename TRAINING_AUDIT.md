# AgentTrainer Training Audit and Roadmap

This document records the July 2026 review of AgentTrainer's data, model,
training, checkpoint, validation, and runtime paths. It distinguishes changes
implemented in the current tree from future work that needs new data contracts.

## Executive conclusion

AgentTrainer is currently an imitation-learning system, not a reinforcement-
learning system. Its strongest path to becoming smarter sooner is to improve the
quality and honesty of imitation first, then add correction and outcome data in
stages. A generic online RL loop that explores through unrestricted macOS input
would be unsafe, sample-inefficient, and easy to reward-hack.

The current tree implements twenty-eight high-impact corrections:

1. Policy v9 removes demonstrated and predicted action history from the model
   entirely. Policy v7 could copy a held key through teacher-forced validation,
   then enter an all-released absorbing state because live inference starts
   with released controls. Visual memory now supplies all temporal model
   context; held keyboard/button state remains exclusively in `InputInjector`.
2. Single-recording validation now has an embargo between train and validation.
   Validation begins only after its complete configured visual-memory window is
   disjoint from training. If the recording
   is too short for an honest split, all samples train and no misleading
   validation score is shown. Disabled or blocked controls no longer erase
   validation availability or consume its representative-example budget.
3. AdamW warmup is now one epoch, bounded to 10...500 optimizer steps, instead
   of always consuming 500 steps. The selected value is checkpointed, and old
   checkpoints retain their historical 500-step schedule.
4. Each epoch uses deterministic salience-balanced ordering. Control transitions
   and active mouse/scroll rows are spread across batches while every training
   row is still used exactly once; there is no oversampling or added GPU work.
5. Exact resume now records the ordered recording IDs and sampling-strategy
   version. A reordered data set can warm-start weights, but cannot silently
   pretend that an old mid-epoch optimizer offset is exact.
6. Fine-tuning re-evaluates the selected brain on the current validation set
   before training. An old validation score from different recordings, targets,
   or split can no longer prevent a genuinely improved fine-tuned brain from
   being selected. The validation contract is versioned, allowing compatible
   optimizer state to resume while older best-score metadata is recalibrated.
7. Policy v6 made two complete visual architectures first-class; Policy v9
   retains both. Hybrid keeps the compact CNN-to-token path; Pure Transformer
   uses direct lossless-edge-padded ViT patches, 2-D positions, and full patch
   attention with no CNN feature hierarchy. Both retain temporal vision, a
   readout token, the same action heads, and no action-input shortcut.
   Superseded brains are archived by their own schema while
   source recordings remain available.
8. New training uses bounded cosine restarts plus a validation-driven plateau
   envelope. A global non-zero floor prevents the effective rate from
   asymptotically collapsing, and all scheduler state is exactly resumable.
   Compatible current checkpoints preserve their stored schedule; profiles
   whose older brains are archived receive adaptive defaults, while explicit
   user choices survive migration.
9. Transition loss receives the real preceding cached action as loss metadata,
   independently of model input. Perception-only training therefore does not
   label every frame of a held control as a fresh transition, while the
   preceding action still cannot become a predictive shortcut.
10. Sparse binary heads add configurable focal weighting on top of class and
    transition balancing, concentrating work on errors rather than easy idle
    negatives.
11. Whole-recording validation now targets sample fraction rather than choosing
    an arbitrary equally weighted recording. Representative evaluation scales
    to 8,192 rows, includes anchors from every held-out recording when budget
    permits, and persists per-head confusion metrics plus continuous MAE and
    idle false-action rate, and correctly directed executable motion recall.
    Best-brain recommendation scores demonstrated execution behavior so a lower
    aggregate loss cannot disguise a collapsed binary or continuous head, while
    all quality thresholds remain advisory and never prevent saving or Run.
12. Epoch-average training loss, effective learning rate, scheduler scale, and
    their bounded histories are checkpointed and visible alongside batch and
    validation curves. Plateau decisions no longer depend on one noisy shuffled
    batch when validation is unavailable.
13. Video sampling now pipelines three ordered Metal preprocessing jobs so
    VideoToolbox decode, the unchanged resize/chroma/quantization kernel, and
    buffered writes overlap. Regression tests require byte-exact equality.
14. Dataset cache identity excludes the legacy action-history field. Model
    architecture never changes sampled pixels, so Policy-v9 migration and
    Hybrid/Pure edits reuse the same lossless video cache.
15. Each optimizer batch gathers current/multi-lag frames, availability, target,
    and real previous target in one mapped traversal instead of separate passes
    and repeated frame-index lookups. One zero row preserves the internal MLX
    call signature without becoming a token or counted input.
16. Held-out evaluation is compiled and shares one policy forward between loss
    and predictions, removing the previous duplicate network evaluation without
    changing validation examples or math.
17. Binary balance now exactly equalizes effective active/inactive mass after
    the 3× press/release transition bonus. Objective v2's smoothed square-root
    correction left the best constant prediction below 0.5 for every sparse
    control, directly rewarding "always off." Keyboard/modifier evidence accepts
    either four independent presses or 0.5 seconds of accumulated held targets,
    so one accidental tap remains blocked without discarding an intentional
    sustained movement key.
18. Binary loss uses `batch rows × active outputs` as its fixed denominator.
    Dividing by the current batch's weight sum canceled rare-positive authority
    exactly when a positive appeared, after which negative-only batches pulled
    the output back toward idle. Focal difficulty weights remain numerator-only,
    so making examples easier lowers the reported objective. Binary loss remains
    Float32 under compact model precision so legitimate rare-control weights
    cannot overflow. Validation exports Mouse, Buttons, Scroll, Keyboard, and
    Modifiers components from the same compiled forward.
19. Each supported binary output receives a conservative validation-calibrated
    runtime threshold from 0.50 to 0.95. Sparse evidence stays at 0.5, support
    confidence shrinks large changes, and physical execution consumes one dense
    threshold table without searching report objects at action FPS.
20. Training-objective schema 4 separates corrected sparse-control math from
    model and video-cache contracts. It retains objective-v3 binary behavior and
    adds exact per-axis transient active/idle balance plus beta-normalized
    Smooth L1. Shape-compatible weights and recordings remain intact across an
    objective-only change, while an incompatible optimizer and loss history
    restart safely. The Training UI exposes binary and transient evidence,
    ignored controls, five head losses, per-key confusion changes, and explicit
    zero-recall/zero-motion warnings instead of hiding them in one scalar.
21. Policy v9 retains Policy-v7's replacement for one-frame-only motion:
    bounded multi-timescale
    visual memory. The default remembers perception lags 1, 5, 9, and 13 as
    signed visual fields plus explicit availability, then fuses them pointwise
    to eight values per pixel before either spatial architecture. Current pixels
    bypass this compression, visual memory adds no Transformer tokens, and
    older slots receive structured training dropout while the immediate frame
    remains reliable. Dataset batches derive every lag from the existing
    deduplicated observation index, live inference uses a fixed-capacity ring,
    and single-recording validation embargoes the complete visual context.
22. `continuousBalancePlan` measures every configured relative-camera and
    scroll axis on the immutable training split. Active rows receive exact
    `idle / active` authority (capped at 4,096), unsupported axes are masked,
    continuous loss stays Float32, and dividing Smooth L1 by beta prevents a
    normalized one-pixel camera target from becoming numerically irrelevant.
23. Mouse-mode diagnosis is now three-state. Locked positions with real deltas
    vote Game Camera, changing absolute positions vote Absolute Cursor, and
    insufficient/stationary recordings do not vote. Classified recording count
    is primary and classified event count breaks ties, so a long stationary
    capture can no longer force the wrong head.
24. Periodic and manual-pause publication preserves the exact current
    continuation state as a runnable version. Paused brains retain optimizer,
    scheduler, batch-offset, and random state and are not auto-pruned; periodic
    autosaves remain prunable. Mid-epoch versions omit validation fields rather
    than attaching an older report to unvalidated current tensors.
25. Held-out continuous telemetry now measures execution semantics: an active
    target axis counts as recalled only if the prediction has the correct sign
    and rounds to at least one runtime unit at default sensitivity. The UI and
    recommendation score therefore expose the exact “no mouse movement”
    behavior without blocking the user's ability to save or run.
26. Policy v9 closes a second constant-output route found in a fresh Policy-v8
    FSD V2 run. After final Transformer normalization, the learned readout is
    averaged with the parameter-free transformed-visual mean and first
    visual-token anchor. This creates an unavoidable perception path without
    adding parameters, changing tensor shapes, or increasing attention work.
    The fixed anchor avoids the nondeterministic tied/parallel gradients of
    hard-maximum or dynamic soft pooling and preserves exact optimizer resume.
27. Game Camera conversion is now one shared execution contract. Validation
    measures fixed 0.5–4.5 pixel pre-rounding candidates and saves the smallest
    deadzone that retains at least 5% executable motion while keeping idle
    continuous execution at or below 10%. `InputInjector` uses that exact
    immutable value; legacy reports fall back to a conservative 1.5 pixels.
28. Best-brain recommendation now optimizes deployment behavior instead of assuming
    the lowest weighted loss is the best agent. Every supported binary control
    contributes equal-weight calibrated F1; camera and scroll contribute
    executable recall discounted by idle execution. Loss remains the optimizer
    scheduler metric and close-score tie-breaker.
29. Validation contract 7 made every execution-quality threshold advisory.
    Low recall, inactive supported controls, continuous-motion collapse, idle
    jitter, and regression checks remain visible diagnostics, but no longer
    prevent Pause & Save, autosave, activation, or Run. Every finite,
    structurally compatible brain remains under user control.
30. Validation contract 8 closes the remaining zero-split runtime discrepancy.
    With no held-out report, Game Camera previously fell back to a conservative
    1.5-pixel pre-rounding deadzone; a 5% split could calibrate that value down
    to 0.5–1.0 pixels. The same small, improving prediction could therefore be
    physically discarded at 0% and visibly move at 5%, making validation appear
    to change learning. Exact zero-split snapshots now run a bounded 2,048-row
    in-sample execution calibration. Its persisted scope is explicit, it never
    becomes validation loss/history, scheduler input, or best-brain evidence,
    and it only supplies runtime thresholds plus transparent diagnostics.

## FSD V2 regression diagnosis

The July 26 investigation used FSD V2's exact 310,735-sample cache, selected
recordings, current weights, best weights, split, preprocessing, restrictions,
and calibrated thresholds:

- Teacher-forced validation reported about 92% aggregate binary F1 and correctly
  recalled 2,956 of 3,086 held-out W targets.
- Replacing the action-history input with released zero state—the state in
  which a live run begins—reduced every supported key's recall to 0% for both
  current and best weights. Outputs became image-independent constants; W was
  approximately 0.484 and therefore below its live threshold.
- Warm-start experiments that masked history or configured history length zero
  for 4,000 more optimizer steps remained at 0% cold-start recall. The v7 head
  was already in the shortcut basin.
- A fresh random perception-only model on the same cache learned visual key
  separation within 4,000 steps. After one epoch, held-out W recall reached
  81.6% and the other frequent movement keys produced nonzero recall. This
  isolates the issue from recording type and user training settings.
- A complete fresh Policy-v8 run exposed a second, seed-sensitive failure: its
  learned readout converged to a near-constant representation, leaving five of
  six demonstrated keys inactive and producing only about 13% aggregate key
  recall. Removing action history was necessary, but not sufficient.
- The 343 selected recordings contained 524,397 move events, 170,090 nonzero
  raw deltas, but only 802 absolute-position changes. The old binary classifier
  treated every unclassifiable stationary recording as a cursor vote, selected
  Absolute Cursor, and ran a head whose target was almost fixed at screen
  center. Meanwhile the v7 relative head emitted zero despite active targets.

These controls establish three independent root causes: teacher-forced action
tokens caused the key/button idle feedback loop, a fully learned readout could
still become a constant bottleneck, and underweighted transient loss plus
two-state mouse classification hid camera motion. Model-contract breaks are
required; warm-starting the collapsed v7/v8 tensors is not a reliable repair.
V7 and v8 artifacts are therefore archived recoverably and FSD V2 retrains from
fresh Policy-v9 weights while reusing its original recordings and cache.

The final deterministic Policy-v9 verification trained against an exact linked
copy of FSD V2's live recordings/cache and published step 25,500. On 6,900
fixed representative held-out rows it produced 2,967 key true positives and
2,779 false negatives (51.6% aggregate key recall and about 42.5% aggregate key
F1). Every supported key executed, including 15 true Space activations. The
brain selected a 1.0-pixel deadzone and achieved 22.9% executable Game Camera
recall with 5.39% idle execution. Cold/released history and the compatibility
history argument produced identical predictions, confirming that the original
live-start feedback loop is absent.

## Current learning system

### Data and targets

- Screen frames and synchronized human input are converted into a packed UInt8,
  memory-mapped cache.
- Each action-rate row references compact perception indices instead of
  duplicating images. Training derives the configured multi-lag memory window
  from those indices without changing the sampled-video cache.
- The target is causal: a frame is paired with the control interval immediately
  after it.
- The model receives exact current vision, signed differences at several past
  perception lags, per-slot availability, and generated X/Y coordinates.
- Outputs cover absolute and relative mouse movement, scroll, keyboard, mouse
  buttons, and Command/Option/Control modifier state.

### Model and objective

Policy v9 shares visual-memory fusion, readout, pre-normalized
attention/gated-SiLU blocks, fusion, and action heads across two model families.
Current pixels bypass one pointwise projection that compresses all remembered
differences and availability to eight values per pixel. This keeps memory work
linear and leaves the attention sequence unchanged. The final normalized
readout is averaged with the transformed visual-token mean and first visual
token, so the action heads always receive direct perception evidence.
Hybrid pools a four-stage GroupNorm/SiLU CNN into learned X/Y-aware spatial
tokens plus mean/max global tokens; Balanced uses 11 total tokens and 121
attention pairs per block. Pure Transformer directly projects non-overlapping
patches, edge-pads only the incomplete right/bottom patches so no source pixel
is dropped, adds separable 2-D positions, and keeps every patch in attention.
At 640×360, Pure Balanced has 240 visual patches, 241 total tokens, and 58,081
pairs per block. Training uses:

- Ordinary Smooth L1 for absolute cursor and beta-normalized, exact
  active/idle-balanced Smooth L1 for relative camera and scroll axes.
- Evidence-gated focal binary cross-entropy for sparse keys, buttons, and
  modifiers (gamma zero retains ordinary BCE). Positive corrections exactly
  balance press/release-weighted active and inactive mass, while focal weights
  affect the numerator but not the fixed active-decision denominator.
- Extra loss weight around binary state transitions.
- Loss masking for blocked/disabled outputs and for keyboard/modifier controls
  with neither four independent presses nor 0.5 seconds of held targets.
- Structured dropout of older visual-memory slots while retaining the immediate
  predecessor and masking each matching availability field.
- Global gradient clipping and resumable AdamW with bounded cosine restarts and
  validation/epoch-loss plateau control. Weight decay applies to capacity-
  bearing kernels/matrices, not biases, normalization affine values, or learned
  token identities.

These weights make rare demonstrations matter more, but they are **loss weights,
not rewards**. The policy is still trained to reproduce demonstrated actions.

### Validation and publication

Multiple recordings use whole-recording validation where possible, balancing
the requested held-out sample fraction while retaining enough repeated-press or
held-duration evidence for every learnable control. A lone recording extends
its contiguous training prefix only as far as needed to retain the same evidence,
then uses a disjoint temporal tail with the embargo described above. Validation
selects a deterministic representative set containing rare
positives, transitions, active deltas, per-recording anchors, and timeline
coverage. It records five action-family losses, per-output 0.5 and calibrated
confusion counts, binary precision/recall/F1/false-positive rate, continuous
error, idle false-action rate, and executable continuous recall in addition to
aggregate weighted loss. Threshold calibration uses fixed-
size score histograms and requires meaningful positive/negative support. The
highest deployment score is recommended after every finite candidate has been
considered; per-control binary execution, transient recall, idle stability, and
regression values are advisory diagnostics. Comparable held-out loss breaks
close-score ties and independently drives the scheduler, while the latest
optimizer and scheduler state remain available for exact continuation.
When the requested split is zero (or honest held-out context is unavailable),
every row remains in training. Exact runnable snapshots still receive a bounded
representative in-sample pass so binary thresholds and the Game Camera deadzone
match the saved tensors. The report is explicitly marked as calibration rather
than held-out quality; epoch-average training loss remains the scheduler metric,
and no best-brain comparison is made.

## What “keep improving” can and cannot mean

The former inverse-square-root schedule created an artificial wall: at millions
of steps its effective rate could become too small to change behavior. Bounded
restarts remove that failure mode. They cannot manufacture information absent
from the demonstrations. Once a policy has fit a fixed data distribution,
additional passes can plateau or overfit even with an ideal optimizer.

Continued real-world improvement therefore needs at least one of:

- new demonstrations covering missing states and failures;
- human corrections collected on states the policy itself visits;
- externally verified outcome-labelled rollouts;
- a better task-conditioned representation when one profile spans incompatible
  behaviors.

Training should stop or reduce its envelope when held-out quality stops
improving; blindly maximizing step count is not an intelligence objective.

## Why simple rewards and punishments are not enough

Reinforcement learning requires transitions of the form:

```text
(state, policy action, resulting state, reward, terminal status)
```

Existing recordings provide observations and the human's action, but not the
counterfactual result of an action chosen by the AI. Assigning positive reward
to demonstrated keys and punishment to other keys would merely re-label
behavior cloning, punish valid alternative solutions, and strongly reward
doing nothing in this sparse action space.

Rewards should measure outcomes, interventions, and safety—not whether an
individual key exactly matches one human trace. They must also be evaluated
after the action has affected the target application.

## Recommended hybrid learning path

### Stage 1: stronger supervised evaluation — baseline implemented

The current tree now records the first useful per-head baseline in addition to
aggregate validation loss:

- Implemented: precision, recall, F1, and false-positive rate for aggregate
  binary output and keys/buttons/modifiers.
- Implemented: absolute cursor error, active-only delta/scroll MAE, idle
  continuous false-action rate, and correctly directed executable motion recall.
- Next: press/release timing error and reports separated by recording and action
  capability rather than only aggregate/per-head totals.

This is the fastest way to expose regressions hidden by one scalar loss.

## Population training assessment

A population can help select learning rate, focal gamma, cycle length, dropout,
and model size, but running many full policies concurrently on one Mac would
multiply unified-memory pressure and compete for the same Metal device. It also
invites validation overfitting if every generation is selected on one fixed
held-out subset. Population training does not create new behavioral information
and therefore cannot by itself solve the demonstration ceiling.

The appropriate design is sequential or low-concurrency ASHA/successive halving:

1. Freeze one immutable starting brain and deterministic train/validation
   contract.
2. Give 4–8 mutations a small equal optimizer-step budget, preferably one at a
   time; never share optimizer moments between candidates.
3. Rank on a constrained score: held-out loss plus explicit penalties for
   per-head F1 regression and false actions. Reject any candidate that loses a
   demonstrated capability, regardless of aggregate score.
4. Promote the best half to a larger budget, then re-evaluate finalists on a
   second recording/task fold that was not used for promotion.
5. Preserve the original and winner as immutable versions and delete only
   disposable candidate checkpoints.

This should be implemented after per-recording reports and correction data, not
as simultaneous in-process PBT. Until then, the new adaptive single-policy
scheduler captures the most valuable LR automation without multiplying memory
or fitting noise across a population.

### Stage 2: correction learning

Run the policy only in a disposable or explicitly scoped target. Let the human
take over when behavior is wrong, and record:

- The observation and exact brain version.
- The policy's proposed action and the action permitted by the firewall.
- The human's replacement action.
- Intervention start/end and terminal reason.

Retrain on the states the policy actually visits, emphasizing corrected
segments while retaining the original demonstrations. This is a DAgger-like
solution to the distribution shift between clean demonstrations and states
caused by the learner's own mistakes. See [Ross, Gordon, and Bagnell,
2011](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf).

This stage is likely to deliver more value per recorded minute than unconstrained
RL because a human correction supplies a dense, unambiguous target.

### Stage 3: outcome-labelled rollouts

Introduce a versioned rollout format separate from immutable demonstrations. A
rollout episode should include:

```text
episode ID and task adapter version
brain/model contract and restrictions
ordered observation references
proposed, permitted, and actually executed actions
human corrections and intervention intervals
reward components and their source
success/failure/aborted terminal status
```

Start with explicit task adapters rather than a universal screen-based reward.
A reasonable normalized default is:

```text
+1.00  verified task success
-1.00  verified failure or safety violation
-0.25  human intervention
-0.001 per decision step, only to prefer equally successful shorter paths
```

Those numbers are starting scales, not universal truth. Each reward component
must be logged separately so changing the formula can rebuild targets without
discarding rollouts. Never infer success solely from the policy's own output.

### Stage 4: support-constrained policy improvement

Begin offline, using demonstrations plus corrected/outcome-labelled rollouts.
For this mixed continuous/binary action space, a practical first objective is
advantage-weighted behavior cloning:

```text
Ltotal = Ldemonstration + lambdaAdv * LadvantageWeighted
                         + lambdaSafety * Lsafety
```

A critic estimates whether an executed action led to better-than-expected
outcomes, and its bounded advantage changes the weight of that **observed**
action. The ordinary imitation term remains as an anchor against forgetting and
out-of-distribution actions. AWAC and IQL are useful primary references for this
family of conservative policy improvement: [AWAC](https://arxiv.org/abs/2006.09359)
and [IQL](https://arxiv.org/abs/2110.06169).

Do not start with general online PPO against the desktop. It would require many
on-policy trials, can learn shortcuts in an incomplete reward, and would explore
actions outside demonstrated support. Any later online fine-tuning belongs in a
resettable environment with the existing restrictions and learned-key firewall
kept outside the trainable model.

## Anti-loophole rules for rewards

- Success must come from an external verifier or explicit human label.
- Terminal failure and safety penalties cannot be masked by a high intermediate
  score.
- Log reward components before summing them; never store only a mutable total.
- Compare task success, intervention rate, and safety events separately from
  return so one metric cannot conceal another.
- Hold out tasks/seeds, not adjacent frames from the same attempt.
- Cap advantage/reward weights so a few noisy rollouts cannot erase broad
  demonstrations.
- Mix a fixed demonstration replay fraction into every policy-improvement run.
- Never train away the runtime safety firewall or output restrictions.
- Treat aborted, crashed, timed-out, and manually stopped episodes explicitly;
  silently dropping them creates survivorship bias.

## Priority order

1. Add per-recording/timing regression slices to the implemented per-head report.
2. Intervention/correction capture with exact proposed-versus-executed actions.
3. Sequential ASHA over safe supervised settings and a second selection fold.
4. Versioned rollout storage and task-specific outcome verifiers.
5. Offline critic plus bounded advantage-weighted imitation.
6. Only after offline gains are repeatable: sandboxed online fine-tuning.

This sequence preserves AgentTrainer's privacy and safety model, makes each step
measurable, and ensures that adding “reinforcement” means learning from verified
consequences rather than disguising another imitation loss as reward.
