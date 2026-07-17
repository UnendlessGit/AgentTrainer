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

The current tree implements six high-impact corrections:

1. Live recurrent history now contains only thresholded actions that were
   permitted and capable of executing. Raw probabilities, blocked keys,
   disabled channels, stale one-shot deltas, and duplicate modifier paths no
   longer become a hidden feedback channel unavailable during training.
2. Single-recording validation now has an embargo between train and validation.
   Validation begins only after its action-history window and both perception
   frames are disjoint from training. If the recording is too short for an
   honest split, all samples train and no misleading validation score is shown.
   Disabled or blocked controls no longer erase validation availability or
   consume its representative-example budget.
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

## Current learning system

### Data and targets

- Screen frames and synchronized human input are converted into a packed UInt8,
  memory-mapped cache.
- Each action-rate row references a current and previous perception frame instead
  of duplicating images.
- The target is causal: a frame is paired with the control interval immediately
  after it.
- The model receives current vision, signed frame difference, generated X/Y
  coordinates, and recurrent ground-truth action history.
- Outputs cover absolute and relative mouse movement, scroll, keyboard, mouse
  buttons, and Command/Option/Control modifier state.

### Model and objective

Policy v4 uses a four-stage GroupNorm/SiLU convolutional spatial encoder plus a
GRU/LSTM action-history branch and separate action heads. Training uses:

- Smooth-L1 losses for continuous controls.
- Class-balanced binary cross-entropy for sparse keys, buttons, and modifiers.
- Extra loss weight around state transitions and active relative movement.
- Loss masking for unseen, blocked, or disabled outputs.
- Independent 50% masking of the complete ground-truth history branch.
- Global gradient clipping and resumable AdamW.

These weights make rare demonstrations matter more, but they are **loss weights,
not rewards**. The policy is still trained to reproduce demonstrated actions.

### Validation and publication

Multiple recordings use whole-recording validation where possible. A lone
recording uses a disjoint temporal tail with the embargo described above.
Validation evaluation selects a deterministic representative set containing
rare positives, transitions, active deltas, and timeline coverage. The lowest
comparable held-out loss is published as the runnable brain while the latest
optimizer state remains available for exact continuation.

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

### Stage 1: stronger supervised evaluation

Keep the corrected imitation learner as the stable base. Add per-head metrics in
addition to aggregate validation loss:

- Precision, recall, and F1 for keys/buttons/modifiers.
- Press and release timing error.
- Absolute cursor error and active-only delta/scroll MAE.
- False-action rate during demonstrated idle periods.
- Metrics separated by recording and by action capability.

This is the fastest way to expose regressions hidden by one scalar loss.

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

1. Per-head validation metrics and a regression report.
2. Intervention/correction capture with exact proposed-versus-executed actions.
3. Versioned rollout storage and task-specific outcome verifiers.
4. Offline critic plus bounded advantage-weighted imitation.
5. Only after offline gains are repeatable: sandboxed online fine-tuning.

This sequence preserves AgentTrainer's privacy and safety model, makes each step
measurable, and ensures that adding “reinforcement” means learning from verified
consequences rather than disguising another imitation loss as reward.
