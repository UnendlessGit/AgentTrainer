# AgentTrainer 2.61 — build 39

This audit repairs concrete learning, input, persistence, and diagnostic defects while retaining Policy v9 and dataset schema 15. The matched live trials **do not establish an overall learning-speed improvement or reliable mouse/scroll mastery**. Weights genuinely changed and optimizer progress continued; those facts alone are insufficient evidence of useful behavior.

## Repairs and audit coverage

- Nonterminal rollout prefixes now bootstrap from their immediate retained successor, rather than a later decision.
- Episode end keys are consumed once, including timeout acknowledgments and identical start/end keys.
- Sampling, likelihood replay, and saved capabilities share executable channels, permissions, reserved keys, and equivalent-modifier restrictions.
- Button-only and scroll-only actions respect the control region. Scroll events carry their intended location; releases remain available.
- Zero-step KL rejections no longer advance policy revisions. Attempted, applied, rejected, stale/buffer, and unread work have separate reporting. Legacy counters are labeled rather than silently treated as applied learning.
- RL uses an explicit constant learning-rate schedule. Demonstration defaults remain unchanged.
- Partial optimizer work retains its selected epoch order, targets, random state, and optimizer state. Learning-semantics version 2 excludes incompatible old pending work while retaining the active actor and compatible heads.
- Experience budgets include timing, control, context, latent, and binary arrays. Optimizer loading rejects non-finite moments/masters and negative variances before they can poison a valid brain.
- Accurate OCR retries scaled copies when native recognition drops an isolated integer zero. Confidence requirements and strict numeric parsing remain in force.
- The opt-in audit harness has explicit frozen evaluation, independent fixture scoring, tensor comparisons, isolated storage, failure propagation, and Panic/Quit cleanup.

Review covered capture/HEVC/input timestamps, preprocessing and reward masking, temporal history, policy/critic/distributions, PPO/GAE, optimizer precision and schedules, actor/learner ownership, action execution, checkpoint transactions, demonstration/RL switching, UI orchestration, updater identity checks, and build/signing. Existing regression coverage for recording and demonstration training remains enabled. No alternative RL algorithm or architecture redesign was introduced.

## Matched real-screen experiment

The original implementation is commit `5da8c0e`, with common QA instrumentation from `164e589`. Its original installed bundle and signed instrumented baseline are retained locally. Three task families and policy seeds 7/42/101 used identical initial actor **and head tensors**, the same training schedules, and disjoint held-out evaluation schedules. Safetensors file headers can differ in tensor order; equality was checked from sorted tensor names, shapes, dtypes, and bytes.

Each cell trained for approximately 90 seconds. Frozen initial/final evaluation used deterministic deployed decoding and seeded stochastic decoding in separate 10-second episodes. Both implementations used Small architecture, 96×64 grayscale, no history, 15 Action FPS, 5 Perception FPS, learning rate 0.0003, rollout 64, minibatch 16, two epochs, and exploration scale 1. These are declared experimental settings, not the Balanced production defaults.

The separate certificate-signed native fixture receives actual keyboard/mouse events. The learner receives only ScreenCaptureKit pixels and Vision OCR rewards. Observer logs never select actions, provide model inputs, or supply training rewards. Reward text is masked from current/history images. Native scoring distinguishes exclusive key occupancy, Space edges, S holds, absolute pointing, left/right clicks, drag completion, relative movement, and horizontal/vertical scroll.

The initial integer-zero fixture presentation failed OCR; both matched arms then used decimal episode length. One improved pointer evaluation hit the human-input safety stop. The three completed keyboard cells were retained, the unfinished pointer cell was technically retested, and the remaining six cells ran as `improved-rest`. Failed attempts count against the one-hour cap and are not silently selected by score.

Deterministic results below average the observed cue types: cue-time accuracy for keyboard (**higher is better**) and Euclidean position error for pointer/relative tasks (**lower is better**). Each cue type has equal weight in this table; its own exposure count and time-weighted accuracy/error remain separate in the JSON, along with stochastic outcomes. Evaluations contain few trials, so these are descriptive engineering results.

| Task | Seed | Original initial → trained | Improved initial → trained | Optimizer steps, original / improved |
| --- | ---: | ---: | ---: | ---: |
| keyboard | 7 | 0.000 → 0.471 | 0.000 → 0.216 | 83 / 71 |
| keyboard | 42 | 0.011 → 0.000 | 0.005 → 0.000 | 74 / 83 |
| keyboard | 101 | 0.010 → 0.000 | 0.020 → 0.000 | 80 / 87 |
| pointer | 7 | 0.328 → 0.379 | 0.314 → 0.392 | 17 / 18 |
| pointer | 42 | 0.321 → 0.377 | 0.321 → 0.362 | 18 / 17 |
| pointer | 101 | 0.353 → 0.486 | 0.353 → 0.435 | 17 / 18 |
| relative | 7 | 0.330 → 0.454 | 0.329 → 0.341 | 17 / 17 |
| relative | 42 | 0.332 → 0.319 | 0.323 → 0.361 | 17 / 17 |
| relative | 101 | 0.348 → 0.324 | 0.344 → 0.356 | 17 / 17 |

Both arms sometimes learned a useful behavior and sometimes regressed. Positioning error is not click/drag success; shaping reward is not task completion. Sparse edge, hold, drag, and scrolling opportunities do not support mastery claims. No global speedup or broad task-learning gain is claimed.

## Additional checks and limits

The final complete suite, with the controlled learning gate enabled, passes **239 tests with six optional dataset/performance diagnostics skipped and zero failures**, in 144.7 seconds. Tests require fresh on-policy continuation with additional optimizer steps and exact actor/head/moment/master/random-state comparisons, including interruption inside an epoch. They also cover retained tails, terminal rewards, OCR gaps, grouping-dependent returns, elapsed discounting, shared controls, reward masks, demonstration/RL switching, non-finite experience/moments, KL rejection, and low-FPS profile saving.

The controlled visual learning gate learns both keyboard cues for all three seeds. Mouse score is positional closeness, not a click or drag success rate. These in-process tasks establish PPO learnability; they do not establish browser-game competence or superiority over the original implementation.

| Controlled task | Seed | Frozen initial score | Trained score |
| --- | ---: | ---: | ---: |
| keyboard | 7 | 0.5000 | 1.0000 |
| keyboard | 42 | 0.5000 | 1.0000 |
| keyboard | 101 | 0.0000 | 1.0000 |
| mouse | 7 | 0.7001 | 0.9907 |
| mouse | 42 | 0.6990 | 0.9974 |
| mouse | 101 | 0.7018 | 0.9917 |

The supplied Dino and Mouse HTMLs were inspected. Dino lacks seeded obstacles; Mouse repeats seeds, exposes reward outside the OCR box, and lacks right-click/scroll tasks. They were not used for the matched learning claims. Native tasks were prioritized within the live budget.

Joint interval likelihood is retained. Longer OCR groups tighten effective clipping/KL behavior; dividing a joint log-ratio by group length would change the importance ratio. Discount/lambda are per nominal action and therefore need conversion when FPS changes. RMS interval reward normalization remains sensitive to OCR grouping. These remain documented design limits rather than being hidden behind an unvalidated algorithm change.

The clipped surrogate and its derivative were checked against [PPO equations 7 and 9–12](https://arxiv.org/pdf/1707.06347), and the residual/trace recurrence against [GAE equations 10–16](https://arxiv.org/pdf/1506.02438). AgentTrainer applies these to variable-duration OCR intervals with joint action likelihoods. That grouping is an implementation approximation: OCR does not reveal when each reward inside an interval occurred. It is not equivalent to observing a separate reward at every decision.

An analytical eight-unit-reward stress case (discount 0.9, lambda 1) makes the effect explicit. These numbers are explanatory calculations covered by a regression, not live learning results:

| Decisions per OCR interval | First return | First return with steady interval RMS | Joint ratio for +0.1 log-ratio per decision | Sample KL |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5.6953 | 5.6953 | 1.1052 | 0.0052 |
| 2 | 5.9951 | 2.9975 | 1.2214 | 0.0214 |
| 4 | 6.6244 | 1.6561 | 1.4918 | 0.0918 |
| 8 | 8.0000 | 1.0000 | 2.2255 | 0.4255 |

Changing reward grouping can therefore change target scale, bias, clipping, and KL admission even for the same underlying reward stream. The critic uses targets computed with the previous RMS scale; that scale can drift as interval duration or reward magnitude changes. No unvalidated normalization rewrite or ratio shortcut was promoted.

Temporal memory is a finite sampled image/control/timing window. Runtime history resets per episode; learned weights and optimizer progress persist. The default four-frame lookback is about 0.267 seconds at 30 Perception FPS.

## Fixture corrections and secondary evaluation

All three primary provenance records have the same fixture-source SHA-256, `7dee17b78291fb6e071ac6d0d93767921aceb9f0e0888188625913113b172471`. Two fixture defects found afterward limit those measurements:

- The keyboard prelude discarded key-down events before the first cue. The fixed fixture retains held state during the prelude. Exact saved initial/final keyboard artifacts were re-evaluated without training, using a shared 2.61 evaluator. The deterministic mean cue-time accuracies for original policies were 0.000→0.444, 0.000→0.250, and 0.042→0.000 for seeds 7/42/101; improved policies were 0.000→0.117, 0.000→0.250, and 0.025→0.250. These remain weak results and can reflect constant-key bias. They do not establish conditional key selection or a complete S hold. Raw primary results remain visible above.
- The drag fixture repeated pickup rewards and retained picked state after invalid releases. The corrected fixture rewards pickup once, requires a held drag, clears invalid releases, and shows the carried object. A separate scripted HID contract check verified zero invalid-drop completions, no repeated pickup reward, and one valid held-drag completion. This validates the fixture, **not learned dragging**. Original drag scores cannot prove valid drag learning; no new drag learning claim was made after this correction.

A development-only pointer run with learning rate 0.00003 and exploration scale 3 applied 60 steps and ended near KL 0.0097, but held-out positioning error worsened and click/drag learning did not improve. It was not confirmed across seeds or promoted to production defaults.

## Measured operating limits and lifecycle

Nine 20-second screens used the Small model and changed one setting at a time, with the FPS staircase explicitly raising both rates. These establish short-run operability, not learning superiority for RGB, resolution, or memory. Values are means of dashboard snapshots after the first five seconds, including episode pauses, on the same Apple M3 Max/36 GiB Mac running macOS 27.0.1.

| Requested action / perception FPS | Vision | Past frames | Measured actions / perceptions / reward reads per second | Head / vision compute ms |
| --- | --- | ---: | --- | --- |
| 15 / 5 | 96×64 gray | 0 | 13.79 / 4.40 / 14.75 | 1.03 / 1.94 |
| 5 / 5 | 96×64 gray | 0 | 4.65 / 4.09 / 5.01 | 1.55 / 2.34 |
| 60 / 5 | 96×64 gray | 0 | 55.34 / 3.73 / 25.28 | 1.59 / 2.00 |
| 60 / 15 | 96×64 gray | 0 | 55.72 / 11.71 / 25.04 | 1.55 / 1.94 |
| 60 / 60 | 96×64 gray | 0 | 55.85 / 44.21 / 25.17 | 1.15 / 2.18 |
| 15 / 5 | 96×64 RGB | 0 | 13.92 / 4.19 / 14.78 | 1.31 / 2.15 |
| 15 / 5 | 320×180 gray | 0 | 13.78 / 4.10 / 14.75 | 1.40 / 2.39 |
| 15 / 5 | 96×64 gray | 4 | 13.60 / 3.99 / 14.74 | 2.25 / 2.64 |
| 15 / 5 | 96×64 gray | 8 | 13.53 / 4.12 / 14.76 | 2.18 / 2.60 |

The screening process peaked at 410.2 MB sampled resident memory; cell peaks of MLX active allocation were 31.5–43.2 MB. Resident memory includes other allocations and caches. These short measurements do not establish long-run memory bounds. The 5-FPS cell exposed an unrelated demonstration-only future-horizon validation check blocking RL; the production validation path was repaired and all nine cells then completed. Ordinary Run also executed saved keyboard, pointer, and relative/scroll policies at about 14.7–14.9 actions/s without modifying their weight files.

The final keyboard pause/resume check retained 79 saved optimizer steps, restored pending work to 86 before the new episode, and reached 94 after another eight seconds. Three timeout episodes received exactly one end key apiece; their independently received rewards (12.75, 1.50, 16.75) matched saved episode records. Missing OCR released inputs and saved an interrupted episode rather than inventing zero rewards. Quitting during frozen evaluation propagated `Application quitting`; subsequent HID state queries found no held audit keys or mouse buttons. The signed dashboard was visually inspected.

Observation packing, repeated learner encoding, inference cache refresh, CPU snapshot transfer, and scheduling were reviewed. Encoding repeats during optimization because encoder weights train; cached inference embeddings cannot replace gradient-bearing forwards. Current microbatches bound retained graphs and the learner yields to inference. The common primary harness did not record separate packing/transfer timers or a cumulative admitted-action total, so no stage-specific speedup is claimed. Per-control received-event counts, episode exposure, numerical actor-output/encoder/critic/exploration changes, exclusions, revisions, and available memory/latency measurements are published in the JSON.

**Total live experiment time was 3,505.55 seconds (58 minutes 26 seconds)**, including canaries, failures, technical retests, frozen evaluations, screening, deployment checks, and lifecycle probes. Optional browser probes were omitted. Later checkpoint-validation, shutdown, low-FPS, and diagnostic fixes have regression/lifecycle/screening coverage; the nine paired long training cells were not repeated after every later patch.

## Reproducing the audit

The originals supplied by the user remain unchanged. The broad native fixture is an additional test environment. To reproduce primary fixture semantics, use the fixture at commit `164e589`; the current fixture contains the disclosed prelude/drag corrections. The original learning implementation with shared measurement instrumentation is also available at `164e589`. Primary improved arms identify their exact commits and binary hashes in the JSON. Use matching versions for both sides of a new comparison.

Build the signed fixture with `AGENTTRAINER_FIXTURE_AUDIT=1 zsh scripts/build-rl-fixture.sh`, and a diagnostic app with `zsh scripts/build-rl-audit.sh "$PWD/work/rl-validation/audit/local/AgentTrainer.app"`. Launch `work/rl-validation/RLTestGame.app/Contents/MacOS/RLTestGame` with `AGENTTRAINER_AUDIT_ROOT="$PWD/work/rl-validation/audit"`; keep it running. Then use `python3 scripts/run-rl-audit.py improved "$PWD/work/rl-validation/audit/local/AgentTrainer.app" --run-name local` on a dedicated desktop. The harness enforces focus/input safety and isolated workspaces. `--seeds`, `--tasks`, `--seconds`, `--evaluation-seconds`, and `--evaluation-seed` specify exposure; defaults reproduce the reference schedule. Training fixture seeds are `1000 + policySeed`, with episode seeds advanced by 7,919; held-out deterministic/stochastic seeds are 900001/900002. The manifest never exposes diagnostic fixture state to the policy.

Frozen re-evaluation uses the `evaluate` operation and `--source-workspace` to load exact initial/final saved artifacts. `smoke` runs an existing policy and verifies checkpoint continuity; `fixture` is explicitly scripted contract validation. `screening` runs the nine bounded operating cells. The launcher writes hashes, manifests, reports, process resource samples, and the cumulative live ledger. `python3 scripts/summarize-rl-audit.py` produces the sanitized published aggregate for this study. Do not remove an existing ledger to extend a study's allowance.

## Delivery verification

Version 2.61/build 39 retains Policy v9, schema 15, the pinned certificate, and the existing transactional installation/package workflow. Signing regression checks, shell syntax checks, and the full enabled regression suite pass. Installation and packaging are verified against the original designated requirement; the repository branch and local installer are updated without publishing a GitHub release.

Full manifests, checkpoints, logs, signed diagnostic apps, and the live-budget ledger remain under ignored `work/rl-validation/audit`. Sanitized tensor checks, per-control results, provenance hashes, and timing data are in [validation-2.61.json](validation-2.61.json). Raw data, screenshots, private keys, and machine-local signing pins are excluded from distribution archives.
