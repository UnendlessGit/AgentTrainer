# Mouse / game-camera regression investigation

This document preserves the original investigation and measurements. Current Policy v9 work and live acceptance status are documented in [MOVEMENT_CONTROL.md](MOVEMENT_CONTROL.md).

Investigation date: 2026-10-25. Application/source: AgentTrainer 2.50, commit `2c430af`.

## Finding

A saved Policy-v7 Reborn brain exhibits substantial amplification when its own predicted mouse movement replaces human movement in the new **current decision timing** input. This reproduces a mechanism consistent with sustained camera drift/spinning. The deleted Blade Ball brain and its exact runtime session cannot be recovered from the available evidence, so this is a reproduced regression mechanism rather than proof that every reported bad movement had the same cause.

Policy v7 added the most recent movement displacement/rate/event evidence directly to fusion on every action decision (`PolicyNetwork.swift`, current `timingEmbedding` concatenation). The runtime fills it from the previously posted output (`AgentRuntime.inferAction`, `InputInjector.temporalSnapshot`). During training and ordinary validation this evidence instead comes from the human demonstration. The model can learn to continue a recent turn, then amplify its own small prediction errors when that input is its own output. The existing half-sample control mask shows the model either expert controls or masked controls; it does not exercise recovery from erroneous nonzero movement histories.

The previous version also had historical control feedback. The regression is the addition of a direct, most-recent movement path alongside that history, plus fresh head decisions between visual updates. It is not evidence that explicit hold durations or observation clocks should all be removed.

## Evidence and method

Used the existing Reborn recording folder: 39 recordings, approximately 965.1 seconds, 115,198 nonzero mouse movement samples. The saved brain is `A9826F1B-2A82-4FF8-B331-C557F7DBF9F1` (epoch 29, 44,235 steps), 128×128 color vision, 60 perception / 60 action FPS, BFloat16, four past frames. Its checkpoint lists the same 39 recording IDs as the current Reborn folder. No training, HID posting, recording edits, model edits, or app replacement was needed.

The source recordings and saved weights are read-only. Disposable caches and aggregate JSON are isolated under ignored `work/mouse-investigation-v7`. The diagnostic uses a seeded recording-level sampling split on its own UUID-ordered cache; it does not claim to reconstruct the original training run's exact held-out split. These are paired input-sensitivity comparisons on a frozen brain, not new generalization or gameplay scores.

The continuous-sequence comparison preserves identical recorded current/past images, historical controls/timing, and current non-mouse features. Only the new current mouse evidence is changed: one arm uses the recorded human evidence, the other reduces the model's own previous rounded raw output through `CausalControlTimeline`. Posted deltas use the production scale/rounding at sensitivity 1, with a simulated 1 ms posting offset. MAE converts the existing normalized/clipped targets back with scale 80 and is measured before integer posting; it does not recover human displacement beyond the existing target clip. Keeping historical controls fixed deliberately isolates this one feedback path; the test does not model the full live controller or a game responding to counterfactual actions.

| 600 consecutive action rows (10 s at 60 Hz) | Recorded current movement | Model's own current movement feedback |
| --- | ---: | ---: |
| Mean mouse error, raw pixels per axis per tick | 0.886 | 4.627 |
| Mean predicted movement, sum of absolute X/Y per tick | 1.828 | 9.148 |
| Mean signed Y output, raw pixels per tick | −0.868 | −5.975 |
| Idle rows producing a rounded nonzero mouse event | 353 / 396 | 396 / 396 |

Error increases **5.22×** when closing this one feedback path, despite retaining the same changing visual sequence. This is consistent with persistent turning rather than a data-format or unit conversion failure. It is not a live-game measurement: recorded video cannot react to the simulated actions.

In a separate fixed-context stress test, 32 contexts use a 1/60 s action clock and feed predicted, rounded movement back for 60 decisions. Mean absolute X/Y movement increases from **2.125** raw pixels at the first decision to **9.219** by decision 15 and remains there. This isolates feedback sensitivity; it does not prove a stationary game scene should always have zero intended movement.

On a 540-row diagnostic sample, the following counterfactuals have much smaller effects than closing the feedback loop:

| Input variant | Mean mouse error, raw pixels per axis |
| --- | ---: |
| Recorded timing | 1.277 |
| Source-frame age forced to 16 ms | 1.276 |
| Source-frame age forced to 125 ms | 1.287 |
| Recorded movement coarsened to one rounded posted event | 1.273 |
| Cursor input forced to capture center | 1.276 |
| Current movement block zeroed | 1.929 |
| Entire current timing input zeroed | 3.574 |

Zeroing all timing in an already-trained brain is therefore not a demonstrated correction. The mouse normalization remains 80 raw pixels per normalized unit in both versions, and the saved current camera sensitivity is 1. The saved brain recommends Game Camera. Current settings cannot establish what settings the deleted Blade Ball session used.

## Why the previous validation did not catch this

1. The earlier Blade Ball experiment compared two Policy-v7 brains. Both retained current held state and mouse displacement, so both retained the new direct feedback path. It was not a v6-versus-v7 live-controller comparison.
2. Ordinary offline validation supplies demonstrated human history for every row. A lower one-step MAE does not establish that repeated predictions remain stable when the history comes from the model itself.
3. `ValidationAccumulator.consumeContinuous` counts an idle false action only above normalized `0.05`, which is **4 raw pixels** at scale 80. Runtime rounding can execute movement at **0.5 pixels**, normalized `0.00625`. Small repeated drift can therefore be absent from the reported idle-false-action metric.
4. Saved-brain quality gates primarily use aggregate immediate loss and binary regressions. There is no dedicated closed-feedback camera-drift release gate. Faster cached-head latency also does not establish good closed-loop control.

## Correction requirements

A production correction should preserve useful key/button duration and observation clocks while addressing the direct recent-movement shortcut into the mouse heads. Candidate changes need matched training and feedback evaluation, rather than a runtime sensitivity reduction or blindly zeroing timing in existing weights. Training should expose the controller to imperfect movement histories and require recovery, not only perfect human history or all-zero masked history.

Validation should separately measure active movement accuracy, actual post-rounding idle movement, signed/cumulative drift, reversal/stopping response, repeated decisions on held/delayed observations, and independent action/perception rates. Use the runtime movement threshold for the stated default sensitivity, and report that non-default sensitivity changes execution thresholds. The existing 20-epoch budget should remain the cap if a new training experiment is run.

Reborn already has enough mouse-rich data to develop and test that correction. A short additional controlled recording could later serve as an independent live stopping/reversal check; collecting more demonstrations is not the prerequisite for resolving the demonstrated input/feedback problem.

## Reproduction and validation

The original read-only diagnostic is opt-in: set `AGENTTRAINER_MOUSE_PROBE=1` and run the release-mode `MouseInvestigationTests` test with the MLX Metal library installed as in `test.sh`. It reads the frozen v7 reference under `work/mouse-comparison-v8/Reference-v7` when present, otherwise the original saved Reborn brain. It posts no input, performs no training, and saves aggregate metrics to `work/mouse-investigation-v7/sensitivity.json`.

At the time of this investigation the installed app and model artifacts were unchanged. The subsequent correction, two-arm real-recording training workflow (`./benchmark-mouse.sh`), migration, and validation are documented separately in [MOVEMENT_CONTROL.md](MOVEMENT_CONTROL.md). The benchmark script now runs that correction comparison; it is not the original read-only probe.
