# AgentTrainer 2.51 movement control

Policy v8 corrects a reproduced Policy-v7 mouse feedback regression. The earlier diagnosis and frozen-brain measurements are in [MOUSE_CONTROL_INVESTIGATION.md](MOUSE_CONTROL_INVESTIGATION.md). Timing inputs remain explicit; removing all timing from an existing v7 brain is not a supported correction.

## Inputs and architecture

The recordings, dataset schema 15, and 2,190-value `TemporalInputContract` are unchanged. A decision still receives real images, paired trailing controls, hold/edge durations, movement/scroll samples and rates, elapsed intervals, frame age, and validity. Runtime feedback comes from events actually posted after output restrictions and integer rounding.

Policy v8 reuses the same CNN embeddings in two policy streams. They share recurrent and fusion weights; they are evaluated together along the batch dimension, rather than duplicating the encoder or parameter set.

- The decision stream receives complete control/timing history and produces keyboard, modifier, button, and predictive-state outputs.
- The motor base stream retains visual history, categorical context, discrete controls and their timing, observation clocks, and validity. It zeros cursor position plus movement/scroll control and timing fields. Absolute cursor, scroll, relative base movement, and relative movement activity come from this stream.
- Bounded mode reintroduces previous relative movement through a separate constrained path. Visual timing only mode omits this path. Both choices have identical parameter shapes and are immutable in the saved brain.

For each axis and history window, let `d` be the normalized signed displacement recovered from the Float32 rate encoding, `Δt` its actual interval, `h` the nominal action interval, and `s` runtime camera sensitivity. The retained value is

```
v = clip(d × min(1, h / Δt) / s, −1, 1)
prior = 0.4 × sigmoid(gain(motor)) × Σ softmax(attention(motor)) × v
relative = clip(tanh(base(motor)) + prior, −0.9999, 0.9999)
```

Invalid windows contribute zero. The current window and every selected historical observation have independent validity. Long windows become a rate-equivalent action step; short windows are never magnified. Sensitivity is removed before reuse, so increasing the external sensitivity does not increase the direct history loop gain. All decoding and mixture arithmetic uses Float32 before casting to model precision.

Attention and gain cannot read the movement being reused. Consequently, the direct movement-history path has continuous gain below 0.4 for fixed visual/discrete/clock context. Its convex mixture cannot amplify an input perturbation. This is a bound on one feedback path, not a stability proof for an interactive game, generated keyboard/button feedback, or arbitrary visual dynamics. With a zero base at 1× sensitivity, the maximum gain also avoids retaining a one-pixel command solely through rounding. There is no runtime smoothing queue to delay a reversal.

Two extra internal logits classify whether X and Y should move. A nonpositive activity logit makes that axis exactly zero. The activity decision is independent of continuous feedback, so a previous turn cannot enable its own next turn. Activity labels come directly from real forward-interval targets at the default posting threshold (`abs(delta) >= 0.5` raw pixels). Per-axis positive weights use the training split only, capped at 8. The activity BCE contributes 0.25 within the movement loss; displacement regression remains active. An early unweighted prototype suppressed useful movement and was rejected.

The physical action vocabulary remains 146 values. Internal immediate logits have 148 values, as do each of the four future-action horizons. Saliency selectors, sizing estimates, precision conversion, strict weight loading, checkpoint signatures, and context fusion follow this layout. GRU/LSTM and current-only vision remain supported. All control-derived timing and the bounded path share the existing per-sample control-dropout mask; clocks and validity remain visible.

## Validation and real-data comparison

Learning uses only the user's real Reborn demonstrations: 39 recordings, about 965 seconds, 115,198 nonzero movement samples. No synthetic mouse trajectories or generated-action labels are used. Small algebra/event fixtures test contracts only and are never training examples. Generic hardware benchmarks test tensor-path equivalence rather than mouse learning.

`./benchmark-mouse.sh` uses an isolated ignored workspace, freezes the v7 reference before migration, and trains bounded-history and visual-only candidates with identical architecture, initialization seed, recording order, whole-recording split, sampler, and optimizer. Both use all 39 recordings, 60 Perception FPS, 60 Action FPS, 128×128 color vision, BFloat16, binary focal gamma 0, four past frames, and exactly 20 epochs / 30,820 updates per arm. Automatic repetition is absent and the step cap is zero. Completed arms are evaluation-only on reruns.

The feedback evaluator uses real recorded video through the native encoder, cached visual context, `RuntimePredictionLatch`, and the production `InputInjector` with a private `CGEvent` sink. It closes both current and historical mouse feedback using the integer deltas actually posted; other controls retain demonstrated context. It separately exercises perception every one, two, or four action ticks, with fresh action clocks and source-frame age between observations. No events reach the global HID stream.

Scores include raw-pixel error after posting, useful-movement recall, idle movement, signed/cumulative drift, direction, stopping/reversal lag with missed-event counts, and action-head latency. The frozen v7 brain had different original training/split provenance; comparisons against it demonstrate reproduced feedback behavior, not matched training superiority. Comparisons between the two v8 arms are matched.

**Recorded video cannot react to generated actions.** This experiment measures feedback robustness against real visual sequences and cannot establish actual gameplay improvement. A real Reborn A/B check is required before selecting the live winner. Recording-feedback measurements appear below; the live comparison remains a separate acceptance step.

## Measured recording-feedback results

All **8,639 held-out action rows** are evaluated as complete recording sequences. Training uses 57,928 total rows, recording-level held-out splits, seed 42, and the same initial tensor values for both v8 modes. Each arm completed 20 epochs; the app's existing held-out loss selector chose its epoch-3 brain for Run. The frozen v7 reference is an earlier epoch-29 brain, not a matched 20-epoch training arm.

At 60 perception / 60 action FPS:

| Metric | Frozen v7 reference | Visual timing only | Bounded history |
| --- | ---: | ---: | ---: |
| Raw-pixel MAE / axis / tick | 5.255 | 2.043 | 2.182 |
| Movement recall | 100.0% | 79.5% | 76.4% |
| Idle rows posting movement | 100.0% | 56.3% | 53.0% |
| Active-axis direction accuracy | 57.2% | 43.9% | 42.4% |
| Matched stops | 112 | 609 | 618 |
| Missed stops before target resumes | 1198 | 706 | 697 |
| Matched-stop p95 lag, ticks | 36 | 8 | 9 |

Both corrections reduce error and runaway movement, while useful movement recall remains above 76%. Neither is perfect: roughly half of demonstrated idle rows still post movement, and direction accuracy is lower than the reference. Stop-lag percentiles describe matched events only; missed stops remain failures and are reported alongside them. Only seven immediate sign reversals occur in these held-out sequences, too few for a strong reversal claim.

At 30/60 FPS, visual-only and bounded MAE are **2.028 / 2.174** pixels; at 15/60 FPS they are **2.049 / 2.207**. The reference worsens to **5.397 / 5.832**. Cached policy-head p95 measurements range **1.58–4.23 ms** across corrected arms/cadences on this Mac, excluding capture, CNN, queueing, and HID delivery. These are measurements, not an end-to-end 60-FPS guarantee.

Bounded history has fewer idle false moves; visual-only has lower displacement error and greater useful-movement recall. These replay results do not settle the user's live Reborn preference. **Live result: both candidates were rejected.** On an unseen parkour trial with the same camera mode, the user observed persistent W/other-key/Space holds and little camera movement; the visual-only candidate moved slowly in short bursts. A follow-up on a recorded trial was somewhat better but still poor, with very similar behavior from both candidates. Neither meets the requested live-control quality. The replay improvement is insufficient for acceptance, and this candidate release has not been published.

The initial replay test closed only mouse feedback while supplying demonstrated discrete controls. It therefore did not test the full live key/button feedback distribution. An expanded diagnostic closes every actual posted control and compares saved best/final checkpoints. Activity bypass and control masking in that diagnostic are inference ablations only, never synthetic training labels. Further development uses additional real recordings and another live acceptance check.

## Migration and execution limits

Model schema 8 / `AgentTrainer.Policy.v8` requires retraining. Incompatible v7 brains and checkpoints are archived, never converted or silently partially loaded. Recordings, profile architecture, temporal settings, and compatible schema-15 caches remain intact. Exact pause/resume is supported only within the matching v8 checkpoint contract.

Only keys/buttons with actual positive training targets enter immutable output capabilities. This uses positive counts, not positive-loss weights (a never-pressed button can have weight one). Validation reports per-axis activity recall and idle false moves at the same half-pixel rounding boundary used by execution at default 1× sensitivity. Nondefault sensitivity changes the posting threshold.

Run settings are captured at startup; live output permissions still act immediately. Each movement is consumed once, action/perception queues remain bounded and serialized, and panic, takeover, context invalidation, stale-observation stopping, and release ordering retain their existing contracts. At 60 Action FPS the execution interval remains about 16.7 ms; no arbitrary sub-tick HID scheduling is added.

## Release validation

The functional suite passed **188 tests with four opt-in workflows skipped**; the real-recording comparison and release hardware benchmark were run separately. Compact, default, and 16-frame Float32 stress benchmarks passed the existing equivalence, quality, auxiliary-overhead, sustained-throughput, memory, and thermal gates without relaxed thresholds. Cached action-head p95 was **2.07 / 2.91 / 4.88 ms**. The default full-objective sustained throughput retained 102.9% of its first window; stress retained 100.3%. These hardware figures are local measurements and exclude capture, image encoding, scheduling, and HID delivery.

Migration tests verify archived v7 weights remain recoverable and compatible schema-15 pixel caches survive. Signing regression checks and shell syntax checks pass. Final installation/package identity and live Reborn acceptance remain separate release checks.
