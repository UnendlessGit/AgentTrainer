# AgentTrainer 2.52 (build 37) — diagnostic candidate

This candidate is not published. Live Reborn acceptance is required before calling the camera regression fixed or choosing a movement-history winner. The prior 2.51 candidates were rejected in actual play.

## Changes

- Policy v9 grounds camera direction, movement confidence, and scrolling in visual history and observation clocks. Complete action/timing history informs a bounded correction to visually grounded key/button/modifier decisions. Bounded previous-movement reuse remains available; the comparison mode omits that movement prior.
- Smooth uncertain per-axis movement confidence instead of flipping a whole camera command on/off at a single logit boundary; confident stops remain exact zero.
- The same real action labels independently supervise the visual base during training. No synthetic mouse trajectories or generated-action targets are used.
- Small augmentation-off batches use diverse frame ordering, preserving independent deterministic CNN reuse. A real single-video diagnostic exposed severe underfitting with adjacent four-row batches.
- Float32 AdamW master weights preserve updates that previously disappeared when rounded to BF16 every step. Checkpoints retain those residuals, moments, scheduler and RNG. FP16 gradient norms accumulate in Float32. Fresh warm starts seed the accumulator from selected weights.
- Fix disabled augmentation applying per-action identity transforms to independently deduplicated image batches, which could crash before the first update.
- Reject incompatible weight-format metadata even when tensor shapes match. Model schema 9 requires retraining; older brains/checkpoints remain archived, recordings and compatible schema-15 caches remain unchanged.
- Training diagnostics expose visual-action loss separately from the immediate action loss and total objective.

## Evidence and limits

On one real 26.98-second recording, a 20-epoch deterministic fitting diagnostic reproduced key actions under full executed-control feedback: longest W and Space holds matched the recording (10.55 s and 1.10 s), with few extra/missed key ticks. The final smooth-confidence decoder had movement recall 86.1%, idle posting 7.4%, and raw-pixel MAE 1.265 per axis/tick. The earlier v8 memorization run moved on only 4.6% of moving rows and held W through almost the entire clip.

This uses recorded video and a private sink on the production injector. The game cannot respond to generated actions, so it is not a gameplay result. Broader real-recording comparison and live acceptance remain separate. See [MOVEMENT_CONTROL.md](MOVEMENT_CONTROL.md).

The full functional suite passed 199 tests (six opt-in workflows skipped). The real single-recording workflow passed separately. Compact/default/stress hardware gates passed without relaxed thresholds, with cached policy-head p95 0.85 / 0.90 / 2.42 ms. These exclude capture/CNN/HID latency and are not gameplay results.
