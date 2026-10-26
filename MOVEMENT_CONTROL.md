# AgentTrainer 2.52 movement-control investigation

**Status: diagnostic candidate, not an accepted live fix.** The 2.51 candidates were rejected by the user in Reborn: persistent W/Space/other keys and little camera movement, including on a recorded trial. The latest published release remains 2.50. Neither an offline loss nor recorded-video replay establishes improved gameplay.

## Reproduced failures

Policy v7 received its own current timing/control state without a constraint on its influence. Frozen-brain probes reproduced mouse feedback amplification. Policy v8 bounded the continuous mouse prior but still let generated keys influence camera confidence and unrestricted held-state copying influence binary outputs. The initial mouse-only replay concealed this problem by supplying demonstrated keys. Full posted-control feedback reproduced Space holds lasting tens of seconds.

A single-video experiment was essential: good predictions using recorded controls collapsed with the model's own controls, even on the same real visual sequence. A second issue appeared in fitting: small augmentation-off locality batches mostly contained neighboring frames from one action. Diverse batches of the same frames substantially improved the visual base without adding or changing targets.

## Policy v9

Recordings, cache schema 15, the 2,190-value timing vocabulary and 146 executable actions are unchanged. Timing remains available with current-only vision. GRU/LSTM, all supported precisions and categorical context remain supported.

The shared CNN encodes each real image independently. Two policy streams share recurrent/fusion weights and run together along the batch dimension:

- Complete context receives all causal controls, held/edge durations, motion samples/rates, clocks and validity.
- The grounded stream receives real visual history, categorical context and observation clocks/validity. Endogenous controls and their timing derivatives are zeroed in this stream.

For binary heads, the final logit is `base + tanh(context - base)`, computed in Float32 before casting. Timing can resolve an uncertain visual decision; its correction is bounded by one logit. The same real targets also supervise the grounded base with weight 0.5 during training. This adds no CNN/recurrent pass or learned parameters at inference.

Absolute cursor, relative movement base, scroll, and movement activity use the grounded stream. Generated keys cannot directly suppress camera confidence. Bounded mode retains the existing motion-independent convex history mixture with gain below 0.4, actual interval normalization, and sensitivity compensation. Visual timing only omits that prior. Movement history cannot enable its own stopping/activity gate. Both choices preserve complete temporal evidence in the context stream. There is no delayed smoothing queue or arbitrary sub-tick HID scheduler; at 60 Hz execution remains about 16.7 ms.

Per-axis movement confidence interpolates smoothly through uncertain logits: `u = clip((logit + 1) / 2, 0, 1)`, `gain = u² × (3 − 2u)`. Logits at or below −1 stop exactly; logits at or above 1 preserve full predicted movement. This avoids magnifying small encoder/precision differences near zero into a full-size camera command. The raw regression targets and confidence labels remain real recorded actions.

These constraints bound individual feedback paths. They are not a proof of game stability or a substitute for learning appropriate actions from images.

## Real single-recording fitting

All arms used one real 26.98-second recording (1,619 action rows), 60/60 FPS, 128×128 vision, focal gamma 0, no augmentation/auxiliary losses, and exactly 20 epochs. The dense fitting arms used Float32, batch 4, no stochastic regularization and 8,100 optimizer updates. No generated mouse trajectories became targets.

The following investigation measurements preceded the smooth-confidence correction:

| Posted-control result | Earlier v8 dense fit | v9, local batches | v9, diverse batches |
| --- | ---: | ---: | ---: |
| Movement recall | 4.6% | 73.8% | 84.4% |
| Idle rows posting movement | 5.7% | 48.9% | 6.1% |
| Raw-pixel MAE / axis / tick | 1.744 | 1.880 | 1.271 |
| Longest W hold | 26.92 s | 26.98 s | 10.55 s |
| Longest Space hold | 0 s | 9.82 s | 1.10 s |

The real recording's longest W and Space intervals are 10.55 and 1.10 seconds. The diverse v9 run had W FP/FN 7/2, Space 9/1, S 4/0, and D 0/0 action ticks. Recorded-history keyboard precision/recall were about 99.2%/99.9%. The inference-only all-control-masked ablation had mouse MAE 1.324 versus 1.271 with retained history. This ablation changes inputs, never the real targets; it does not replace the matched movement-mode comparison.

The final smooth-confidence decoder was reevaluated on the same completed weights, without additional training. It achieved **86.1% movement recall, 7.4% idle posting, and 1.265 raw-pixel MAE**. W/Space/S/D counts and holds above stayed unchanged; Shift had no extra or missed ticks and matched its 1.167-second maximum. The masked-control ablation had 84.0% motion recall and MAE 1.321. The candidate remains a single-trial diagnostic, not a general gameplay claim.

The evaluator uses actual integer events posted through the production injector, a private event sink, the prediction latch, and causal current/historical timelines. No events reach global HID. Recorded video cannot react to generated actions. Memorizing the clip demonstrates fitting and feedback behavior, not generalization or gameplay success.

## Initial live confirmation

The user tested the retained-history memorization brain in the matching Reborn trial. They reported substantially better control, good mouse movement for the small training set, and keys pressed at the right times. The brain still did not approach completing the trial; that was outside this diagnostic's goal. This is positive live evidence for the control correction, not proof of generalization or proof that additional epochs would solve the remaining errors. A matched single-recording visual-movement arm also completed 20 epochs and 8,100 updates, using the same real clip, preprocessing, seed and optimizer settings. It had 81.1% motion recall, 37.0% idle posting and MAE 1.850 under posted-control feedback, failing the diagnostic idle/key gates. It is a comparison brain, not an accepted correction. The user subsequently compared both in live Reborn with cursor movement enabled and preferred the retained-history brain: visual-only was more random and worse at both keys and mouse movement. Bounded movement history remains the selected approach. This preference is established for the tested trial, not general gameplay. The comparator is preserved in a recovery archive.

## Broader comparison

The new frozen selection contains 46 real recordings: the original 39, an additional 3.59-second Reborn recording saved before the check, and the six camera-check recordings. Total duration is 1,066.245 seconds, with 63,999 action rows and 9,605 whole-recording held-out rows. Both movement modes use the same architecture, initialization seed, sampler, optimizer, split, 60/60 FPS, 128×128 vision, focal gamma 0 and exactly 20 epochs / 34,000 updates. Existing profiles and recordings are preserved. Completed arms evaluate only on reruns.

Before the smooth-confidence correction, the visual-movement arm failed the useful-movement recall gate. Its teacher-loss-selected epoch-1 brain had 50.9% motion recall, 40.7% idle posting and a 7.68-second maximum Space hold. The final checkpoint released Space much better (maximum 1.02 seconds) but still had only 65.3% motion recall and 52.3% idle posting. These are failed acceptance results, not an improvement claim. The retained-history arm also failed: its selected brain had 56.1% motion recall, 43.6% idle posting and an 11.65-second maximum Space hold; the final checkpoint had 67.0% recall, 54.2% idle posting and a 1.30-second maximum Space hold. Both final checkpoints reduce the prolonged Space failure, but neither meets the broader movement gate. No full-data live winner has been chosen. The next controlled live check uses the successful single-recording fitting brain.

The previous v8 comparison used 39 recordings and mouse-only feedback. Its improved replay errors did not transfer to live Reborn; the user rejected both candidates on an unseen and a recorded trial. Do not reuse that comparison as live acceptance evidence.

## Training and compatibility repairs

AdamW now retains Float32 master weights, preserving updates smaller than a BF16 weight's rounding step. All master weights, moments, scheduler and RNG state survive exact same-contract resume. Fresh warm starts reseed masters after loading selected weights. Legacy optimizer containers explicitly preserve their original arithmetic; model-generation migration still requires retraining.

Disabled augmentation bypasses transforms entirely, fixing shape errors between independently deduplicated images and per-action rows. Small batches use the established diverse salience ordering; sampling remains separate from deterministic CNN reuse. Global gradient norms accumulate in Float32 to avoid FP16 overflow.

Policy v9 validates weight-format metadata, including same-shaped v8 weights. Older brains/checkpoints are archived and compatible schema-15 caches retained. The signing identity and recording bytes are unchanged. Installation, final test/benchmark results, and live acceptance must be verified before publication.

## Candidate validation

Compact, default and 16-frame stress hardware benchmarks passed the unchanged equivalence, 1% quality-drift, overhead, memory and sustained-throughput gates. Canonical, reused and augmented training references all include the visual-base loss. Cached policy-head p95 was 0.85 / 0.90 / 2.42 ms on this Mac; these figures exclude capture, CNN encoding, queueing and HID delivery. Full-objective throughput retention was 100.1% for default and 104.2% for stress, with nominal thermal state and bounded cache memory. The functional suite passed 199 tests with six opt-in workflows skipped; the real single-recording workflow and hardware benchmarks were run separately. Signed installation/package verification is recorded with candidate validation; actual Reborn acceptance remains pending.
