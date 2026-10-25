# Policy v7 temporal input and validation record

This documents the historical 2.50 / Policy-v7 release. Current mouse feedback corrections and Policy-v8 contracts are in [MOVEMENT_CONTROL.md](MOVEMENT_CONTROL.md).

AgentTrainer 2.50 (build 35) uses model schema 7, `AgentTrainer.Policy.v7`, dataset schema 15, and `TemporalInputContract.version = 1`. This document records the source audit, contracts, and local offline evaluation. It contains aggregate results, never recording contents or private model artifacts.

## Source audit: recording to execution

| Stage | Existing timing | Previous loss | v7 contract |
| --- | --- | --- | --- |
| `Capture/CaptureService.swift`, `Capture/InputCaptureService.swift`, `Storage/InputEventFile.swift` | Shared monotonic host origin, nanosecond event timestamps, individual down/up edges, raw movement/scroll samples, seeded held controls, balanced shutdown releases | No recording format gap identified | Original video/event bytes retained; startup-seeded onset is explicitly left-censored |
| `Capture/HEVCWriter.swift`, AVFoundation decode | Presentation timestamps, including static frames and irregular gaps | Cache used nominal perception ticks; first decoded trim frame could be rebased incorrectly | Decode from movie origin, select only source PTS ≤ query, retain both query time and actual source PTS |
| `Training/DatasetCache.swift` | Event timestamps already align forward action targets and video observations | Sampled state/tap OR discarded edge ages/duty; pre-trim history was insufficiently explicit; zero-duration endpoint labels | Separate causal control reducers for action/perception queries, pre-trim priming, positive-duration forward targets, mapped timing/clock files |
| `Training/TrainingEngine.swift` | Configured rates, sequence ordering, whole-recording splits | No elapsed-time model tensors; shared visual rows could share head computation | Row-specific current and historical timing survives deduplicated CNN gathering, masking, validation, and auxiliary objectives |
| `Training/PolicyNetwork.swift` | Ordered past visual embeddings and 146 sampled controls | Sequence position alone cannot distinguish real elapsed duration or source-frame age | Shared 64-value timing projection feeds each recurrent step and current fusion, with Float32 duration math |
| `Agent/AgentRuntime.swift` | Capture host time, action timer, safety timeouts | Policy inferred at perception rate; action timer repeated latched controls; history used predictions | CNN at Perception FPS, independent serialized head at Action FPS, cached causal visual embeddings, fresh executed-control timing |
| `Agent/InputInjector.swift` | Final restrictions, clipping, rounding, physical posted events | Predictions did not prove an event executed | Two shared reducers consume tagged posted events under injection lock, including forced releases; action/perception windows reset independently |
| `Replay/`, storage, updater and UI | Recording replay already schedules timestamped events; transactional artifacts and stable signing | No model-time change required in replay or recording storage | Preserve replay and recording formats; archive incompatible learned artifacts, update sizing/diagnostics and signed release version |

The audit covered application orchestration, capture/recording, binary input, preprocessing, dataset construction, loss/optimizer/checkpoints, inference/injection, replay, workspace/migration, updater, and UI/diagnostics, together with the README, development guide, build scripts, and prior release documentation. Recording schema remains 4 and binary event format remains `ATREVT01` version 1 (12-byte header, 72-byte records).

The 145 Blade Ball recordings contain 3,660.702 recorded seconds and 393,808 input events: 383,940 movement samples, 9,681 key events, 30 flag events, and 157 scroll samples. Ordering and balanced-control checks passed. There are no mouse-button events, so dedicated deterministic button fixtures are required. Presentation gaps reach 75 ms; a nominal 60 Hz grid alone does not express these gaps. Trims reduce the experiment to 211,261 supervised rows.

## Versioned vocabulary

One timing row has 2,190 Float32 values (8,760 bytes). It is present for each current decision and each selected historical observation. All time calculations subtract UInt64 monotonic nanoseconds before conversion. A duration is encoded as three values: `min(seconds / 0.25, 1)`, `log1p(seconds) / log(61)`, and validity. The logarithmic component remains informative beyond 60 seconds. A valid zero duration differs from missing history.

| Range | Meaning |
| --- | --- |
| 0 | Real timing-row validity; padding is zero |
| 1–3 | Elapsed time since this reducer's preceding query |
| 4–6 | Actual interval between perception queries |
| 7–9 | Source image age at the row's observation/decision query |
| 10–12 | Historical query age relative to the current action decision (valid zero for current) |
| 13–15 | Separation from preceding selected historical query; unavailable for the first selected frame |
| 16–17 | Normalized cursor coordinates |
| 18–2161 | 134 binary-control slots × 16 values, in `ActionLayout.binary` order |
| 2162–2175 | Mouse movement evidence |
| 2176–2189 | Scroll evidence |

Each binary-control block contains: actual held state; hold age triplet; latest press age triplet; latest release age triplet; most recent completed hold duration triplet; `log1p` interval press/release counts; and held-time fraction integrated at event boundaries. Dedicated Shift/Control/Option/Command slots represent modifiers; reserved duplicate key slots are never used to bypass modifier restrictions. A seeded press at recording origin does not establish an onset or completed duration. Injection starts from a known released state.

Each motion block contains normalized displacement X/Y, signed `log1p` real-time rates, `log1p` nonzero-sample count, first/last sample offsets in seconds, observed span triplet, latest-event age triplet, and interval-validity flag. One nonzero sample has a valid zero span. No samples have an invalid span while latest-event recency may remain known. Observed samples do not reveal the duration of physical motion between samples. Movement is normalized with the existing raw-delta scale 80, scrolling with scale 20; extreme finite aggregates are bounded before rate encoding.

Historical row clocks retain their own source-frame age; historical query age is recomputed for every current decision. Together these describe the full relationship to the source image even if several action rows use one observation. Timing remains available with Past frames set to zero.

## Causality and performance

At query `t`, only events strictly before `t` may enter inputs. The target covers `[t, min(t + action interval, trim end))`. An edge exactly at `t` belongs to the target and to later history. Initial priming preserves pre-trim hold onsets; pre-supervision actions do not become labels. Cache segments cannot share observations, clocks, future labels, or prediction horizons across recording boundaries. Metadata validates ownership, monotonicity, finite rows, source/query ordering, and interval arithmetic.

The shared bias-free timing projection is 2,190 → 64 followed by LayerNorm and SiLU. Projection accumulation stays Float32 for Float16/BFloat16 visual models. Current timing joins visual/recurrent fusion; historical timing joins each visual/control recurrent input. GRU and LSTM, categorical context, all 146 immediate outputs, augmentation, and predictive objectives remain available. One sample-level keep mask removes both controls and their timing derivatives, including cursor, while retaining clock/validity columns 0–15. Disabled/restricted controls cannot leak through timing.

Only deterministic CNN embeddings may be reused across action rows. Gather to supervised row shape before timing-conditioned recurrence/fusion/head computation. Runtime stores evaluated embeddings in the bounded perception ring rather than re-encoding past images for each action. Serialized inference admits one pending head and coalesces newest images, yielding between encodes. Context/permission revisions reject obsolete work. Action-timer staleness checks remain independent of pending GPU work. Stop drains physical actions/releases, then capture/inference, then late action callbacks before restart. Transient movement/scroll is consumed once. At 60 Action FPS output resolution remains approximately 16.7 ms, without arbitrary sub-tick HID scheduling.

Native model publication derives immutable binary capabilities from positive classes in the supervised training split. A loss-masked/untrained key or button cannot execute merely because its arbitrary logit exceeds 0.5. Live permissions and restrictions still apply after those immutable capabilities.

## Reproducible local experiment

`./benchmark-temporal.sh` uses all 145 Blade Ball recordings read-only and writes only the isolated ignored `work/temporal-evaluation-v7` workspace. It asserts the recording count. Both arms use the same fresh Policy-v7 architecture/initialization, seed 42, whole-recording split, sampler, optimizer, batch size 32, default four past frames (spacing 2, downscale 2), BFloat16, and default augmentation/predictive objectives. Perception and Action FPS are both 60; vision is 128×128; binary focal gamma is 0. Automatic repetition and the step cap are disabled. Each arm performs exactly 20 epochs (112,220 optimizer updates), with 179,532 training and 31,729 held-out rows. A completed arm is evaluated without restarting training.

The ablated arm retains the same parameter layout, observation clocks, held states, cursor, and displacement, while zeroing control timing derivatives and motion sample/rate/span/recency features. This isolates explicit action timing rather than confounding model capacity. Standard lowest-held-out-loss publication selected epoch 2, step 11,222 in each arm after the full 20-epoch run; the final optimizer checkpoints retain epoch 20. Reports evaluate these selected brains. The separate identical-image controlled task varies only elapsed hold time and requires a duration-dependent release decision.

Binary and edge metrics consider supervised positive classes only, matching runtime capabilities and validation loss masks. Edge matching is per control and recording with a 100 ms tolerance; edge error averages matched edges. Hold error includes holds with both boundaries matched. These subset metrics must be read with precision/recall and matched counts; they do not establish arbitrary-resolution event recovery. Movement/scroll MAE uses normalized action coordinates. All results are offline; no gameplay improvement is claimed.

Measured on one 36 GB Apple-silicon Mac (Mac15,11), macOS 15+ deployment target, MLX Swift 0.31.3:

| Held-out metric | Timing ablated | Timing enabled |
| --- | ---: | ---: |
| Immediate loss | 0.03149519 | 0.02985121 |
| Binary precision / recall | 70.43% / 97.43% | 68.56% / 97.61% |
| Press precision / recall | 59.03% / 89.04% | 52.93% / 87.11% |
| Release precision / recall | 63.19% / 97.44% | 58.13% / 96.71% |
| Matched press error | 25.294 ms | 27.151 ms |
| Matched release error | 9.215 ms | 11.484 ms |
| Matched hold-duration error | 24.432 ms (721 holds) | 26.522 ms (704 holds) |
| Relative mouse MAE | 0.01238038 | 0.01067779 |
| Scroll MAE | 0.00270057 | 0.00303965 |
| Median reported training rows/s, epochs 2–20 | 3,662 | 3,698 |
| Evaluation rows/s, one sequential run | 6,310 | 12,186 |
| Reported peak MLX allocation during evaluation | 66.53 MB | 66.70 MB |

Immediate loss improves **5.22%**, passing the ≤1% regression gate; relative mouse MAE improves **13.75%**. The controlled same-image release task improves BCE from **0.6932144 to 0.1651511**, with **95.3125%** timing-enabled accuracy, passing the controlled learning gate. Real-data binary precision, press/release matching, hold error, and scroll error do not all improve. These are retained in the report rather than hidden by the aggregate loss. The experiment does not demonstrate better gameplay, finer-than-action-tick output scheduling, or improved button behavior on this button-free dataset.

Evaluation throughput is order/cold-cache sensitive and is not a matched speedup claim. MLX counters exclude mapped dataset pages and external framework allocations, so they are not process RSS or total device memory. Training rows/s are rate-limited telemetry medians, not an isolated microbenchmark.

The release `./benchmark.sh` run uses nonzero row-specific timing and compares equivalent original/reused Policy-v7 graphs. Compact/default/16-frame Float32 stress profiles pass existing loss/gradient, 1% quality, auxiliary overhead, sustained retention, memory, and thermal gates. Warm cached single-action-head p50/p95 latencies over 64 decisions are respectively **0.575/0.655 ms**, **0.615/0.721 ms**, and **0.800/1.016 ms**. These exclude capture, CNN, queueing, HID posting, and OS scheduling, so they are not full response latency. Full-objective sustained throughput is about 444 rows/s for the default vision workload and 1,772 rows/s for the temporal stress workload; end/start retention is 1.009 and 1.033, with nominal thermal state. Peak reported stress MLX active/cache allocations are 458.53 MB / 2.224 GB. Benchmark profiles differ from the Blade Ball training profile and cannot be compared as equal workloads.

## Regression coverage and migration

Deterministic fixtures cover taps, repeated taps, continuous holds, startup censoring, modifier combinations, mouse buttons, movement/scroll spans and rates, interval boundaries, irregular/stale source images, independent decision/perception windows, actual posted-event feedback and forced releases, missing history, trims/pre-supervision context, recording boundaries, and future-event non-leakage. Shared-visual timing rows remain independent. GRU/LSTM and Float16/BFloat16/Float32 cached/full forwards agree. Existing gradient, auxiliary-objective, exact optimizer/random-state resume, context conditioning, permission/release ordering, transient-once, input-monitor stop/restart, corruption, and storage tests remain part of the full suite. Runtime context/session revision paths were also reviewed in the source audit; these checks do not substitute for live gameplay testing. The initial baseline passed 159 tests with one opt-in benchmark skipped; the final suite passes 181 tests with two opt-in tests skipped. Both opt-in workflows were run separately.

Model schema migration archives incompatible versions and checkpoints without converting their weights. Source recordings are retained; schema-15 caches are rebuildable. v6 architecture/temporal profile choices are preserved. Within v7, exact checkpoint resume still requires compatible data/layout/sampling/optimizer/random state; dropout-only edits keep weights but start a fresh stochastic sequence. User profiles and experiment artifacts remain separate.
