# Model and training

Back to [architecture overview](../ARCHITECTURE.md).

Vision preprocessing, dataset formats, explicit timing, Policy v9, and saved training state in AgentTrainer 2.61. The live execution path is described in [Runtime](runtime.md).

## Vision and dataset construction

`PreprocessingSpec` describes image dimensions, color mode, chroma layout, bit detail, and resize policy. Packed observations use UInt8 luma for grayscale or luma plus Cb/Cr in 4:2:0, 4:2:2, or 4:4:4 layouts. MLX expands packed pixels into dense image channels. The CNN adds generated X/Y coordinate planes, giving the encoder explicit spatial location information.

`TemporalVisionConfiguration` defaults to four past frames at spacing two, downscaled by two in each dimension. Its current validator accepts 0–32 frames, spacing 1–240 perception intervals, and downscale 1–8, with an estimated retained-runtime-context limit of 512 MiB. The estimate includes packed history, controls, timing, and cached embeddings across the retained ring. Current-only vision omits visual recurrence but retains current decision timing.

Past images copy the current color/chroma/bit/resize settings with reduced dimensions, subject to integer rounding. They are ordinary images at their reduced resolution; the implementation does not generate optical-flow or difference channels. Reduction lowers the memory and CNN cost of historical context. Displayed lookback seconds are nominal values derived from Perception FPS; delayed live perceptions can produce longer actual separations.

Dataset construction decodes video from movie time zero using native YUV output. It retains the last frame with source PTS at or before each observation query, including at trims. A static frame can supply several perception ticks, keeping observation cadence independent of new-frame delivery. Metal creates both configured resolutions in one submission; a bounded ordered pipeline overlaps decode, packing, and writes with up to eight pending observations and a 64 MiB packed-work budget. Repeated uses of a decoded frame can reuse its packed bytes.

For an action query `t`, model inputs include events strictly before `t`; targets cover `[t, min(t + actionInterval, trimEnd))`. Events exactly at `t` enter the target and later history. Pre-trim input primes held-state timing. A legacy `supervisionStart` excludes earlier action labels while retaining earlier observation context. Zero-duration terminal targets are excluded. Past frames are ordered oldest to newest and paired with trailing-interval controls.

### Cache format

The current `TrainingDataContract.schemaVersion` is 15. A cache contains:

| File | Contents |
| --- | --- |
| `manifest.json` | Configuration, source identity, segment ranges, file layout, and context schema |
| `current-observations.bin` | One configured full-resolution packed image per observation |
| `past-observations.bin` | One configured reduced packed image per observation |
| `observation-indices.bin` | Current/past observation indices for each action row; missing history uses `UInt32.max` |
| `frame-actions.bin` | 146-value trailing-interval controls for historical observations |
| `actions.bin` | Forward action targets, also used to gather previous targets for transition weighting |
| `decision-timing.bin` | A 2,190-value Float32 timing row per action decision |
| `frame-timing.bin` | The same timing vocabulary per historical observation |
| `observation-times.bin` | Little-endian UInt64 query time and source PTS per observation, relative to recording origin |
| `decision-times.bin` | Little-endian UInt64 query time and positive target interval duration per action row |

Current-only caches have a single observation index per decision and zero-byte past-image/control/timing files. Cache validation checks file sizes, finite timing, arithmetic bounds, monotonic clocks, causal indices, configured spacing, and segment ownership. Each recording segment has separate observation and action starts so pre-supervision context is represented correctly.

Dataset files are memory-mapped. Gathering copies selected packed pixels, controls, timing, and targets into pooled Metal shared-memory inputs; MLX expands vision in the compiled graph. With augmentation disabled, deterministic CNN work can be shared across identical visual windows and overlapping historical observations, then gathered back to independent action rows before recurrence and head evaluation. Validation uses the same distinction between shared images and independent timed decisions.

### Executable action layout

`ActionLayout.count` is 146. Array ranges below are zero-based and half-open:

| Range | Meaning |
| --- | --- |
| `0..<2` | Normalized absolute cursor position |
| `2..<4` | Relative mouse movement |
| `4..<12` | Eight mouse buttons |
| `12..<14` | Scroll X/Y |
| `14..<142` | 128 macOS key-code slots |
| `142` | Shift |
| `143..<146` | Control, Option, Command |

Relative mouse targets use raw delta scale 80 and scrolling uses scale 20, with clipping to normalized ranges. Held controls and presses occurring within a sampling interval both contribute to targets, retaining taps shorter than an action tick. Shift follows the keyboard channel; Control/Option/Command use their dedicated modifier slots. Duplicate ordinary modifier-key slots are masked in data, loss, and runtime.

## Explicit timing

`TemporalInputContract.version` is 1. `CausalControlTimeline` reduces recording events during dataset construction and actually posted events during Run. Separate action and perception timelines let each cadence reset interval aggregates independently.

One timing row has 2,190 Float32 values, or 8,760 bytes. Host timestamps are subtracted as UInt64 nanoseconds before conversion to seconds, avoiding loss of small intervals from large absolute Float32 timestamps. Each duration is encoded as `min(seconds / 0.25, 1)`, `log1p(seconds) / log(61)`, and a validity bit. This represents both short and long durations and distinguishes known zero duration from missing information.

| Inclusive indices | Meaning |
| --- | --- |
| 0 | Row validity |
| 1–3 | Interval since the preceding reducer query |
| 4–6 | Interval between perception queries |
| 7–9 | Source-image age |
| 10–12 | Historical query age relative to the current decision |
| 13–15 | Separation from the preceding selected historical query |
| 16–17 | Normalized cursor X/Y |
| 18–2161 | 134 binary controls, each with 16 features |
| 2162–2175 | Movement evidence |
| 2176–2189 | Scroll evidence |

Each binary block contains held state; duration triplets for hold age, latest press age, latest release age, and latest completed hold; logarithmic interval press/release counts; and held-time fraction integrated at event boundaries. Controls seeded as held at recording startup have unknown onset duration. Runtime injection begins from a known released state.

Each motion block contains displacement, signed-log real-time rates, nonzero sample count, first/last sample offsets, a span triplet, a latest-event-age triplet, and interval validity. These features describe observed events rather than an inferred continuous physical motion. Historical source-frame age is retained while history age is recomputed at each action tick, allowing different decisions from the same cached image.

## Policy v9

`ModelContract.schemaVersion` is 9; the saved weight format is `AgentTrainer.Policy.v9`. The encoder shares convolutional weights across current and past images. Attention pooling shares its projection across image sizes; legacy flattened pooling has separate current/past projections because spatial dimensions differ. Architecture settings select GRU or LSTM history, layer widths, pooling, dropout, and precision.

The timing encoder is a bias-free 2,190 → 64 projection, LayerNorm, and SiLU. Duration projection accumulates in Float32 before casting the embedding to model precision.

The policy has two streams with shared recurrent/fusion weights and CNN embeddings:

- The complete-context stream receives past visual embeddings, complete controls, and timing.
- The grounded stream receives the same images, observation clocks/validity, and categorical context, with endogenous control inputs and their timing derivatives zeroed.

With visual history enabled, each recurrent step combines an image embedding, 146 controls, and 64 timing features. Fusion combines current visual embedding, final recurrent state, current timing embedding, and optional categorical inputs. Current-only mode uses a zero temporal state with current image/timing features. The paired streams run together along the batch dimension.

Binary heads combine grounded and complete-context logits as `base + tanh(context - base)` in Float32. This limits the history correction to one logit, reducing direct copying of generated holds. Absolute cursor, relative movement base, scroll, and movement activity use grounded features. The visual base also receives the real action labels during training, reducing dependence on demonstrated control history as an easy shortcut.

### Movement and stopping

Bounded movement history builds a convex attention mixture over current and past executed motion intervals. Time normalization averages long intervals without amplifying short ones; runtime sensitivity is removed before reuse. Attention and gain depend on grounded features, independently of the movement values being fed back. The learned gain is below 0.4. Visual timing only omits this movement prior.

Two internal per-axis activity logits supplement the 146 physical outputs. For activity logit `l`, decoding uses `u = clip((l + 1) / 2, 0, 1)` and gain `u² × (3 − 2u)`. Logits at or below −1 stop exactly; logits at or above 1 retain full movement. The smooth transition reduces abrupt camera commands near uncertain activity decisions. Activity is independent of the bounded movement prior. The final decoded action still has 146 values.

These bounds limit individual feedback paths; they do not establish that a learned controller will complete a task or remain stable in every game. Runtime rounding and sensitivity also affect whether a small predicted delta produces a posted event.

### Context inputs

The generic field/value catalog is stored by `WorkspaceStore`. Training derives `ModelContextSchema` from values assigned in selected recordings, ordering field/value UUIDs deterministically. Each value has a dimension in a multi-hot row; each field has at most one selected value, with all-zero segments for unassigned fields. Segment assignments expand during batch gathering rather than being duplicated per cached action row.

Categorical inputs are concatenated before fusion. Context-free models omit the tensor/allocation; the existing context-free weight layout therefore has no extra context parameters. Warm-start layout comparison uses ordered IDs; display-name changes alone do not change tensor shape. Saved model versions snapshot their own vocabulary, independently of later catalog edits.

## Training and saved brains

Training uses behavioral cloning, class/transition-weighted losses, focal binary loss, control-history masking, and compiled MLX updates. One per-sample keep mask drops historical controls and current/historical control-derived timing for half the training samples, while retaining images and observation clocks. This reduces the shortcut of predicting an action solely from a preceding hold.

The total objective includes immediate action loss, `0.5 × visualActionLoss`, and configured future-action/state terms. The visual-action term supervises the grounded base with the same recorded targets. The editor exposes three optional training features:

- Photometric augmentation changes luminance, contrast, saturation, hue, and gamma. One independent transform is sampled per supervised row and shared across that row's current/past/future images. No spatial transform is applied, keeping pixel/action alignment intact. Augmented rows expand before the CNN so different sample transforms remain independent.
- Future-action prediction has four configurable horizons with stronger weighting near the present.
- Future-state prediction uses an action-conditioned latent-state delta and a stopped-gradient visual target. It does not generate future pixels. Repeated future observation indices can share encoding work.

Future targets are masked at recording boundaries. The auxiliary heads are training-only; Run uses the immediate action output. Validation reports immediate action loss separately from the full training objective.

### Splits, ordering, and optimization

With multiple recording segments, validation selects whole recordings, aiming for the requested row fraction and binary-control coverage while retaining positive examples of learned controls in training. With a single recording, the engine uses a contiguous training prefix and disjoint validation suffix when feasible, extending the training boundary to cover needed controls. If disjoint validation cannot be constructed, it trains on all rows without a held-out score.

The sampler balances salient action transitions and temporal locality. Augmentation-off workloads can reuse deterministic visual encoding; batches below 16 use the diverse salience ordering rather than adjacent local batches. CPU preparation of the next batch overlaps the current Metal graph. Each step evaluates model, optimizer, scheduler, and random state before advancing its checkpoint position.

`ResumableAdamW` uses Float32 master weights and moment accumulators, then casts executed model weights to configured precision. Master weights retain updates smaller than one low-precision weight rounding step. Gradient-norm accumulation also uses Float32. The default adaptive cosine schedule has warmup, bounded restarts, and a validation-adjusted envelope; legacy optimizer state can retain its saved arithmetic/schedule. Optimizer format is `AgentTrainer.AdamW.v2`.

Metrics are rate-limited and chart histories are bounded. Diagnostics include step/input timing, MLX active/cache/peak allocations, throughput retention, thermal state, and per-loss terms. MLX allocation counters describe MLX-managed memory rather than total process or device memory.

### Checkpoints, versions, and compatibility

A checkpoint stores model weights, Float32 masters, AdamW moments/metadata, scheduler state, training progress, model/data identity, and MLX random state. Compatible resume restores those together, including the batch cursor and sampling version. Missing or mismatched resume artifacts produce an error instead of an exact-resume claim. Weights-only activation starts a fresh optimizer from the selected brain and removes an unrelated resumable checkpoint.

Pause publishes a runnable snapshot and retains a resume checkpoint. Ordinary completion selects the best acceptable held-out-validation brain when available; training without validation uses its non-validation publication path. Periodic saves prune bounded history while protecting active/protected versions. Publication and profile activation use staged filesystem transactions.

Saved versions snapshot architecture, vision, precision, channels, cursor visibility, context schema, and demonstrated/output capabilities. Native v9 binary capabilities derive from positive classes in the training split, disabling heads without supervised examples. Run takes learned settings from the saved version while applying current profile restrictions and channel disables.

The learned-layout comparison normalizes dropout out of the architecture. A dropout-only edit can retain compatible brain weights, while its changed stochastic objective starts a fresh optimizer sequence. Ordered categorical IDs and other learned-layout settings participate in compatibility checks.

The current migration archives incompatible versions/checkpoints instead of converting v7/v8 weights into v9. It retains recordings, profile architecture/temporal settings, and compatible schema-15 caches. Weight loading checks format metadata as well as parameter shapes, distinguishing same-shaped weights with different policy semantics. Older removed-feature settings are ignored on decode and omitted when saved.

## Live reinforcement learning

PPO reuses Policy v9's encoder, temporal network, and action heads. A separate Float32 critic and learned exploration parameters do not change the standalone policy weight layout. The learner re-encodes masked original pixels and causal controls/timing rather than differentiating stale inference embeddings. Dropout and demonstration-only objectives are disabled for PPO likelihood consistency; those heads remain available for later demonstration training.

Binary outputs use a masked Bernoulli distribution with explicit exploration; mouse values and movement intent use Gaussian latent samples followed by the existing deterministic decoder. PPO retains the latent samples and their exact behavior probabilities. Short cumulative reward intervals can contain multiple executed actions; their joint trajectory likelihood and elapsed duration are retained. RL does not invent which individual missed frame earned a cumulative delta.

Compiled microbatches combine independent intervals without mixing their likelihoods, and gradients accumulate to the configured optimizer minibatch. Probability/advantage/value arithmetic and master weights/moments use Float32. KL stopping and finite-value checks precede optimizer updates. The actor receives owned weight buffers, refreshes selected historical embeddings by policy revision, and retains bounded raw history.

Demonstration checkpoints carry active-policy lineage. A method switch loads the active compatible brain and restarts stale optimizer state, preserving previous artifacts. Newly introduced context dimensions widen input projections with zero-initialized columns while mapping existing context IDs. RL and demonstration timing counters remain separately attributable in profile progress.

See [Reinforcement learning](reinforcement-learning.md) for the full environment and persistence contract.
