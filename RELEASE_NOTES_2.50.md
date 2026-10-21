# AgentTrainer 2.50

AgentTrainer now supplies explicit timing to Policy v7 on every action decision: held-key/button ages, press/release recency and durations, tap counts, held-time fractions, movement/scroll sample timing and rates, observation intervals, source-frame age, and historical ages. Unknown startup durations and missing history have explicit validity indicators. Timing stays active with past images disabled.

The CNN runs at Perception FPS while a lightweight policy runs at Action FPS using cached visual embeddings and fresh feedback from events actually posted. GRU/LSTM, categorical context, all 146 immediate outputs, and predictive training objectives remain available. At 60 Action FPS execution resolution remains approximately 16.7 ms; this release does not add arbitrary sub-tick HID scheduling.

Cache construction preserves real video PTS at trims and excludes future events from current inputs. Decisions sharing an image retain independent timing. Runtime preserves panic/takeover, forced releases, permission/context revisions, bounded serialized work, stale-inference stopping, and stop/restart ordering. Untrained binary heads cannot post keys/buttons absent from supervised training.

## Migration

Recordings keep their original video and event bytes. Disposable caches rebuild under dataset schema 15. Incompatible brains/checkpoints are preserved in the managed archive; retrain them for model schema 7. Profile architecture and temporal settings are retained. There is no exact v6-to-v7 resume or weight conversion; exact pause/resume remains supported within v7.

## Measured validation

Both matched Blade Ball arms used all 145 recordings, whole-recording held-out splits, 60 Perception FPS / 60 Action FPS, 128×128 vision, binary focal gamma 0, and exactly 20 epochs (112,220 updates each). Timing improved held-out immediate loss **5.22%** and relative mouse MAE **13.75%**. A controlled identical-image duration task reached **95.31%** accuracy. Some real-data edge-timing, hold-duration, precision, and scroll metrics worsened; these offline measurements do not establish better gameplay. Full methods and results: [TEMPORAL_DESIGN.md](https://github.com/UnendlessGit/AgentTrainer/blob/v2.50/TEMPORAL_DESIGN.md).

Compact, default, and 16-frame stress release benchmarks passed the existing quality, throughput-retention, memory, and auxiliary-overhead gates. Cached action-head p95 latency was **0.66–1.02 ms** on the test Mac, excluding capture/CNN/queueing/HID latency. The functional suite passed 181 tests with two opt-in tests skipped; both opt-in workflows were run separately. Signing regressions passed. No performance thresholds were relaxed.

Version **2.50, build 35**, tag `v2.50` (ordered after 2.47 by the updater). Requires Apple silicon and macOS 15+. The build preserves the existing designated signing requirement and default `/Applications/AgentTrainer.app` installation. This release uses Apple Development signing and is not notarized; a new Mac still needs its own privacy grants. Private keys, local signing pins, recordings, and experiment artifacts are excluded from source assets.
