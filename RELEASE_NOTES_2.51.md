# AgentTrainer 2.51 candidate — live validation rejected

Policy v8 addresses a reproduced mouse/game-camera regression where previous movement could amplify the next prediction. Bounded movement history retains recent executed motion through time-normalized, limited feedback. A second training option, Visual timing only, omits movement feedback from the motor path. Both preserve visual history, key/button durations, and observation timing. Separate per-axis activity decisions allow exact mouse stops.

The physical action vocabulary remains 146 values. The shared CNN runs at Perception FPS and policy decisions run at Action FPS. Runtime history still reflects actual posted events after restrictions and rounding. Validation now measures idle movement at the actual half-pixel posting boundary at 1× sensitivity, and output capabilities use actual positive key/button counts.

The first Reborn live comparison rejected both candidates: persistent key holds and inadequate camera movement remained, including on a recorded trial. Replay gains did not translate into acceptable live behavior. This candidate is installed locally for investigation and is not a published GitHub release.

## Migration

Model schema 8 requires retraining. Incompatible brains/checkpoints are preserved in the managed archive; recordings, profile architecture/settings, and schema-15 caches remain intact. There is no v7-to-v8 exact optimizer resume or weight conversion. Exact pause/resume remains supported within v8.

## Validation

Two matched candidates use only 39 real Reborn recordings, 60/60 FPS, 128×128 vision, binary focal gamma 0, and exactly 20 epochs per arm. Production event-posting feedback is evaluated at full and reduced perception cadences. Recorded-video feedback measurements supplement, and cannot replace, an actual live-game comparison. Full methods and verified results are documented in MOVEMENT_CONTROL.md.

Version **2.51, build 36**, tag `v2.51`. Apple silicon, macOS 15+. Signing identity and the default `/Applications/AgentTrainer.app` installation are preserved. Apple Development signing is not notarization; a new Mac needs its own privacy grants. Private recordings, weights, experiment artifacts, signing pins, and keys are excluded from source assets.
