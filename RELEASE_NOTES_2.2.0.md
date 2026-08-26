# AgentTrainer 2.2.0

AgentTrainer 2.2 makes correction capture and visual generalization behave like one coherent workflow.

## Highlights

- DAgger now visibly suspends Run, switches the app and capture-excluded HUD to **DAgger Recording**, and resumes the resident AI with stale temporal state cleared when recording stops.
- DAgger uses the normal Record source, FPS, cursor visibility, output resolution, Color Bit, compression, and universal lead-up. Correction weight remains AI-specific, and ordinary physical takeover never starts recording implicitly.
- Record adds configurable output resolution with ratio lock, 1–8 bit color detail, and a simple continuous compression control. All new settings and DAgger lead-up are included in recording presets, with safe decoding of older presets.
- Visual Augmentation now samples an independent photometric transform for every supervised sample while sharing that transform across only the sample's current, temporal, and future frames.
- Future Action Prediction and Future State Prediction continue to train jointly with the immediate policy without adding runtime action latency.

## Verification

- 155 automated tests executed: 153 passed and 2 hardware benchmarks skipped in the normal suite.
- Compact, default, and temporal-stress release benchmarks passed separate gates for per-sample augmentation and predictive-head overhead.
- The 1280×720 Record-quality DAgger video benchmark encoded 60 frames above 140 fps with zero drops on the release test Mac.

AgentTrainer remains local-first, requires Apple silicon and macOS 15 or later, and ships as a hardware-HEVC macOS application.
