# AgentTrainer 2.3.0

AgentTrainer 2.3 adds a generic Context / Labels system across recording, Library management, training, and Run. Nothing is game-specific or predefined.

## Highlights

- Create any number of custom context fields and values, select them on Record, and retain them in recording presets.
- Inspect and edit context in Library for one recording or a multi-selection. Mixed bulk values stay unchanged until explicitly replaced; metadata changes are transactional and never rewrite video or synchronized input.
- Portable recording exports include their referenced context catalog. Standalone recording packages remain self-describing, and imports safely remap conflicting field/value identifiers.
- Training conditions Policy v6 on a compact categorical vector derived only from fields and values actually observed in the selected recordings. Blank fields use an all-zero segment.
- Run loads the immutable trained vocabulary from the active brain and shows every field blank for the current operating context. Live changes release stale held controls, clear temporal state, and affect the next admitted perception. Brains trained without context show no context UI.
- Existing recordings, presets, profiles, and context-free Policy-v6 brains remain compatible; context-free models keep their exact established parameter and compiled-input layout.

## Verification

- 159 automated tests executed: 157 passed and 2 opt-in hardware benchmarks skipped in the normal suite.
- Focused coverage includes catalog validation, transactional single/bulk edits, ID-collision-safe import/export, stable cache identity, real video-cache propagation, blank/multi-hot encoding, model sizing, and Policy-v6 context fusion.
- Record, Library, bulk-editor scrolling, the field/value manager, and Run context layouts were visually checked at the standard app window size in an isolated workspace.
- Compact, default, and 16-frame temporal-stress release benchmarks passed matched-quality, full-objective, bounded-memory, sustained-throughput, and thermal gates; every reported thermal state was nominal.
- The 1280×720 Record-quality DAgger gate encoded 60 frames at 132.3 fps with zero drops on the release Mac.

AgentTrainer remains local-first, requires Apple silicon and macOS 15 or later, and ships as a hardware-HEVC macOS application.
