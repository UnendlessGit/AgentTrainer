# AgentTrainer 2.3.5

AgentTrainer 2.3.5 fixes physical controls that could appear as repeated taps or releases when recording another game or app.

## Highlights

- Physical reconciliation now reads macOS's HID-system hardware state instead of the foreground login session's accumulated event-source state. Another process can no longer make a physically held key such as W appear released in a recording.
- The same hardware-state contract covers modifiers and all tracked mouse buttons, including additional gaming-mouse buttons.
- Reconciliation snapshots carry a state revision. If a first-frame seed, DAgger seed, or physical edge arrives while polling is in flight, the stale poll is discarded instead of overwriting newer input truth.
- Missed-release recovery remains intact: the next fresh hardware snapshot repairs genuinely lost key, modifier, and mouse-button releases, and recording shutdown still writes balanced terminal releases.
- Hotkey lifecycle suppression, recording key exclusions, immediate HUD state, event-file ordering, replay, normal training, runtime takeover safety, and DAgger correction capture retain their existing behavior.

## Verification

- 161 automated tests executed: 159 passed and the 2 opt-in hardware benchmarks were skipped in the normal suite.
- New regressions reproduce divergent foreground-session and HID state for W, Shift, and an additional mouse button, and prove that a stale poll cannot release a newly seeded held key.
- Compact, default, and 16-frame temporal-stress release benchmarks passed matched-quality, throughput, bounded-memory, sustained-load, and thermal gates; every reported thermal state was nominal.
- The 1280×720 Record-quality DAgger gate encoded 60 frames at 145.6 fps with zero drops.

AgentTrainer remains local-first, requires Apple silicon and macOS 15 or later, and ships as a hardware-HEVC macOS application.
