# AgentTrainer

A native macOS app for recording gameplay or app use, training Pixel2Play-style
behavior-cloning agents locally on Apple Silicon (MLX), and running them in real time.

> Work in progress — see `docs/ARCHITECTURE.md` as milestones land.

## Build

```bash
scripts/build.sh          # Release build + signing verification
scripts/install.sh        # install to /Applications (keeps macOS permissions between updates)
scripts/test.sh           # unit tests
```

Signing uses a fixed Apple Development identity and a designated requirement pinned to
`com.unendless.AgentTrainer` + team `V5S6J7KC33` (see `Config/Signing.xcconfig`), so Screen
Recording, Accessibility and Input Monitoring permissions persist across rebuilds.
