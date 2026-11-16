# AgentTrainer

A native macOS app for **recording** yourself playing a game or using an app, **training** a
Pixel2Play-style agent on those recordings locally on Apple Silicon, and **running** it in real time.

Everything runs on the Mac: capture (ScreenCaptureKit), training and inference (MLX on the GPU),
instruction understanding (EmbeddingGemma) and automatic labelling (a local vision-language model).

## What's inside

| Tab | What it does |
|---|---|
| **Recordings** | Pick a window or display, resolution and frame rate, which inputs to capture, the Library folder to save into and an optional task description. A HUD (hidden from capture) shows the keys, buttons, mouse movement and scrolling being recorded. |
| **Library** | Real folders on disk: create, rename, delete, drag recordings between folders. Inspector with video, an input-activity timeline, timed instructions (manual or written by the local AI) and ranges to leave out of training. |
| **AI Models** | Presets (Pixel2Play 150M/300M pretrained, App control, Compact 60M, Medium 22M, Small 10M, Tiny 5M) and every setting: network size, vision input, decision rate and memory length, instructions / screen text / typing, the keys and mouse actions the model may use, training hyperparameters, plus an on-device speed benchmark. |
| **Training** | Choose recordings, start / pause / resume / stop, live loss and validation charts, GPU use, memory, thermals, ETA, checkpoints (resume from any, choose which one Run uses), optional best-model saving and early stopping. |
| **Run** | Choose a model and a target, allow or block keyboard / typing / mouse movement / buttons / scrolling (without touching the model), give an instruction, watch FPS and latency. Touching the keyboard or mouse takes over; your corrections can be recorded for further training. |
| **Test Arena** | Built-in mini-games with a scripted expert to verify the whole pipeline: generate demonstrations, play them yourself, evaluate a model closed-loop, measure input/frame sync. |
| **Settings** | Storage locations, global shortcuts, GPU memory limit, downloads, permissions. |

## The model

The same architecture as **Open Pixel2Play** (elefant-ai/open-p2p), re-implemented in MLX and verified
against P2P's PyTorch code (logits match to 7·10⁻⁵). P2P's released weights can be imported as the
starting point. Details, extensions and measured numbers: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

Measured on an M3 Max (36 GB):

- **Inference**: P2P-150M with 10 s of memory at 30 Hz: 13 ms per step (~76 FPS); 300M: ~48 FPS. In a live run on the Test Arena: 30.1 decisions/s, 14.5 ms per step, 25 ms screen → action, no missed ticks.
- **Training**: Compact 60M, 10-second clips at 30 Hz, batch 3: 1.30 s per step (~690 frames/s, 1.65× faster than 1.0.0 with identical math), ~10 GB peak. Tiny 5M: ~1,550 frames/s.
- **Sync**: key press → captured frame 16 ms median, ±5 ms jitter.

## Build and install

```bash
scripts/build.sh      # Release build + signing verification
scripts/install.sh    # installs to /Applications, keeps permissions
scripts/test.sh       # unit tests
```

Every build is signed with the same Apple Development identity and a designated requirement pinned to
`com.unendless.AgentTrainer` + team `V5S6J7KC33` (see `Config/Signing.xcconfig` and `Config/*.req`),
so macOS keeps Screen Recording, Input Monitoring and Accessibility granted across rebuilds and updates.

## Developer CLI

```bash
atctl parity <p2p.ckpt> <ref.safetensors>     # MLX vs PyTorch Pixel2Play
atctl stream-parity                           # streaming KV cache vs full forward
atctl bench --ckpt <p2p.ckpt>                 # real-time step latency
atctl train-bench                             # training step time / memory
AT_TRAIN_PROFILE=1 atctl train ...            # per-phase timing of real training steps
AT_LEGACY_CHECK=1 atctl train ...             # new vs original training path, same batch
atctl arena-gen --mode chase --minutes 10 --out <dir>
atctl train --model m.atmodel --data <dir> --create gameP2P150 --steps 600
atctl arena-eval --model m.atmodel --mode chase [--diagnose]
```

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for licenses of the model architecture, weights and libraries.
