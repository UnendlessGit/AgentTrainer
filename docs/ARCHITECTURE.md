# AgentTrainer architecture

## Processes and modules

| Piece | Where | What it does |
|---|---|---|
| `AgentTrainer.app` | app process | SwiftUI UI, recording (ScreenCaptureKit + event tap), real-time inference, HUD |
| `AgentTrainerEngine.xpc` | XPC service | training and auto-labelling jobs (crash-isolated, frees GPU memory when done) |
| `ATCore` | package | formats (recordings, models), action space, timeline alignment, key catalog |
| `ATCapture` | package | capture, input tap, Metal resizer, recording session, OCR, Test Arena |
| `ATModel` | package | the Pixel2Play policy in MLX, streaming stepper, checkpoint import |
| `ATTraining` | package | dataset, loader, augmentation, trainer, validation |
| `ATLanguage` | package | EmbeddingGemma instruction encoder, VLM auto-labeller |
| `ATRuntime` | package | live agent loop, action executor, arena evaluator |
| `atctl` | CLI | parity checks, benchmarks, arena data, headless training |

## The model (Pixel2Play, ported exactly)

Per timestep, 12 tokens: `[text, image, thinking, a_out, 8 action tokens]`.

- **Image**: EfficientNet-B0 `features[0:6]` on 192×192 RGB (`/255`, no mean/std) → flatten → Linear → LayerNorm → 1 token.
- **Text**: EmbeddingGemma-300m, masked mean of the last hidden state (no Dense/normalize) → Linear(768→D, no bias); absent → learned "no text" token.
- **Backbone**: pre-norm transformer, fp32 RMSNorm, fused qkv, QK-RMSNorm over the full width, interleaved RoPE (base 1e4) on the flat token index, SwiGLU, no final norm. Block-causal mask: image-type tokens see each other inside a step; `a_out` is never read by other tokens; `a_out(t)` can't see actions(t); past actions are visible.
- **Action decoder**: 3-layer causal transformer at width D with one learned KV sink per layer; inputs `[proj(a_out), emb(a0..a7)]`; teacher-forced in training, sampled (Gumbel-max) at inference.
- **Actions**: 4 key slots (sorted set), 2 pointer slots, mouse dx/dy bins (P2P's log-spaced edges, truncated-normal decoding). AgentTrainer extends the vocabularies (full keyboard, back/forward buttons, scroll folded into the pointer slots, optional absolute pointer grid) without changing the 12-token layout, so P2P's weights load directly; new rows start silent.

Parity: on fixed inputs the MLX port matches P2P's PyTorch logits to 7·10⁻⁵ (`atctl parity`, `tools/p2p_reference`).

### Streaming inference

`PolicyStepper` runs **one** 12-token backbone pass per timestep: the previous step's action tokens plus this frame's `[text, image, thinking, a_out]` (equivalent to P2P's two passes because nothing sees same-step actions except other action tokens). The KV cache is a ring of `memory × rate` timesteps (10 s × 30 Hz = 300 steps = 3600 tokens), keys stored with RoPE applied and periodically re-based so positions never lose precision. Streaming matches the full forward to 2·10⁻⁶ across window slides. The 8-step decoder sampling loop is compiled. P2P-150M (183M params) with a full 10 s window: **13 ms / step (~76 FPS) on an M3 Max**.

**GPU clock.** At a fixed 30 Hz the model leaves the GPU idle about half the time, and macOS then lowers the GPU clock: paced steps drifted from 14 ms to 30–37 ms after a few hundred steps (p90 30 ms), even though unpaced steps stay at 13 ms. `GPUWarmer` fixes this by queueing a tiny one-threadgroup spin kernel after each step that ends ~2.5 ms before the next tick, so the GPU stays clocked up without competing with the model (paced 700-step run: mean 13.9 ms, p99 17.8 ms; live in-app: 14.5 ms early vs 14.6 ms late). Settings → Performance → "Keep the GPU clocked up while a model runs" (on by default).

## Recording and alignment

Frames carry ScreenCaptureKit's display time; input events carry their CGEvent timestamp, converted to the same host clock (domain auto-detected). A recording stores HEVC video, a **lossless 192×192 proxy made from the raw frames with the same Metal Hamming resize used at inference**, and binary frame/event timelines. Training resamples to the model's rate: the input at tick k is the latest frame displayed by τk, the label is the input in (τk, τk+1] (Pixel2Play's convention), keys/buttons = held at the end **or pressed during** the window (P2P loses quick taps), pointer deltas and scroll summed. Measured key-to-frame latency on this machine: 16 ms median, ±5 ms jitter (Sync Check).

## Training

- fp32 master weights + AdamW state, bf16 compute model (3× faster, half the memory of fp32).
- **Stem gradient sampling**: a random subset of frames (32/clip) goes through EfficientNet with gradients scaled N/K (unbiased), the rest without gradients. BatchNorm statistics are frozen (clips are single-recording, so batch statistics would be biased). 10 s clips at 30 Hz train at ~1.4 s/clip, ~11 GB peak.
- P2P loss (CE / log #classes per head + z-loss), warmup + cosine schedule, gradient clipping, P2P's clip-consistent augmentations.
- **Anti-copycat action-history dropout** (beyond P2P): past-action tokens are hidden in random spans and replaced by a learned "unknown" token, forcing decisions from vision. In the Test Arena this raised vision-only agreement with the expert from 34% to 78%.
- Validation on held-out *recordings*: per-head NLL/perplexity, relative cross-entropy vs. label frequency, key/click onset F1, mouse direction/correlation, visual reliance (frames shuffled in time), history reliance (past actions hidden), language reliance (instruction removed). The stem runs once per validation clip; the four passes reuse its tokens.
- **Best model and early stopping** (both optional): every validation that lowers the held-out loss saves checkpoint `best` (never pruned; Run uses it unless another is pinned). Early stopping ends the run after N checks without improvement.

### Training speed (same math, measured on the Compact 60M model, 10 s × 30 Hz clips, batch 3)

| | step | frames/s |
|---|---|---|
| before | 2.15 s | 420 |
| after | 1.30 s | 690 |

What changed, each checked against the original path:
- **Vision stem.** BatchNorm statistics are frozen, so each BatchNorm is folded into its conv weights and bias + SiLU run as one fused kernel; the squeeze-excite gates and the projection's BatchNorm are folded into per-sample 1×1 projection weights. Depthwise convolutions get a hand-written backward (MLX's grouped-conv gradients were 10–40× slower than the forward): strided input gradients via zero-insertion + a stride-1 correlation, weight gradients via a small Metal kernel. Stem forward+backward: 5× faster. Each frame now goes through the stem once (sampled frames with gradients, the rest without), instead of the sampled ones twice.
- **Attention.** The mask is causal at timestep granularity, so attention is computed in query blocks that read only the keys they can see (exactly equal, ~40% fewer scores), with an additive mask (faster in MLX's fused kernels).
- **Whole step compiled.** Forward + backward is one `compile`d graph (fused elementwise ops, no re-tracing); identical loss and gradients to the uncompiled step.
- **Optimizer.** AdamW with MLX's exact update, each parameter's update + bf16 copy one fused kernel; same checkpoint format.
- **Augmentation.** Warp, colour and blur are single Metal passes reading the uint8 frames (190 → 40 ms per step).
- **Embedding gradients in fp32.** Found while checking the compiled step: in bf16, the action-embedding gradients (thousands of contributions scatter-added into a few rows) differed by up to 18% between identical runs. The compute copy now keeps these four tables in fp32 (agreement 4·10⁻⁸). This is a precision fix, not a speed change.
- Verification: in fp32, new vs original gradients differ by 1.2·10⁻⁷ overall; in bf16 both paths are equally far from an fp32 reference (0.50% overall). P2P parity (8·10⁻⁵) and streaming parity are unchanged; unit tests cover each rewrite. Larger batches gave no extra throughput (the GPU is saturated), so automatic batch size is unchanged.

## Language

- **Instructions**: EmbeddingGemma text token (P2P), instruction dropout, timed segments edited in the Library or written by the local VLM (P2P used Gemini with a goal-level prompt; AgentTrainer adapts that prompt for Qwen3-VL-4B running on the Mac).
- **Screen text** (optional): Apple Vision OCR at 2 Hz, top items embedded with EmbeddingGemma + boxes. A zero-initialized summary is added to the thinking token (so it enters the 10 s memory) and items feed a **gated cross-attention memory** in the action decoder (gate starts closed, Flamingo-style), so pretrained behavior is unchanged until training opens it.
- **Literal typing** (optional): instruction bytes join the decoder memory; typing bursts in recordings become `type "…"` instructions automatically.

## Run

A fixed-rate loop on a real-time thread: newest captured frame → shared GPU resize → stepper → action executor (key/button diffs, relative deltas or absolute pointer, scroll), all events tagged as synthetic. Run-tab permissions filter actions without touching the model. Physical input pauses the model and becomes its action history; corrections can be recorded for DAgger-style fine-tuning (model-controlled ranges are masked from the loss).
