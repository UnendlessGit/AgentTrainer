# Third-party notices

AgentTrainer re-implements, in Swift/MLX, the model architecture of **Open Pixel2Play (P2P)** and can
load its released weights.

- **open-p2p** — https://github.com/elefant-ai/open-p2p — MIT License, Copyright (c) 2025 elefant-ai.
  Weights: https://huggingface.co/elefantai/open-p2p (MIT). Paper: "Scaling Behavior Cloning Improves
  Causal Reasoning: An Open Model for Real-Time Video Game Playing" (Yue et al., 2026, arXiv:2601.04575).
- **MLX / MLX Swift / mlx-swift-lm** — https://github.com/ml-explore — MIT License, Copyright © Apple Inc.
- **swift-transformers** — https://github.com/huggingface/swift-transformers — Apache License 2.0.
- **EmbeddingGemma-300m** (downloaded on request, not bundled) — Google, Gemma Terms of Use
  (https://ai.google.dev/gemma/terms). Used as the instruction / screen-text encoder, as in P2P.
- **Qwen3-VL-4B-Instruct (MLX 4-bit)** (downloaded on request, not bundled) — Alibaba Qwen, Apache License 2.0.
  Used locally to write instructions for recordings.
- **torchvision EfficientNet-B0** architecture (BSD-3-Clause) — re-implemented in MLX; ImageNet-trained
  values arrive only through P2P's checkpoint.
