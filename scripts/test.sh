#!/bin/zsh
# Runs the AgentKit unit tests (MLX needs xcodebuild so its Metal kernels get built).
set -euo pipefail
cd "$(dirname "$0")/../Packages/AgentKit"
xcodebuild test -scheme AgentKit-Package -destination 'platform=macOS' \
  -derivedDataPath ../../build/DerivedData "$@" 2>&1 | grep -E "✔|✘|passed|failed|error:|TEST SUCCEEDED|TEST FAILED" | grep -vE "^\s*$" || true
