#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

# Xcode produces MLX's required Metal library. Refresh it when dependency
# resolution changes, without touching the signed app or distribution outputs.
METALLIB="$ROOT/.build/xcode/Build/Products/release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
if [[ ! -f "$METALLIB" || "$ROOT/Package.resolved" -nt "$METALLIB" ]]; then
  xcodebuild \
    -skipPackagePluginValidation \
    -scheme AgentTrainer \
    -destination 'platform=macOS,arch=arm64' \
    -configuration release \
    -derivedDataPath "$ROOT/.build/xcode" \
    CLANG_ENABLE_CODE_COVERAGE=NO \
    GCC_GENERATE_TEST_COVERAGE_FILES=NO \
    GCC_INSTRUMENT_PROGRAM_FLOW_ARCS=NO \
    build >/dev/null
fi

swift test -c debug list >/dev/null
TEST_MACOS="$ROOT/.build/arm64-apple-macosx/debug/AgentTrainerPackageTests.xctest/Contents/MacOS"
cp "$METALLIB" "$TEST_MACOS/mlx.metallib"
swift test -c debug
