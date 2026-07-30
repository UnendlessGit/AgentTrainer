#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

# Xcode produces MLX's required Metal library. Refresh it when dependency
# resolution changes, without touching the installed app or release artifacts.
XCODE_PRODUCTS="$ROOT/.build/xcode/Build/Products"
METALLIB="$XCODE_PRODUCTS/release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
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
if [[ ! -f "$METALLIB" ]]; then
  METALLIB="$(find "$XCODE_PRODUCTS" -type f \( -name 'default.metallib' -o -name 'mlx.metallib' \) | head -1)"
fi
if [[ -z "$METALLIB" || ! -f "$METALLIB" ]]; then
  echo "Xcode did not produce the MLX Metal test library." >&2
  exit 1
fi

swift test -c debug list >/dev/null
TEST_BUNDLE="$(find "$ROOT/.build" -type d -path '*/debug/AgentTrainerPackageTests.xctest' ! -path '*/xcode/*' | head -1)"
if [[ -z "$TEST_BUNDLE" || ! -d "$TEST_BUNDLE/Contents/MacOS" ]]; then
  echo "SwiftPM did not produce the AgentTrainer test bundle." >&2
  exit 1
fi
TEST_MACOS="$TEST_BUNDLE/Contents/MacOS"
cp "$METALLIB" "$TEST_MACOS/mlx.metallib"
swift test -c debug
