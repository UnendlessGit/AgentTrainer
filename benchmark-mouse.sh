#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

# MLX Swift 0.31.3's metallib is produced by its Xcode build support but is not
# copied into SwiftPM's test executable. Keep this in sync with `test.sh`.
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
    print -u2 "Xcode did not produce the MLX Metal test library."
    exit 1
fi

swift test --build-system native -c release list >/dev/null
TEST_BUNDLE="$(find "$ROOT/.build" -type d -path '*/release/AgentTrainerPackageTests.xctest' ! -path '*/xcode/*' | head -1)"
if [[ -z "$TEST_BUNDLE" || ! -d "$TEST_BUNDLE/Contents/MacOS" ]]; then
    print -u2 "SwiftPM did not produce the AgentTrainer release test bundle."
    exit 1
fi
cp "$METALLIB" "$TEST_BUNDLE/Contents/MacOS/mlx.metallib"

# Both candidates use the same real recordings, split, architecture, initial
# seed, and optimizer. Each stops at exactly 20 epochs. Completed arms only
# evaluate on reruns; neither arm can change an existing user brain.
export AGENTTRAINER_WORKSPACE_ROOT="$ROOT/work/mouse-comparison-v8"
export AGENTTRAINER_MOUSE_CORRECTION=1
benchmark_status=0
for history_mode in visual bounded; do
    if ! AGENTTRAINER_MOUSE_HISTORY_MODE="$history_mode" swift test --build-system native -c release --skip-build --filter 'MouseCorrectionTests.testRealRebornTrainingAndFeedback'; then
        benchmark_status=1
    fi
done
exit "$benchmark_status"
