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

swift test -c release list >/dev/null
TEST_BUNDLE="$(find "$ROOT/.build" -type d -path '*/release/AgentTrainerPackageTests.xctest' ! -path '*/xcode/*' | head -1)"
if [[ -z "$TEST_BUNDLE" || ! -d "$TEST_BUNDLE/Contents/MacOS" ]]; then
    print -u2 "SwiftPM did not produce the AgentTrainer release test bundle."
    exit 1
fi
cp "$METALLIB" "$TEST_BUNDLE/Contents/MacOS/mlx.metallib"

benchmark_filter="TrainingPerformanceTests.testCompiledMetalDataPathPreservesOrderedUpdatesAndMeasuresThroughput"
AGENTTRAINER_RUN_PERFORMANCE_TESTS=1 \
    AGENTTRAINER_BENCHMARK_SUSTAINED_STEPS=0 \
    AGENTTRAINER_BENCHMARK_SUSTAINED_BASELINE_STEPS=0 \
    swift test -c release --skip-build --filter "$benchmark_filter"
AGENTTRAINER_RUN_PERFORMANCE_TESTS=1 \
    AGENTTRAINER_BENCHMARK_DEFAULT_PROFILE=1 \
    AGENTTRAINER_BENCHMARK_SUSTAINED_STEPS="${AGENTTRAINER_BENCHMARK_SUSTAINED_STEPS:-256}" \
    AGENTTRAINER_BENCHMARK_SUSTAINED_BASELINE_STEPS=0 \
    swift test -c release --skip-build --filter "$benchmark_filter"
AGENTTRAINER_RUN_PERFORMANCE_TESTS=1 \
    AGENTTRAINER_BENCHMARK_STRESS_PROFILE=1 \
    AGENTTRAINER_BENCHMARK_SUSTAINED_STEPS="${AGENTTRAINER_BENCHMARK_STRESS_STEPS:-192}" \
    AGENTTRAINER_BENCHMARK_SUSTAINED_BASELINE_STEPS="${AGENTTRAINER_BENCHMARK_STRESS_BASELINE_STEPS:-0}" \
    swift test -c release --skip-build --filter "$benchmark_filter"
