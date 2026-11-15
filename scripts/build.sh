#!/bin/zsh
# Usage: scripts/build.sh [Debug|Release]
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:-Release}
xcodebuild -project AgentTrainer.xcodeproj -scheme AgentTrainer -configuration "$CONFIG" \
  -derivedDataPath build/DerivedData -destination 'platform=macOS' build -quiet \
  2> >(grep -E "error:|BUILD FAILED" >&2)
APP=build/DerivedData/Build/Products/$CONFIG/AgentTrainer.app
scripts/verify-signing.sh "$APP"
echo "Built $APP"
