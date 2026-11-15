#!/bin/zsh
# Builds Release and replaces /Applications/AgentTrainer.app in place.
# The bundle identifier and designated requirement never change, so macOS keeps
# the Screen Recording / Accessibility / Input Monitoring grants across updates.
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/build.sh Release
SRC=build/DerivedData/Build/Products/Release/AgentTrainer.app
DST=/Applications/AgentTrainer.app
osascript -e 'tell application id "com.unendless.AgentTrainer" to quit' 2>/dev/null || true
sleep 1
rm -rf "$DST"
ditto "$SRC" "$DST"
scripts/verify-signing.sh "$DST"
"$DST/Contents/MacOS/AgentTrainer" --smoke-test
echo "Installed $DST"
