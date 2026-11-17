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
# Move, don't copy: a second AgentTrainer.app left in build/ shows up in Spotlight,
# Launchpad and "Open With" next to the real one (often with a stale icon).
LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
"$LSREGISTER" -u "$PWD/$SRC" 2>/dev/null || true
mv "$SRC" "$DST"
"$LSREGISTER" -f "$DST"
touch "$DST"
scripts/verify-signing.sh "$DST"
"$DST/Contents/MacOS/AgentTrainer" --smoke-test
echo "Installed $DST"
