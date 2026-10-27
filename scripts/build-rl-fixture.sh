#!/bin/zsh
set -euo pipefail
ROOT="${0:A:h:h}"
APP="/Applications/AgentTrainer.app"
SIGN_IDENTITY="${CODE_SIGN_IDENTITY:-}"
SIGN_KEYCHAIN="${CODE_SIGN_KEYCHAIN:-}"
source "$ROOT/scripts/signing.sh"
configure_signing
FIXTURE_APP="$ROOT/work/rl-validation/RLTestGame.app"
if pgrep -x RLTestGame >/dev/null 2>&1; then
  print -u2 -- "Quit the RL validation game before rebuilding it."
  exit 1
fi
mkdir -p "$FIXTURE_APP/Contents/MacOS"
FIXTURE_SOURCE="$ROOT/Tests/Fixtures/RLTestGame.swift"
if [[ "${AGENTTRAINER_FIXTURE_AUDIT:-0}" == 1 ]]; then FIXTURE_SOURCE="$ROOT/Tests/Fixtures/RLAuditGame.swift"; fi
xcrun swiftc "$FIXTURE_SOURCE" -O -target arm64-apple-macos15.0 -o "$FIXTURE_APP/Contents/MacOS/RLTestGame" -framework AppKit
cat > "$FIXTURE_APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd"><plist version="1.0"><dict><key>CFBundleIdentifier</key><string>local.agenttrainer.rlfixture</string><key>CFBundleExecutable</key><string>RLTestGame</string><key>CFBundleName</key><string>RL Validation Game</string><key>CFBundleShortVersionString</key><string>2.60</string><key>CFBundleVersion</key><string>38</string></dict></plist>
PLIST
KEYCHAIN_ARGUMENTS=()
if [[ -n "$SIGN_KEYCHAIN" ]]; then KEYCHAIN_ARGUMENTS=(--keychain "$SIGN_KEYCHAIN"); fi
codesign --force --options runtime --timestamp=none "${KEYCHAIN_ARGUMENTS[@]}" --sign "$SIGN_IDENTITY" "$FIXTURE_APP"
codesign --verify --deep --strict "$FIXTURE_APP"
print -r -- "$FIXTURE_APP"
