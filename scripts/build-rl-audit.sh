#!/bin/zsh
# Build a separately signed diagnostic app without replacing the installation.
set -euo pipefail
ROOT="${0:A:h:h}"
APP="/Applications/AgentTrainer.app"
SIGN_IDENTITY="${CODE_SIGN_IDENTITY:-}"
SIGN_KEYCHAIN="${CODE_SIGN_KEYCHAIN:-}"
source "$ROOT/scripts/signing.sh"
configure_signing
DESTINATION="${1:?Pass the diagnostic app destination}"
[[ "$DESTINATION" == "$ROOT"/work/rl-validation/*/*.app ]] || exit 1
cd "$ROOT"
swift build --build-system native -c release
/usr/bin/ditto "$APP" "$DESTINATION"
/bin/cp .build/release/AgentTrainer "$DESTINATION/Contents/MacOS/AgentTrainer"
KEYCHAIN_ARGUMENTS=()
if [[ -n "$SIGN_KEYCHAIN" ]]; then KEYCHAIN_ARGUMENTS=(--keychain "$SIGN_KEYCHAIN"); fi
codesign --force --options runtime --timestamp=none "${KEYCHAIN_ARGUMENTS[@]}" --sign "$SIGN_IDENTITY" "$DESTINATION"
verify_permission_identity "$DESTINATION"
