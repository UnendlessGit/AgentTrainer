#!/bin/zsh
# Verifies that an AgentTrainer build carries the stable signing identity.
# macOS privacy grants (Screen Recording, Accessibility, Input Monitoring) are tied
# to the designated requirement checked here, so it must never drift.
set -euo pipefail
APP=${1:-build/DerivedData/Build/Products/Release/AgentTrainer.app}
TEAM=V5S6J7KC33
fail() { echo "✗ $1" >&2; exit 1; }

codesign --verify --deep --strict "$APP" || fail "codesign verification failed for $APP"
check() {
  local path=$1 ident=$2
  local dr; dr=$(codesign -d -r- "$path" 2>&1 | grep '^designated')
  [[ $dr == *"identifier \"$ident\""* ]] || fail "$path: identifier is not $ident ($dr)"
  [[ $dr == *"leaf[subject.OU] = $TEAM"* ]] || fail "$path: requirement not pinned to team $TEAM ($dr)"
  local info; info=$(codesign -dvv "$path" 2>&1)
  [[ $info == *"TeamIdentifier=$TEAM"* ]] || fail "$path: wrong team"
  [[ $info == *"flags=0x10000(runtime)"* ]] || fail "$path: hardened runtime missing"
  [[ $info != *"Signature=adhoc"* ]] || fail "$path: ad-hoc signed"
  echo "✓ $ident  $(echo $info | grep -m1 '^Authority=')"
}
check "$APP" com.unendless.AgentTrainer
check "$APP/Contents/XPCServices/AgentTrainerEngine.xpc" com.unendless.AgentTrainer.Engine
