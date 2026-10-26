#!/bin/zsh
set -euo pipefail
ROOT="${0:A:h:h}"
source "$ROOT/scripts/signing.sh"
FIXTURE="$(mktemp -d "${TMPDIR:-/tmp}/agenttrainer-signing-tests.XXXXXX")"
trap 'rm -rf "$FIXTURE"' EXIT
export AGENTTRAINER_SIGNING_STATE_DIR="$FIXTURE/pins"
GOOD=1111111111111111111111111111111111111111
REVOKED=2222222222222222222222222222222222222222
OTHER=3333333333333333333333333333333333333333
ORIGINAL_REQUIREMENT='identifier "local.agenttrainer.mac" and anchor apple generic'

# Exercise selection/continuity without reading or changing real Keychains.
security() {
  if [[ "$1" == find-identity ]]; then print -r -- "$MOCK_IDENTITIES"
  elif [[ "$1" == list-keychains ]]; then print -r -- '"/mock/login.keychain-db"'
  else return 1
  fi
}
codesign() {
  if [[ "$1" == --verify ]]; then return "$MOCK_VERIFY_STATUS"; fi
  if [[ "$*" == *--verbose=4* ]]; then
    print -u2 -r -- "Authority=Development Certificate"
  else
    print -u2 -r -- "designated => $MOCK_REQUIREMENT"
  fi
}
reset_fixture() {
  rm -rf "$AGENTTRAINER_SIGNING_STATE_DIR"
  APP="$FIXTURE/AgentTrainer.app"
  mkdir -p "$APP"
  SIGN_IDENTITY=""; SIGN_KEYCHAIN=""
  MOCK_REQUIREMENT="$ORIGINAL_REQUIREMENT"
  MOCK_VERIFY_STATUS=0
  ALLOW_ADHOC_SIGNING=0
  MOCK_IDENTITIES="  1) $REVOKED \"Development Certificate\" (CSSMERR_TP_CERT_REVOKED)
  2) $GOOD \"Development Certificate\""
}
expect_failure() {
  if ( "$@" ) >"$FIXTURE/failure.log" 2>&1; then
    print -u2 -r -- "Expected failure: $*"; exit 1
  fi
}

reset_fixture
configure_signing
[[ "$SIGN_IDENTITY" == "$GOOD" ]]
verify_permission_identity "$APP"
save_signing_pin
[[ "$(head -n 1 "$SIGN_STATE")" == "$GOOD" ]]
# A clean build without an installed app must still use the persistent pin.
APP="$FIXTURE/absent.app"
SIGN_IDENTITY=""; SIGN_KEYCHAIN=""
configure_signing
[[ "$SIGN_IDENTITY" == "$GOOD" && "$EXPECTED_REQUIREMENT" == "$ORIGINAL_REQUIREMENT" ]]
MOCK_IDENTITIES="  1) $OTHER \"Unrelated Certificate\""
SIGN_IDENTITY=""
expect_failure configure_signing
ALLOW_ADHOC_SIGNING=1
expect_failure configure_signing
MOCK_REQUIREMENT='identifier "different.app" and anchor apple generic'
expect_failure verify_permission_identity "$APP"

reset_fixture
MOCK_IDENTITIES+="
  3) $OTHER \"Development Certificate\""
expect_failure configure_signing
SIGN_IDENTITY="$GOOD"
configure_signing
[[ "$SIGN_IDENTITY" == "$GOOD" ]]
MOCK_VERIFY_STATUS=1
expect_failure verify_permission_identity "$APP"

reset_fixture
configure_signing
verify_permission_identity "$APP"
save_signing_pin
MOCK_REQUIREMENT='identifier "replaced.app"'
SIGN_IDENTITY=""
expect_failure configure_signing

reset_fixture
rm -rf "$APP"
MOCK_IDENTITIES=""
expect_failure configure_signing
ALLOW_ADHOC_SIGNING=1
expect_failure configure_signing
SIGN_IDENTITY="-"
expect_failure configure_signing
[[ ! -e "$SIGN_STATE" ]]
print -r -- 'Signing regression checks passed.'
