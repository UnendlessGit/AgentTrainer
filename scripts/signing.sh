#!/bin/zsh
# Sourced by build.sh. Machine-local pins survive clean builds and source updates.

signing_error() { print -u2 -r -- "$*"; return 1; }

signing_requirement() {
  codesign -d -r- "$1" 2>&1 | sed -n 's/^designated => //p'
}

configure_signing() {
  local identities line fingerprint name matches=() current_authority=""
  local bundle_id installed_requirement="" pinned_identity=""
  bundle_id="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$ROOT/Resources/Info.plist")"
  SIGN_STATE_DIR="${AGENTTRAINER_SIGNING_STATE_DIR:-$HOME/Library/Application Support/AgentTrainer/BuildSigning}"
  SIGN_STATE="$SIGN_STATE_DIR/$bundle_id.signing"
  EXPECTED_REQUIREMENT=""
  if [[ -f "$SIGN_STATE" ]]; then
    pinned_identity="$(head -n 1 "$SIGN_STATE")"
    EXPECTED_REQUIREMENT="$(tail -n +2 "$SIGN_STATE")"
    [[ "$pinned_identity" =~ '^[[:xdigit:]]{40}$' && -n "$EXPECTED_REQUIREMENT" ]] || {
      signing_error "Invalid signing pin: $SIGN_STATE"; return 1
    }
  fi
  if [[ -d "$APP" ]]; then
    installed_requirement="$(signing_requirement "$APP")"
    current_authority="$(codesign -d --verbose=4 "$APP" 2>&1 | sed -n 's/^Authority=//p' | head -n 1)"
    if [[ -n "$current_authority" ]]; then
      codesign --verify --deep --strict "$APP" || return 1
      if [[ -n "$EXPECTED_REQUIREMENT" && "$installed_requirement" != "$EXPECTED_REQUIREMENT" ]]; then
        signing_error "Installed app differs from the saved permission identity. Refusing to replace it."; return 1
      fi
      EXPECTED_REQUIREMENT="$installed_requirement"
    fi
  fi

  local keychain_scope=()
  [[ -z "$SIGN_KEYCHAIN" ]] || keychain_scope=("$SIGN_KEYCHAIN")
  identities="$(security find-identity -v -p codesigning "${keychain_scope[@]}" 2>/dev/null || true)"
  # security sometimes lists rejected certificates even with -v. Never select
  # those, and always pass the exact SHA-1 fingerprint to codesign, not a name.
  for line in "${(@f)identities}"; do
    [[ "$line" == *CSSMERR_* ]] && continue
    fingerprint="$(print -r -- "$line" | awk '/^[[:space:]]*[0-9]+\)/ {print $2}')"
    [[ "$fingerprint" =~ '^[[:xdigit:]]{40}$' ]] || continue
    name="$(print -r -- "$line" | cut -d '"' -f 2)"
    if [[ -n "$SIGN_IDENTITY" ]]; then
      [[ "$SIGN_IDENTITY:u" == "$fingerprint" || "$SIGN_IDENTITY" == "$name" ]] || continue
    elif [[ -n "$pinned_identity" ]]; then
      [[ "$fingerprint" == "$pinned_identity" ]] || continue
    elif [[ -n "$current_authority" ]]; then
      [[ "$name" == "$current_authority" ]] || continue
    fi
    matches+=("$fingerprint")
  done
  matches=("${(@u)matches}")
  if [[ "$SIGN_IDENTITY" == "-" || ( ${#matches} == 0 && "${ALLOW_ADHOC_SIGNING:-0}" == 1 ) ]]; then
    if [[ -n "$EXPECTED_REQUIREMENT" ]]; then
      signing_error "Ad-hoc signing would discard the saved permission identity. Refusing to continue."; return 1
    fi
    SIGN_IDENTITY="-"
    return 0
  fi
  if (( ${#matches} != 1 )); then
    signing_error "Expected one usable signing certificate; found ${#matches}. Set CODE_SIGN_IDENTITY to an exact certificate fingerprint. The build will not silently switch identities. Saved pin: $SIGN_STATE"
    return 1
  fi
  SIGN_IDENTITY="$matches[1]"

  # Locate by fingerprint too, so duplicate certificate names cannot select a
  # different keychain. Do not try empty passwords on the user's login keychain.
  if [[ -z "$SIGN_KEYCHAIN" ]]; then
    local candidate
    for candidate in "${(@f)$(security list-keychains -d user | sed -E 's/^[[:space:]]*"//; s/"[[:space:]]*$//')}"; do
      if security find-identity -v -p codesigning "$candidate" 2>/dev/null | grep -Fq "$SIGN_IDENTITY"; then
        SIGN_KEYCHAIN="$candidate"
        break
      fi
    done
  fi
}

verify_permission_identity() {
  local bundle="$1" requirement
  codesign --verify --deep --strict --verbose=2 "$bundle" || return 1
  requirement="$(signing_requirement "$bundle")"
  [[ -n "$requirement" ]] || { signing_error "Signed app has no designated requirement."; return 1; }
  if [[ -n "$EXPECTED_REQUIREMENT" && "$requirement" != "$EXPECTED_REQUIREMENT" ]]; then
    signing_error "Signing changed the app's permission identity. Installed app was not replaced."; return 1
  fi
  if [[ "$SIGN_IDENTITY" != "-" ]]; then
    EXPECTED_REQUIREMENT="$requirement"
  fi
}

save_signing_pin() {
  [[ "$SIGN_IDENTITY" != "-" ]] || return 0
  mkdir -p "$SIGN_STATE_DIR"
  local temporary
  temporary="$(mktemp "$SIGN_STATE_DIR/.signing.XXXXXX")" || return 1
  { print -r -- "$SIGN_IDENTITY"; print -r -- "$EXPECTED_REQUIREMENT"; } > "$temporary"
  chmod 600 "$temporary"
  mv -f "$temporary" "$SIGN_STATE"
}
