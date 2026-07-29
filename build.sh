#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h}"
CONFIGURATION="${CONFIGURATION:-release}"
BUILD_ROOT="$ROOT/.build/xcode"
APP="${AGENTTRAINER_APP_PATH:-/Applications/AgentTrainer.app}"
VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$ROOT/Resources/Info.plist")"
EXPECTED_BUNDLE_ID="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$ROOT/Resources/Info.plist")"
DMG="$ROOT/outputs/AgentTrainer-$VERSION.dmg"
SIGN_IDENTITY="${CODE_SIGN_IDENTITY:-}"
SIGN_KEYCHAIN="${CODE_SIGN_KEYCHAIN:-}"

if [[ "$APP" != /* || "$APP" != *.app || -L "$APP" ]]; then
  echo "AGENTTRAINER_APP_PATH must be an absolute, non-symlink .app path." >&2
  exit 1
fi
if [[ -e "$APP" && ! -d "$APP" ]]; then
  echo "$APP exists but is not an application bundle directory." >&2
  exit 1
fi
if [[ -d "$APP" ]]; then
  EXISTING_BUNDLE_ID="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$APP/Contents/Info.plist" 2>/dev/null || true)"
  if [[ "$EXISTING_BUNDLE_ID" != "$EXPECTED_BUNDLE_ID" ]]; then
    echo "Refusing to replace $APP because its bundle identifier is not $EXPECTED_BUNDLE_ID." >&2
    exit 1
  fi
  if [[ ! -w "$APP/Contents" ]]; then
    echo "$APP is not writable. Fix its ownership or choose another path with AGENTTRAINER_APP_PATH." >&2
    exit 1
  fi
fi
APP_PARENT="${APP:h}"
mkdir -p "$APP_PARENT"
if [[ ! -w "$APP_PARENT" ]]; then
  echo "$APP_PARENT is not writable. Choose a writable .app path with AGENTTRAINER_APP_PATH." >&2
  exit 1
fi

# Ad-hoc signing makes the designated requirement equal the binary's CDHash,
# which changes on every build and forces macOS TCC permissions to be granted
# again. Prefer the installed app's current signer; on a fresh install, use the
# only available identity when the choice is unambiguous.
IDENTITY_SEARCH_ARGUMENTS=()
if [[ -n "$SIGN_KEYCHAIN" ]]; then
  IDENTITY_SEARCH_ARGUMENTS=("$SIGN_KEYCHAIN")
fi
IDENTITY_LIST="$(security find-identity -v -p codesigning "${IDENTITY_SEARCH_ARGUMENTS[@]}" 2>/dev/null || true)"
if [[ -z "$SIGN_IDENTITY" && -d "$APP" ]]; then
  EXISTING_AUTHORITY="$(codesign -dvv "$APP" 2>&1 | sed -n 's/^Authority=//p' | head -1)"
  if [[ -n "$EXISTING_AUTHORITY" ]] && grep -Fq "\"$EXISTING_AUTHORITY\"" <<<"$IDENTITY_LIST"; then
    SIGN_IDENTITY="$EXISTING_AUTHORITY"
  fi
fi
if [[ -z "$SIGN_IDENTITY" ]]; then
  AVAILABLE_IDENTITIES=("${(@f)$(sed -n 's/.*"\(.*\)"/\1/p' <<<"$IDENTITY_LIST")}")
  if (( ${#AVAILABLE_IDENTITIES[@]} == 1 )) && [[ -n "${AVAILABLE_IDENTITIES[1]}" ]]; then
    SIGN_IDENTITY="${AVAILABLE_IDENTITIES[1]}"
  fi
fi
if [[ -z "$SIGN_IDENTITY" ]]; then
  if [[ "${ALLOW_ADHOC_SIGNING:-0}" == "1" ]]; then
    SIGN_IDENTITY="-"
  else
    echo "No unambiguous stable code-signing identity was found. Set CODE_SIGN_IDENTITY (and optionally CODE_SIGN_KEYCHAIN), or use ALLOW_ADHOC_SIGNING=1 for a disposable build that will require permissions again." >&2
    exit 1
  fi
fi
if [[ "$SIGN_IDENTITY" != "-" && -z "$SIGN_KEYCHAIN" ]]; then
  SIGN_KEYCHAIN="$(security find-certificate -a -c "$SIGN_IDENTITY" -Z 2>/dev/null | sed -n 's/^keychain: "\(.*\)"/\1/p' | head -1)"
fi
SIGN_KEYCHAIN_ARGUMENTS=()
if [[ -n "$SIGN_KEYCHAIN" ]]; then
  # Custom local build keychains commonly use an empty password. A nonempty
  # keychain should be unlocked by the developer before running the build.
  security unlock-keychain -p "" "$SIGN_KEYCHAIN" >/dev/null 2>&1 || true
  SIGN_KEYCHAIN_ARGUMENTS=(--keychain "$SIGN_KEYCHAIN")
fi

mkdir -p "$ROOT/outputs"
xcodebuild \
  -skipPackagePluginValidation \
  -scheme AgentTrainer \
  -destination 'platform=macOS,arch=arm64' \
  -configuration "$CONFIGURATION" \
  -derivedDataPath "$BUILD_ROOT" \
  CLANG_ENABLE_CODE_COVERAGE=NO \
  GCC_GENERATE_TEST_COVERAGE_FILES=NO \
  GCC_INSTRUMENT_PROGRAM_FLOW_ARCS=NO \
  build

# Xcode is required for MLX's Metal shader plugin, but its auto-generated
# package scheme enables coverage instrumentation even for Release. Build the
# shipped executable with SwiftPM so the hot inference/training paths contain
# no profile counters; both builds consume the exact same Package.resolved pin.
swift build -c "$CONFIGURATION"
BIN="$ROOT/.build/$CONFIGURATION/AgentTrainer"
if [[ ! -x "$BIN" ]]; then
  BIN="$(find "$ROOT/.build" -type f -path "*/$CONFIGURATION/AgentTrainer" -perm +111 ! -path '*/xcode/*' | head -1)"
fi
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "AgentTrainer executable was not produced by Xcode." >&2
  exit 1
fi
MINIMUM_OS="$(xcrun vtool -show-build "$BIN" | awk '/minos/{print $2; exit}')"
if [[ "$MINIMUM_OS" != "15.0" ]]; then
  echo "Release binary targets macOS $MINIMUM_OS; expected the Sequoia-compatible 15.0 target." >&2
  exit 1
fi
if strings "$BIN" | grep -Fq 'default.profraw'; then
  echo "Release executable still contains code-coverage instrumentation." >&2
  exit 1
fi
PLIST_MINIMUM_OS="$(/usr/libexec/PlistBuddy -c 'Print :LSMinimumSystemVersion' "$ROOT/Resources/Info.plist")"
if [[ "$PLIST_MINIMUM_OS" != "15.0" ]]; then
  echo "Info.plist requires macOS $PLIST_MINIMUM_OS; expected 15.0." >&2
  exit 1
fi

# Preserve the authorized bundle directory and path when updating an existing
# app. TCC permission identity depends on the bundle identifier, signer, and the
# copy the user actually launches. Bundle contents are rebuilt in place, then
# the complete bundle is signed again below.
for RUNNING_PID in $(pgrep -x AgentTrainer 2>/dev/null || true); do
  RUNNING_EXECUTABLE="$(ps -p "$RUNNING_PID" -o comm= | xargs)"
  if [[ "$RUNNING_EXECUTABLE" == "$APP/Contents/MacOS/AgentTrainer" ]]; then
    echo "AgentTrainer is running from $APP. Stop the agent and quit the app before updating its signed bundle." >&2
    exit 1
  fi
done
APP_WAS_PRESENT=0
if [[ -d "$APP" ]]; then APP_WAS_PRESENT=1; fi
rm -rf "$APP/Contents/MacOS" "$APP/Contents/Resources" "$APP/Contents/_CodeSignature"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
/bin/cp "$BIN" "$APP/Contents/MacOS/AgentTrainer"
/bin/chmod 755 "$APP/Contents/MacOS/AgentTrainer"
/bin/cp "$ROOT/Resources/Info.plist" "$APP/Contents/Info.plist"
/bin/cp "$ROOT/Resources/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"

METALLIB="$BUILD_ROOT/Build/Products/$CONFIGURATION/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
if [[ ! -f "$METALLIB" ]]; then
  METALLIB="$(find "$BUILD_ROOT/Build/Products" -type f \( -name 'default.metallib' -o -name 'mlx.metallib' \) | head -1)"
fi
if [[ -z "$METALLIB" || ! -f "$METALLIB" ]]; then
  echo "MLX Metal library was not produced by Xcode." >&2
  exit 1
fi
/bin/cp "$METALLIB" "$APP/Contents/MacOS/mlx.metallib"
/bin/cp "$METALLIB" "$APP/Contents/Resources/mlx.metallib"
codesign --force --deep --options runtime --timestamp=none "${SIGN_KEYCHAIN_ARGUMENTS[@]}" --sign "$SIGN_IDENTITY" "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"

rm -f "$ROOT/outputs/README.md"
SOURCE_ARCHIVE="$ROOT/outputs/AgentTrainer-Source.zip"
rm -f "$SOURCE_ARCHIVE"
cd "$ROOT"
/usr/bin/zip -qry "$SOURCE_ARCHIVE" Package.swift Package.resolved Sources Tests Resources WindowsRecorder .github RECORDING_FORMAT.md build.sh test.sh README.md -x '*.DS_Store' '*/bin/*' '*/obj/*' '*/artifacts/*'

DMG_STAGE="$BUILD_ROOT/DMG"
DMG_APP="$DMG_STAGE/AgentTrainer.app"
rm -rf "$DMG_STAGE" "$DMG"
mkdir -p "$DMG_STAGE"
cp -R "$APP" "$DMG_APP"

# Ship one size-optimized disk image. The duplicate Resources metallib is not
# needed because MLX loads the executable-adjacent copy. Stripping link/debug
# symbols changes neither app behavior nor model quality; re-sign the staged
# copy afterward so its designated requirement remains stable.
rm -rf "$DMG_APP/Contents/_CodeSignature"
rm -f "$DMG_APP/Contents/Resources/mlx.metallib"
xcrun strip -x -S "$DMG_APP/Contents/MacOS/AgentTrainer"
codesign --force --deep --options runtime --timestamp=none "${SIGN_KEYCHAIN_ARGUMENTS[@]}" --sign "$SIGN_IDENTITY" "$DMG_APP"
codesign --verify --deep --strict --verbose=2 "$DMG_APP"
ln -s /Applications "$DMG_STAGE/Applications"
hdiutil create -quiet -volname "AgentTrainer $VERSION" -srcfolder "$DMG_STAGE" -ov -format UDZO "$DMG"
if [[ "$SIGN_IDENTITY" != "-" ]]; then
  codesign --force --timestamp=none "${SIGN_KEYCHAIN_ARGUMENTS[@]}" --sign "$SIGN_IDENTITY" "$DMG"
fi
hdiutil verify -quiet "$DMG"
DMG_BYTES=$(stat -f %z "$DMG")
if (( DMG_BYTES >= 10000000 )); then
  echo "DMG is $DMG_BYTES bytes; the required limit is below 10,000,000 bytes." >&2
  exit 1
fi

# Keep the output directory unambiguous: a successful build supersedes older
# versioned DMGs and historical zipped-DMG artifacts.
for old in "$ROOT"/outputs/AgentTrainer-*.dmg(N) "$ROOT"/outputs/AgentTrainer-*.dmg.zip(N); do
  if [[ "$old" != "$DMG" ]]; then rm -f "$old"; fi
done

CHECKSUM_INPUTS=("${DMG:t}" "${SOURCE_ARCHIVE:t}")
(cd "$ROOT/outputs" && shasum -a 256 "${CHECKSUM_INPUTS[@]}" > SHA256SUMS.txt)

echo "$APP"
echo "$DMG ($DMG_BYTES bytes)"
echo "Bundle update: $([[ "$APP_WAS_PRESENT" == "1" ]] && echo 'in place' || echo 'created')"
echo "Signed with: $SIGN_IDENTITY"
