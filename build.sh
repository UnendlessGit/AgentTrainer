#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h}"
CONFIGURATION="${CONFIGURATION:-release}"
BUILD_ROOT="$ROOT/.build/xcode"
APP="$ROOT/outputs/AgentTrainer.app"
VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$ROOT/Resources/Info.plist")"
DMG="$ROOT/outputs/AgentTrainer-$VERSION.dmg"
SIGN_IDENTITY="${CODE_SIGN_IDENTITY:-}"
SIGN_KEYCHAIN="${CODE_SIGN_KEYCHAIN:-}"

# Ad-hoc signing makes the designated requirement equal the binary's CDHash,
# which changes on every build and forces macOS TCC permissions to be granted
# again. Prefer this Mac's long-lived local identity so Screen Recording, Input
# Monitoring, and Accessibility see rebuilt bundles as the same application.
LOCAL_SIGN_NAME="MimicClone Local Code Signing"
LOCAL_SIGN_KEYCHAIN="$HOME/Documents/MimicClone/.codesign/MimicClone.keychain-db"
if [[ -z "$SIGN_IDENTITY" && -f "$LOCAL_SIGN_KEYCHAIN" ]]; then
  security unlock-keychain -p "" "$LOCAL_SIGN_KEYCHAIN" >/dev/null 2>&1 || true
  if security find-identity -v -p codesigning "$LOCAL_SIGN_KEYCHAIN" 2>/dev/null | grep -Fq "$LOCAL_SIGN_NAME"; then
    SIGN_IDENTITY="$LOCAL_SIGN_NAME"
    SIGN_KEYCHAIN="$LOCAL_SIGN_KEYCHAIN"
  fi
fi
if [[ -z "$SIGN_IDENTITY" ]]; then
  if [[ "${ALLOW_ADHOC_SIGNING:-0}" == "1" ]]; then
    SIGN_IDENTITY="-"
  else
    echo "No stable code-signing identity was found. Set CODE_SIGN_IDENTITY (and optionally CODE_SIGN_KEYCHAIN), or use ALLOW_ADHOC_SIGNING=1 for a disposable build that will require permissions again." >&2
    exit 1
  fi
fi
SIGN_KEYCHAIN_ARGUMENTS=()
if [[ -n "$SIGN_KEYCHAIN" ]]; then
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
# copy the user actually launches. Known bundle contents are replaced in place,
# then the complete bundle is signed again below.
if pgrep -x AgentTrainer >/dev/null 2>&1; then
  echo "AgentTrainer is running. Stop the agent and quit the app before updating its signed bundle." >&2
  exit 1
fi
APP_WAS_PRESENT=0
if [[ -d "$APP" ]]; then APP_WAS_PRESENT=1; fi
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
rm -rf "$APP/Contents/_CodeSignature"
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

cp "$ROOT/README.md" "$ROOT/outputs/README.md"
SOURCE_ARCHIVE="$ROOT/outputs/AgentTrainer-Source.zip"
rm -f "$SOURCE_ARCHIVE"
cd "$ROOT"
/usr/bin/zip -qry "$SOURCE_ARCHIVE" Package.swift Package.resolved Sources Tests Resources build.sh test.sh README.md DEVELOPMENT_GUIDE.md IN_PLACE_UPDATE_GUIDE.md -x '*.DS_Store'

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

CHECKSUM_INPUTS=("${DMG:t}" "AgentTrainer.app/Contents/MacOS/AgentTrainer" "AgentTrainer-Source.zip")
(cd "$ROOT/outputs" && shasum -a 256 "${CHECKSUM_INPUTS[@]}" > SHA256SUMS.txt)

echo "$APP"
echo "$DMG ($DMG_BYTES bytes)"
echo "Bundle update: $([[ "$APP_WAS_PRESENT" == "1" ]] && echo 'in place' || echo 'created')"
echo "Signed with: $SIGN_IDENTITY"
