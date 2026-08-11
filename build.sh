#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h}"
CONFIGURATION="${CONFIGURATION:-release}"
BUILD_ROOT="$ROOT/.build/xcode"
OUTPUTS="$ROOT/outputs"
APP="${APP_PATH:-/Applications/AgentTrainer.app}"
APP="${APP:A}"
APP_PARENT="${APP:h}"
STAGED_APP="$BUILD_ROOT/Install/AgentTrainer.app"
VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$ROOT/Resources/Info.plist")"
DMG="$OUTPUTS/AgentTrainer-$VERSION.dmg"
SIGN_IDENTITY="${CODE_SIGN_IDENTITY:-}"
SIGN_KEYCHAIN="${CODE_SIGN_KEYCHAIN:-}"
SWIFTPM_SANDBOX_FLAGS=()
if [[ "${AGENTTRAINER_DISABLE_SWIFTPM_SANDBOX:-0}" == "1" ]]; then
  SWIFTPM_SANDBOX_FLAGS+=(--disable-sandbox)
fi

if [[ "$APP" != *.app || "$APP_PARENT" == "/" ]]; then
  echo "APP_PATH must name a specific application bundle in a normal folder." >&2
  exit 1
fi
if pgrep -x AgentTrainer >/dev/null 2>&1; then
  echo "AgentTrainer is running. Stop the agent and quit the app before rebuilding it." >&2
  exit 1
fi
if [[ -e "$APP" && ! -d "$APP" ]]; then
  echo "$APP exists but is not an application bundle." >&2
  exit 1
fi
mkdir -p "$APP_PARENT" "$OUTPUTS"
if [[ ! -w "$APP_PARENT" ]]; then
  echo "$APP_PARENT is not writable. Choose a writable Applications folder with APP_PATH." >&2
  exit 1
fi

# Stable signing preserves the app's designated requirement and therefore its
# Screen Recording, Input Monitoring, and Accessibility grants. Prefer the
# identity already used by the installed app; otherwise use the first valid
# identity in the selected keychain search scope.
if [[ -n "$SIGN_KEYCHAIN" ]]; then
  IDENTITIES="$(security find-identity -v -p codesigning "$SIGN_KEYCHAIN" 2>/dev/null || true)"
else
  IDENTITIES="$(security find-identity -v -p codesigning 2>/dev/null || true)"
fi
if [[ -z "$SIGN_IDENTITY" && -d "$APP" ]]; then
  CURRENT_AUTHORITY="$(codesign -d --verbose=4 "$APP" 2>&1 | awk -F= '/^Authority=/{print $2; exit}' || true)"
  if [[ -n "$CURRENT_AUTHORITY" ]] && grep -Fq "\"$CURRENT_AUTHORITY\"" <<< "$IDENTITIES"; then
    SIGN_IDENTITY="$CURRENT_AUTHORITY"
  fi
fi
if [[ -z "$SIGN_IDENTITY" ]]; then
  SIGN_IDENTITY="$(print -r -- "$IDENTITIES" | awk -F'"' '/^[[:space:]]*[0-9]+\)/ { print $2; exit }')"
fi
if [[ -z "$SIGN_IDENTITY" ]]; then
  if [[ "${ALLOW_ADHOC_SIGNING:-0}" == "1" ]]; then
    SIGN_IDENTITY="-"
  else
    echo "No stable code-signing identity was found. Set CODE_SIGN_IDENTITY (and optionally CODE_SIGN_KEYCHAIN), or use ALLOW_ADHOC_SIGNING=1 for a disposable build that will need fresh permissions." >&2
    exit 1
  fi
fi

# `find-identity` can see identities in a locked non-default keychain, while
# codesign cannot use their private keys. Discover the identity's keychain
# without relying on a machine-specific path, then unlock it. Local development
# keychains commonly use an empty password; protected keychains can provide
# CODE_SIGN_KEYCHAIN_PASSWORD.
if [[ "$SIGN_IDENTITY" != "-" && -z "$SIGN_KEYCHAIN" ]]; then
  for candidate in "${(@f)$(security list-keychains -d user | sed -E 's/^[[:space:]]*"//; s/"[[:space:]]*$//')}"; do
    if security find-identity -v -p codesigning "$candidate" 2>/dev/null | grep -Fq "\"$SIGN_IDENTITY\""; then
      SIGN_KEYCHAIN="$candidate"
      break
    fi
  done
fi
SIGN_KEYCHAIN_ARGUMENTS=()
if [[ -n "$SIGN_KEYCHAIN" ]]; then
  security unlock-keychain -p "${CODE_SIGN_KEYCHAIN_PASSWORD:-}" "$SIGN_KEYCHAIN" >/dev/null 2>&1 || true
  SIGN_KEYCHAIN_ARGUMENTS=(--keychain "$SIGN_KEYCHAIN")
fi

sign_app_bundle() {
  local bundle="$1"
  local metallib
  for metallib in "$bundle"/Contents/{MacOS,Resources}/*.metallib(N); do
    codesign --force --options runtime --timestamp=none "${SIGN_KEYCHAIN_ARGUMENTS[@]}" --sign "$SIGN_IDENTITY" "$metallib"
  done
  codesign --force --options runtime --timestamp=none "${SIGN_KEYCHAIN_ARGUMENTS[@]}" --sign "$SIGN_IDENTITY" "$bundle"
}

# Xcode runs MLX's Metal shader plugin. Its output depends on the pinned MLX
# package, not AgentTrainer sources, so preserve a current metallib across app
# rebuilds instead of recompiling the dependency's complete shader library.
METALLIB="$BUILD_ROOT/Build/Products/$CONFIGURATION/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
if [[ ! -f "$METALLIB" || "$ROOT/Package.resolved" -nt "$METALLIB" ]]; then
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
fi

# SwiftPM builds the shipped release executable without Xcode package-scheme
# coverage instrumentation.

swift build "${SWIFTPM_SANDBOX_FLAGS[@]}" -c "$CONFIGURATION"
BIN="$ROOT/.build/$CONFIGURATION/AgentTrainer"
if [[ ! -x "$BIN" ]]; then
  BIN="$(find "$ROOT/.build" -type f -path "*/$CONFIGURATION/AgentTrainer" -perm +111 ! -path '*/xcode/*' | head -1)"
fi
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "SwiftPM did not produce the AgentTrainer executable." >&2
  exit 1
fi

MINIMUM_OS="$(xcrun vtool -show-build "$BIN" | awk '/minos/{print $2; exit}')"
if [[ "$MINIMUM_OS" != "15.0" ]]; then
  echo "Release binary targets macOS $MINIMUM_OS; expected macOS 15.0." >&2
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

if [[ ! -f "$METALLIB" ]]; then
  METALLIB="$(find "$BUILD_ROOT/Build/Products" -type f \( -name 'default.metallib' -o -name 'mlx.metallib' \) | head -1)"
fi
if [[ -z "$METALLIB" || ! -f "$METALLIB" ]]; then
  echo "Xcode did not produce the MLX Metal library." >&2
  exit 1
fi

# AppIcon.icon is the editable Icon Composer source of truth. Compile it with
# Xcode's asset compiler so its Default, Dark, and Mono variants remain native
# bundle assets; flattening to one .icns would lose those appearances.
ICON_DOCUMENT="$ROOT/Resources/AppIcon.icon"
ASSET_CATALOG_OUTPUT="$BUILD_ROOT/AppIconAssets"
ASSET_CATALOG_PLIST="$ASSET_CATALOG_OUTPUT/asset-info.plist"
GENERATED_ICON="$ASSET_CATALOG_OUTPUT/AppIcon.icns"
GENERATED_ASSETS="$ASSET_CATALOG_OUTPUT/Assets.car"
if [[ ! -d "$ICON_DOCUMENT" ]]; then
  echo "AppIcon.icon is unavailable." >&2
  exit 1
fi
mkdir -p "$ASSET_CATALOG_OUTPUT"
# Icon Composer output depends only on AppIcon.icon. Reuse a current compiled
# catalog across source-only rebuilds; actool otherwise starts Xcode's complete
# platform service stack even for this macOS-only package.
if [[ ! -f "$GENERATED_ICON" || ! -f "$GENERATED_ASSETS" ]]; then
  PREVIOUS_ICON="$STAGED_APP/Contents/Resources/AppIcon.icns"
  PREVIOUS_ASSETS="$STAGED_APP/Contents/Resources/Assets.car"
  if [[ -f "$PREVIOUS_ICON" && -f "$PREVIOUS_ASSETS" \
        && -z "$(find "$ICON_DOCUMENT" -type f -newer "$PREVIOUS_ICON" -print -quit)" ]]; then
    /bin/cp "$PREVIOUS_ICON" "$GENERATED_ICON"
    /bin/cp "$PREVIOUS_ASSETS" "$GENERATED_ASSETS"
  fi
fi
if [[ ! -f "$GENERATED_ICON" || ! -f "$GENERATED_ASSETS" \
      || -n "$(find "$ICON_DOCUMENT" -type f -newer "$GENERATED_ICON" -print -quit)" ]]; then
  rm -rf "$ASSET_CATALOG_OUTPUT"
  mkdir -p "$ASSET_CATALOG_OUTPUT"
  actool --compile "$ASSET_CATALOG_OUTPUT" \
    --output-partial-info-plist "$ASSET_CATALOG_PLIST" \
    "$ICON_DOCUMENT" \
    --platform macosx \
    --minimum-deployment-target 15.0 \
    --app-icon AppIcon
fi
if [[ ! -f "$GENERATED_ICON" || ! -f "$GENERATED_ASSETS" ]]; then
  echo "Xcode did not compile the AppIcon assets." >&2
  exit 1
fi

# Assemble and verify a complete bundle before touching the installed copy.
rm -rf "$STAGED_APP"
mkdir -p "$STAGED_APP/Contents/MacOS" "$STAGED_APP/Contents/Resources"
/bin/cp "$BIN" "$STAGED_APP/Contents/MacOS/AgentTrainer"
/bin/chmod 755 "$STAGED_APP/Contents/MacOS/AgentTrainer"
/bin/cp "$ROOT/Resources/Info.plist" "$STAGED_APP/Contents/Info.plist"
/bin/cp "$GENERATED_ICON" "$STAGED_APP/Contents/Resources/AppIcon.icns"
/bin/cp "$GENERATED_ASSETS" "$STAGED_APP/Contents/Resources/Assets.car"
/bin/cp "$METALLIB" "$STAGED_APP/Contents/MacOS/mlx.metallib"
/bin/cp "$METALLIB" "$STAGED_APP/Contents/Resources/mlx.metallib"
sign_app_bundle "$STAGED_APP"
codesign --verify --deep --strict --verbose=2 "$STAGED_APP"

# Copy onto the target volume first, then swap complete bundles. A failed copy,
# move, or signature check restores the previous installation.
APP_WAS_PRESENT=0
if [[ -d "$APP" ]]; then APP_WAS_PRESENT=1; fi
INCOMING="$APP_PARENT/.AgentTrainer Build Incoming.$$.app"
BACKUP="$APP_PARENT/.AgentTrainer Build Backup.$$.app"
rm -rf "$INCOMING" "$BACKUP"
/usr/bin/ditto --rsrc --extattr "$STAGED_APP" "$INCOMING"
codesign --verify --deep --strict --verbose=2 "$INCOMING"

if [[ "$APP_WAS_PRESENT" == "1" ]]; then
  /bin/mv "$APP" "$BACKUP"
  if ! /bin/mv "$INCOMING" "$APP"; then
    if ! /bin/mv "$BACKUP" "$APP"; then
      echo "Installation failed. The previous app remains at $BACKUP and must be restored manually." >&2
    fi
    exit 1
  fi
  if ! codesign --verify --deep --strict --verbose=2 "$APP"; then
    rm -rf "$APP"
    if ! /bin/mv "$BACKUP" "$APP"; then
      echo "Signature verification failed. The previous app remains at $BACKUP and must be restored manually." >&2
    fi
    exit 1
  fi
  if ! rm -rf "$BACKUP"; then
    echo "Warning: the new app is installed, but the hidden backup could not be removed: $BACKUP" >&2
  fi
else
  /bin/mv "$INCOMING" "$APP"
  if ! codesign --verify --deep --strict --verbose=2 "$APP"; then
    rm -rf "$APP"
    exit 1
  fi
fi

# Keep distribution artifacts in the repository's ignored outputs folder.
rm -rf "$OUTPUTS/AgentTrainer.app"
/bin/cp "$ROOT/README.md" "$OUTPUTS/README.md"
SOURCE_ARCHIVE="$OUTPUTS/AgentTrainer-Source.zip"
rm -f "$SOURCE_ARCHIVE"
cd "$ROOT"
/usr/bin/zip -qry "$SOURCE_ARCHIVE" Package.swift Package.resolved Sources Tests Resources build.sh test.sh benchmark.sh README.md DEVELOPMENT_GUIDE.md -x '*.DS_Store'

DMG_STAGE="$BUILD_ROOT/DMG"
DMG_APP="$DMG_STAGE/AgentTrainer.app"
rm -rf "$DMG_STAGE" "$DMG"
mkdir -p "$DMG_STAGE"
/usr/bin/ditto --rsrc --extattr "$APP" "$DMG_APP"

# The executable-adjacent Metal library is the runtime copy. Remove only its
# duplicate Resources copy and link/debug symbols from the DMG, then re-sign.
rm -rf "$DMG_APP/Contents/_CodeSignature"
rm -f "$DMG_APP/Contents/Resources/mlx.metallib"
xcrun strip -x -S "$DMG_APP/Contents/MacOS/AgentTrainer"
sign_app_bundle "$DMG_APP"
codesign --verify --deep --strict --verbose=2 "$DMG_APP"
ln -s /Applications "$DMG_STAGE/Applications"
if [[ "${SKIP_DMG:-0}" == "1" ]]; then
  # Restricted/headless builders may not expose DiskImages.framework's helper
  # service. Preserve the same stripped, signed app as a resource-safe zip so
  # app verification and handoff can still complete without a stale old DMG.
  DISTRIBUTION_ARTIFACT="$OUTPUTS/AgentTrainer-$VERSION.app.zip"
  rm -f "$DMG" "$DISTRIBUTION_ARTIFACT"
  /usr/bin/ditto -c -k --sequesterRsrc --keepParent "$DMG_APP" "$DISTRIBUTION_ARTIFACT"
  DISTRIBUTION_DESCRIPTION="Application archive"
else
  hdiutil create -quiet -volname "AgentTrainer $VERSION" -srcfolder "$DMG_STAGE" -ov -format UDZO "$DMG"
  if [[ "$SIGN_IDENTITY" != "-" ]]; then
    codesign --force --timestamp=none "${SIGN_KEYCHAIN_ARGUMENTS[@]}" --sign "$SIGN_IDENTITY" "$DMG"
  fi
  hdiutil verify -quiet "$DMG"
  DISTRIBUTION_ARTIFACT="$DMG"
  DISTRIBUTION_DESCRIPTION="Disk image"
fi
DISTRIBUTION_BYTES="$(stat -f %z "$DISTRIBUTION_ARTIFACT")"
if (( DISTRIBUTION_BYTES >= 10000000 )); then
  echo "$DISTRIBUTION_DESCRIPTION is $DISTRIBUTION_BYTES bytes; the required limit is below 10,000,000 bytes." >&2
  exit 1
fi

for old in "$OUTPUTS"/AgentTrainer-*.dmg(N) "$OUTPUTS"/AgentTrainer-*.dmg.zip(N) "$OUTPUTS"/AgentTrainer-*.app.zip(N); do
  if [[ "$old" != "$DISTRIBUTION_ARTIFACT" ]]; then rm -f "$old"; fi
done

CHECKSUM_INPUTS=("${DISTRIBUTION_ARTIFACT:t}" "${SOURCE_ARCHIVE:t}")
(cd "$OUTPUTS" && shasum -a 256 "${CHECKSUM_INPUTS[@]}" > SHA256SUMS.txt)

echo "Installed app: $APP"
echo "$DISTRIBUTION_DESCRIPTION: $DISTRIBUTION_ARTIFACT ($DISTRIBUTION_BYTES bytes)"
echo "Installation: $([[ "$APP_WAS_PRESENT" == "1" ]] && echo 'replaced transactionally' || echo 'created')"
echo "Signed with: $SIGN_IDENTITY"
