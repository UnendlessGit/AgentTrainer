# AgentTrainer run-output controls and in-place update guide

This is the durable handoff for the Run-tab cursor/keyboard permissions and for updating an already-authorized macOS app without unnecessarily losing Screen Recording, Input Monitoring, or Accessibility grants. Read this together with `DEVELOPMENT_GUIDE.md` before changing runtime input behavior or replacing the installed app.

## What was added

The Run tab's **Runtime output firewall** has two independent, persisted switches:

- **Allow cursor movement** controls absolute cursor moves and Game Camera/raw-delta moves.
- **Allow keyboard** controls normal keys and Shift, Control, Option, and Command modifiers.

Both default to enabled for backward compatibility. They are run-only permissions, not training settings:

- They do not change `ActionChannels`, model architecture, weights, checkpoints, dataset cache identity, or the learned-brain contract.
- They do not change the per-profile `ActionRestrictions` key/button blacklist.
- They can be changed before startup, while the model is loading, or during an active run.
- Disabling cursor movement deliberately leaves mouse buttons and scrolling available.
- Disabling keyboard immediately releases every AI-held key and modifier, then prevents later action ticks from re-pressing them.

## Implementation map

- `Core/Domain.swift`: `RuntimeOutputPermissions`, with `cursorMovement` and `keyboard` booleans.
- `App/AppModel.swift`: publishes `runtimeOutputPermissions`, persists it in `AgentTrainer.WorkflowSettings`, sends live changes to the current `AgentRuntime`, and passes the current value into startup.
- `UI/Pages.swift`: exposes both toggles in Run → Runtime output firewall. The toggles intentionally remain enabled while the AI is running.
- `Agent/AgentRuntime.swift`: stores the current permission value and orders startup enablement and live updates with its runtime lock. This closes the startup race where a toggle changed during model loading could otherwise be overwritten by the older value captured at the beginning of `start`.
- `Agent/InputInjector.swift`: owns the final enforcement boundary. Cursor posting checks `cursorMovement`; keyboard and modifier desired-state construction checks `keyboard`.
- `Tests/AgentTrainerTests/DomainTests.swift`: regression coverage for cursor blocking and transactional keyboard release.

`PersistentWorkflowSettings.runtimeOutputPermissions` is optional. Existing `UserDefaults` data therefore decodes normally and falls back to both permissions enabled.

## Runtime safety and concurrency contract

`InputInjector.execute` holds the injector lifecycle lock for its complete decision/posting operation. `updateOutputPermissions` uses the same lock.

When keyboard output changes from enabled to disabled:

1. The update takes the injector lock, so no action tick can post concurrently.
2. The permission is changed first.
3. The held-key and modifier sets are cleared.
4. The lock is released.
5. Key-up/flags-changed events are posted for the captured held set.
6. The HUD receives a state with no active AI keys.

A later action tick sees keyboard output disabled and cannot re-press those keys. A simultaneous full agent stop may produce an additional harmless key-up, but cannot produce a late key-down.

When cursor movement changes from enabled to disabled after Game Camera output was used, the injector re-associates the system cursor and posts a tagged zero-delta mouse move. This prevents a game from retaining the last raw delta. Absolute and relative movement are then skipped, while button and scroll processing continues.

Runtime predictions contain two different semantics. Keyboard and mouse-button outputs are held state and may remain active across action ticks. Game Camera and scroll outputs are additive and are consumed only once for each new inference result; stale timer ticks write zero transient values into recurrent history instead of replaying the last movement. Game Camera always emits `.mouseMoved` independently of held buttons, and normal zero-delta predictions are suppressed. The explicit tagged zero-delta event used during disable/stop remains intentional.

Policy v4 also has a freshness deadline. Once at least one prediction has been published, an action tick stops the complete runtime if that prediction is older than `max(350 ms, 3 / Perception FPS)`. Stop then drains inference/action queues and releases held state. Runtime vision, timing, history length, channels, cursor visibility, and Auto mouse mode come from the selected immutable version; mutable profile or Record-tab edits may not change a running brain's contract. Runtime startup is revision-tokened so Stop during weight loading cannot later install an orphan model or input injector.

Do not move these checks only into SwiftUI or `AppModel`. The injector must remain the final authority because it is the component that serializes and posts OS events.

## Regression checks

Run the complete test suite:

```sh
./test.sh
```

The feature-specific tests are:

```sh
swift test -c debug --filter 'DomainTests/testRuntimeCursorPermissionBlocksMovementWithoutBlockingMouseButtons|DomainTests/testDisablingRuntimeKeyboardImmediatelyReleasesKeysAndPreventsRepress|DomainTests/testRuntimePredictionLatchConsumesTransientOutputsOnce|DomainTests/testGameCameraDoesNotReplayStaleOrZeroDeltasAndNeverUsesDragEvents'
```

The complete current suite must pass; do not preserve a hard-coded historical test count in release decisions.

Manual checks with a disposable target are still required for a release:

1. Start with both switches enabled and confirm trained cursor/key output appears in the capture-excluded HUD.
2. Hold an AI-generated key, turn **Allow keyboard** off, and confirm the key releases immediately and remains released.
3. Turn keyboard output back on and confirm normal predictions resume.
4. In Absolute Cursor mode, turn cursor movement off and confirm clicks/scroll still work but pointer position does not change.
5. In Game Camera mode, turn cursor movement off and confirm camera motion stops without an edge/last-delta drift.
6. Stop or panic and confirm every held key/button releases and no input tap or action timer remains active.
7. Slow inference below Action FPS and confirm each predicted Game Camera delta appears once, not once per timer tick.
8. Hold the predicted left button while moving the Game Camera and confirm movement events remain `mouseMoved`, not drag events.
9. Pause or stall inference after one prediction and confirm the freshness watchdog stops the AI and releases every held control.

## macOS permissions and code identity

AgentTrainer uses three independent macOS privacy grants:

- **Screen Recording**: ScreenCaptureKit recording and live AI vision.
- **Input Monitoring**: physical input recording and the human-interruption safety event tap.
- **Accessibility**: synthetic keyboard/mouse output and guarded reenactment.

These grants are controlled by macOS TCC. Updating files in place is not sufficient by itself: the updated app must keep the same effective code identity. For this workspace the stable identity consists of:

- Bundle identifier: `local.agenttrainer.mac`
- Signing identity: `MimicClone Local Code Signing`
- Certificate leaf SHA-1: `51e78a945e47715731ac0eb152c69500a77c63ff`
- Normal authorized bundle path: the exact app bundle the user actually launches

The designated requirement verified before and after the 2026-07-13 in-place update was:

```text
identifier "local.agenttrainer.mac" and certificate leaf = H"51e78a945e47715731ac0eb152c69500a77c63ff"
```

The executable CDHash changes on every build; that is expected. The designated requirement above must remain stable. Ad-hoc signing (`-`) makes identity depend on the changing binary hash and commonly causes macOS to request permissions again.

AgentTrainer 1.8.3 uses native hotkey registration when Carbon is available and compiles an AppKit global/local-monitor fallback otherwise. This does not change the three TCC categories or the designated-requirement rules on current systems. The release baseline supports Sequoia 15, Tahoe 26, and macOS 27; compile with Xcode 27 as a separate release check when that toolchain is available.

Do not reset TCC, delete privacy database entries, change the bundle identifier, switch signers, or move/recreate the authorized app as a troubleshooting shortcut. Those actions can discard working grants. If the stable signing identity is unavailable, stop and report the blocker instead of silently ad-hoc signing.

## In-place update procedure

Use this procedure when the user asks to preserve the existing app and permissions. It incrementally compiles in the existing DerivedData directory, replaces only the executable inside the existing bundle, and re-signs that same bundle.

Set paths for the app the user actually launches:

```sh
ROOT="/Users/endless/Documents/AgentTrainer"
APP="$ROOT/outputs/AgentTrainer.app"
BUILT="$ROOT/.build/xcode/Build/Products/release/AgentTrainer"
KEYCHAIN="$HOME/Documents/MimicClone/.codesign/MimicClone.keychain-db"
IDENTITY="MimicClone Local Code Signing"
```

If the authorized copy lives in `/Applications`, set `APP` to that copy. Updating a different duplicate will not update the process the user launches and may have a separate TCC record.

### 1. Record and verify the existing identity

```sh
codesign -d --verbose=4 "$APP" 2>&1
codesign -d -r- "$APP" 2>&1
security find-identity -v -p codesigning "$KEYCHAIN"
file "$APP/Contents/MacOS/AgentTrainer"
stat -f '%N %z bytes modified %Sm' "$APP/Contents/Info.plist"
```

Require the expected bundle identifier, arm64 executable, stable authority, and certificate requirement before modifying the app.

### 2. Run tests

```sh
cd "$ROOT"
./test.sh
```

### 3. Compile incrementally

For a minimal Swift-only update, use the incremental command below and replace only the executable. The release `build.sh` is now also safe for a complete output release: it preserves an existing `outputs/AgentTrainer.app` bundle directory, refreshes its known executable/plist/icon/Metal contents, re-signs it, and rebuilds exactly one size-optimized DMG below 10,000,000 bytes. It refuses to mutate a running copy. Use the manual path here when unchanged resources should retain their exact timestamps.

Use the existing `.build/xcode` DerivedData directory:

```sh
xcodebuild \
  -skipPackagePluginValidation \
  -scheme AgentTrainer \
  -destination 'platform=macOS,arch=arm64' \
  -configuration release \
  -derivedDataPath "$ROOT/.build/xcode" \
  CLANG_ENABLE_CODE_COVERAGE=NO \
  GCC_GENERATE_TEST_COVERAGE_FILES=NO \
  GCC_INSTRUMENT_PROGRAM_FLOW_ARCS=NO \
  build
```

This reuses the package checkouts, module cache, DerivedData, and unchanged build products. Swift whole-module optimization may still recompile the app target, but the existing `.app` bundle is not deleted or rebuilt.

### 4. Replace only what changed

For a Swift-only change, replace only the executable:

```sh
/bin/cp "$BUILT" "$APP/Contents/MacOS/AgentTrainer"
```

Do not replace `Info.plist`, the app icon, or `mlx.metallib` when they did not change. If MLX or its Metal library genuinely changed, update both metallib copies intentionally before signing and verify launch/inference separately.

### 5. Re-sign the existing bundle with the same identity

Unlock the keychain through the normal local mechanism, then sign:

```sh
codesign \
  --force \
  --deep \
  --options runtime \
  --timestamp=none \
  --keychain "$KEYCHAIN" \
  --sign "$IDENTITY" \
  "$APP"
```

Never substitute `--sign -` for a permission-preserving update.

### 6. Verify the result before launch

```sh
codesign --verify --deep --strict --verbose=2 "$APP"
codesign -d --verbose=4 "$APP" 2>&1
codesign -d -r- "$APP" 2>&1
file "$APP/Contents/MacOS/AgentTrainer"
stat -f '%N %z bytes modified %Sm' \
  "$APP/Contents/MacOS/AgentTrainer" \
  "$APP/Contents/Info.plist"
```

Verification must show:

- `valid on disk`
- `satisfies its Designated Requirement`
- Identifier `local.agenttrainer.mac`
- Authority `MimicClone Local Code Signing`
- The same designated requirement recorded before replacement
- A changed executable timestamp, with unchanged `Info.plist` timestamp for a Swift-only update

The 2026-07-13 update met all of these conditions. The app's `Info.plist` remained untouched while the executable was replaced and re-signed.

### 7. Restart the app once

An already-running process continues executing the old in-memory binary. Quit normally so runtime cleanup releases inputs and stops capture, then reopen the same app path. Do not force-kill an active agent unless the panic/normal stop paths have already released its inputs.

## If macOS asks for permissions again

Before asking the user to re-grant anything, compare:

```sh
codesign -d --verbose=4 "$APP" 2>&1
codesign -d -r- "$APP" 2>&1
/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$APP/Contents/Info.plist"
```

Check whether the user launched a different copy from Downloads, a mounted DMG, `/Applications`, or `outputs`. Also check whether the app was ad-hoc signed or signed by a different certificate. Do not run `tccutil reset` automatically; it removes grants rather than repairing identity.

If identity and path are correct, use AgentTrainer's permission diagnostics and System Settings → Privacy & Security to inspect the three grants. A one-time re-grant is expected only when moving from the historical ad-hoc build to the stable certificate, or when macOS has explicitly removed the grant.
