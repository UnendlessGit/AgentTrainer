# AgentTrainer

AgentTrainer is a local-first macOS app for recording screen-and-input
demonstrations, training an imitation-learning policy with MLX, and running the
trained policy on the same Mac. Version 1.9.4.1 targets Apple silicon and macOS
15 or newer.

Video, input streams, caches, profiles, checkpoints, and model weights remain
local. The only normal network request is the signed GitHub Releases update
check.

## What the app does

1. **Record** a display, window, or region together with synchronized keyboard,
   pointer, button, modifier, and scroll input.
2. **Inspect** recordings, organize them in folders, trim their usable ranges,
   and optionally reenact recorded input behind an explicit safety toggle.
3. **Configure** an AI profile: vision size, visual memory, architecture,
   control channels, restrictions, and selected recordings.
4. **Train** locally with MLX. Packed dataset caches decode each perception
   frame once and reuse it across training runs and memory settings.
5. **Run** a saved brain with independent output permissions, focus/staleness
   guards, human-input stopping, and a global panic shortcut.

Policy v9 is perception-first: live decisions use current and remembered visual
frames, never demonstrated previous actions. The runtime retains held controls
in the output injector instead of feeding them back into the model.

## Requirements

- Apple-silicon Mac.
- macOS 15.0 or newer.
- Xcode with command-line tools for building.
- Screen Recording and Input Monitoring permission for recording.
- Accessibility permission only when reenacting input or running an AI with
  output enabled.

AgentTrainer asks for permissions when needed. After changing a permission,
macOS may require the app to be reopened.

## Build and test

Run the complete test suite:

```bash
./test.sh
```

Build the release app and distribution artifacts:

```bash
./build.sh
```

By default, `build.sh` creates or updates the real application at:

```text
/Applications/AgentTrainer.app
```

It keeps that bundle path, verifies the bundle identifier before replacing
anything, rebuilds its contents, signs it, and produces these release artifacts
under `outputs/`:

```text
AgentTrainer-<version>.dmg
AgentTrainer-Source.zip
SHA256SUMS.txt
```

The script first reuses the installed app's signing identity. For a new install
it automatically uses the only available code-signing identity; if the choice
is ambiguous, set one explicitly:

```bash
CODE_SIGN_IDENTITY="Developer ID Application: Example" ./build.sh
```

Use `CODE_SIGN_KEYCHAIN` when the identity is in a non-default keychain.
`ALLOW_ADHOC_SIGNING=1` is intended only for disposable builds because every
ad-hoc rebuild can require new privacy grants. A temporary app path can be used
without changing the default installation:

```bash
AGENTTRAINER_APP_PATH=/tmp/AgentTrainer.app ALLOW_ADHOC_SIGNING=1 ./build.sh
```

Quit AgentTrainer before running the release build. The script refuses to
replace a running app.

## Storage

The default workspace is:

```text
~/Library/Application Support/AgentTrainer/
├── Recordings/
├── Caches/
├── Profiles/
└── recording-folders.json
```

Training data and models can be relocated independently from Settings. A move
uses copy, verification, and an atomic location switch; the source is removed
only after the new library is usable. Never edit a live workspace by hand.

Each published recording is a self-contained `.atrrecord` directory. The
manifest is written last, so interrupted recordings are not exposed as library
items. See [RECORDING_FORMAT.md](RECORDING_FORMAT.md) for the portable format.

## Safety and lifecycle rules

- Recording opens its event clock on the first usable encoded video frame.
- An unexpected screen-capture or input-monitor stop ends the active operation.
- Recording-library mutations are blocked while recording, importing,
  training, running, replaying, or relocating storage.
- Profile brains cannot be reset, deleted, duplicated, or switched while that
  profile is training or running.
- The AI runtime disables and releases every held input before waiting for
  capture or inference teardown.
- Run-time cursor and keyboard permissions are independent and apply
  immediately.
- The panic shortcut stops recording, training, replay, and runtime activity.
- Imports validate paths, links, filenames, manifest bounds, event bytes,
  timing, video metadata, and a decodable frame before publishing anything.

## Architecture

The executable is a Swift Package with one test target:

```text
Sources/AgentTrainer/
├── App/             app lifecycle and operation coordination
├── Capture/         ScreenCaptureKit, HEVC writing, and input monitoring
├── Storage/         workspace transactions and the binary event format
├── Preprocessing/   bounded Metal/MLX vision preprocessing
├── Training/        packed caches, policy, optimizer, validation, checkpoints
├── Agent/           live inference, focus safety, and synthetic input
├── Replay/          guarded real-input reenactment
├── Hotkeys/         global record, run, and panic shortcuts
├── Core/            domain contracts, logging, updates, and process execution
└── UI/              SwiftUI pages and diagnostics
```

`AppModel` owns user-visible state on the main actor. File libraries and cache
builders are actors. Capture, input, inference, and injection services use
small explicit locks or serial queues around their cross-thread state. Start
and stop paths use revision tokens so a cancelled launch cannot install a late
resource.

Dataset cache and exact-resume identity follow training-relevant content:
source file size/modification time, timeline, geometry, dimensions, and trims.
Renaming a recording, changing its thumbnail, or moving it between folders does
not decode the video again or invalidate training. Existing cache/checkpoint
identities are accepted once and migrated on reuse.

## Development rules

- Keep model, training-data, objective, and recording schemas separate. Change
  a schema only when its actual compatibility boundary changes.
- Preserve recordings when model contracts change. Incompatible model
  artifacts belong in recovery archives, not in destructive migrations.
- Write large generated files to private temporary directories and publish
  them with an atomic move.
- Drain child-process stdout and stderr concurrently.
- Keep ScreenCaptureKit callbacks small and never invoke external callbacks
  while holding a lifecycle lock.
- Add a regression test for every format, migration, concurrency, cache, or
  safety correction.
- Run `./test.sh`, `git diff --check`, and a signed-bundle verification before a
  release.

## Windows recorder status

`WindowsRecorder/` is an experimental capture companion, not a Windows version
of the trainer. It contains Record, Library, and recorder Settings only; model
training and AI execution remain on macOS. The recorder is unfinished and is
not the main development focus.

Its exported `.atrrecord.zip` packages use the same schema and Apple virtual-key
policy space as native recordings. macOS import remains the authoritative
validation boundary. See [WindowsRecorder/README.md](WindowsRecorder/README.md)
for its current requirements and developer workflow.

## Troubleshooting

- **No capture source:** grant Screen Recording permission, quit, and reopen.
- **No keyboard or pointer capture:** grant Input Monitoring permission and
  reopen.
- **AI cannot send input:** grant Accessibility permission and enable the
  corresponding Run output permission.
- **Privacy prompts return after rebuilding:** keep the same app path, bundle
  identifier, and stable signing identity; do not use ad-hoc signing.
- **Training cache looks stale or damaged:** use Diagnostics → Clear Caches.
  Recordings, profiles, checkpoints, and saved brains are not deleted.
- **Storage disk disconnected:** reconnect it or choose another location in
  Settings; AgentTrainer will not silently create a replacement library.
