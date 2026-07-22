# AgentTrainer Recorder for Windows

AgentTrainer Recorder is the Windows capture companion for AgentTrainer. It
contains only **Record**, **Library**, and recorder **Settings**. Training, model management, and AI
execution remain in the Apple-silicon macOS app.

Windows and macOS recordings use the same schema-2 `.atrrecord` package and
the same 72-byte input-event records. Windows physical scan codes are converted
to AgentTrainer's existing Apple virtual-key policy space before they are
written. The macOS app therefore treats imported Windows recordings exactly as
native recordings, including when both platforms appear in one training set.
The normative layout is [`../RECORDING_FORMAT.md`](../RECORDING_FORMAT.md).

## Requirements

- Windows 10 version 2004 or later, or Windows 11, on x64 hardware.
- A GPU/driver that exposes a Media Foundation H.264 encoder. HEVC is preferred
  and selected automatically when Windows can initialize it.
- Microsoft Media Foundation. Windows N/KN editions may require Microsoft's
  [Media Feature Pack](https://support.microsoft.com/windows/media-feature-pack-for-windows-n-8622b390-4ce6-43c9-9b42-549e5328e407); Windows Server requires the Media Foundation feature.
- [Microsoft Visual C++ 2015–2022 x64 runtime](https://aka.ms/vs/17/release/vc_redist.x64.exe). The installer includes the
  official redistributable and runs it only when needed.

The UI is per-monitor-DPI-aware. Capture uses physical desktop pixels and raw
input uses `WM_INPUT`, so mixed-DPI displays and high-polling-rate mice do not
pass through WPF coordinate or pointer-event coalescing.

The recommended download is the single `AgentTrainer-Recorder-1.8.8-Setup-x64.exe`
installer—the Windows equivalent of the Mac DMG. It installs the app, required
Microsoft runtime, Start-menu entry, optional desktop shortcut, and uninstaller,
then offers to launch the recorder. For the portable app zip, extract the complete folder and launch
`AgentTrainer Recorder.exe`. On a clean PC, first launch offers to run the
bundled Microsoft-signed `VC_redist.x64.exe`, then restarts the recorder. Do
not run the executable from inside the zip.
Production releases should Authenticode-sign the executable and installer with
the publisher's Windows signing certificate.

## Transfer workflow

1. Choose Display, Window, Window Region, or Screen Region in **Record**.
2. Set the same options available in the Mac recorder: source, library folder,
   region, FPS, cursor visibility, start/end trims, and key blacklist.
3. Record. The default `Ctrl+Alt+Win+R` starts/stops globally without entering
   the input stream. Change it in **Settings → Global keybind** by clicking the
   shortcut and pressing a new modifier chord. The app and optional live-input
   HUD are capture-excluded.
4. In **Library**, inspect the video and recorded controls, then choose
   **Export for Mac**. Copy the resulting single `.atrrecord.zip` file to the Mac.
5. In AgentTrainer for macOS, open **Library → Import**, choose one or more
   `.atrrecord.zip` files (unpacked `.atrrecord` folders also work), and select a
   native library folder.
6. Select any mixture of Windows and Mac recordings in AI Models/Training.

Imports are batch-transactional. Every source manifest, event stream, filename,
timestamp, count, video metadata, and first decodable video frame is checked on
the Mac before any item is published. Imported recordings receive a fresh local
ID and folder assignment; their video and control semantics are not rewritten.

## Storage and privacy

The Windows library is stored at:

```text
%LOCALAPPDATA%\AgentTrainer Recorder\
├── Recordings\
├── recording-folders.json
├── recorder-settings.json
└── Logs\
```

The app does not train, upload recordings, or require an account. Capture and
input stay local. Logs contain application/capture errors and recording IDs,
dimensions, counts, and durations—not video frames or recorded key contents.

## Build and test

Install the .NET 8 SDK, then run from PowerShell:

```powershell
cd WindowsRecorder
.\build.ps1
```

The script restores dependencies, runs the portable compatibility tests,
publishes a self-contained `win-x64` app, downloads and verifies the official
VC++ runtime, and produces both a portable zip and the recommended one-file
installer. Install Inno Setup 6 first (`winget install JRSoftware.InnoSetup`),
or pass `-SkipInstaller` explicitly for a portable-only development build.

For development without packaging:

```powershell
dotnet test .\tests\AgentTrainer.Recorder.Core.Tests -c Release
dotnet build .\src\AgentTrainer.Recorder -c Release -p:Platform=x64
```

After a real Windows capture has been copied to the Mac, the optional end-to-end
test imports it through the production Mac validator and builds one training
cache containing both that Windows package and a native Mac-authored package:

```bash
AGENTTRAINER_WINDOWS_SMOKE_PACKAGE=/path/to/capture.atrrecord \
  swift test -c debug --filter WindowsRuntimeSmokeTests
```

[`ScreenRecorderLib`](https://github.com/sskodje/ScreenRecorderLib) is pinned in the project file. It provides the native
Windows Graphics Capture/Desktop Duplication and Media Foundation writer. The
app requests hardware encoding, bounded rendering, no audio, no frame bitmap
preview, and a buffered event writer to keep capture overhead low.
