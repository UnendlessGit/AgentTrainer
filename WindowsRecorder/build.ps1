[CmdletBinding()]
param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Artifacts = Join-Path $Root "artifacts"
$Publish = Join-Path $Artifacts "publish"
$Dependencies = Join-Path $Artifacts "dependencies"
$Project = Join-Path $Root "src\AgentTrainer.Recorder\AgentTrainer.Recorder.csproj"
$Tests = Join-Path $Root "tests\AgentTrainer.Recorder.Core.Tests\AgentTrainer.Recorder.Core.Tests.csproj"
$Version = "1.8.8"

if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
    throw "The .NET 8 SDK is required. Install it from https://dotnet.microsoft.com/download/dotnet/8.0"
}

New-Item -ItemType Directory -Force -Path $Artifacts, $Dependencies | Out-Null
if (Test-Path $Publish) { Remove-Item -Recurse -Force $Publish }

dotnet restore $Project -p:Platform=x64
dotnet test $Tests -c $Configuration --nologo
dotnet publish $Project -c $Configuration -r win-x64 --self-contained true -p:Platform=x64 -o $Publish --nologo
Get-ChildItem $Publish -Filter "*.pdb" -File | Remove-Item -Force
Get-ChildItem $Publish -Filter "*.xml" -File | Remove-Item -Force
foreach ($Required in @("AgentTrainer Recorder.exe", "AgentTrainer.Recorder.Core.dll", "ScreenRecorderLib.dll")) {
    if (-not (Test-Path (Join-Path $Publish $Required))) { throw "Publish output is missing $Required" }
}

$Runtime = Join-Path $Dependencies "VC_redist.x64.exe"
if (-not (Test-Path $Runtime)) {
    Invoke-WebRequest -UseBasicParsing -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile $Runtime
}
$RuntimeSignature = Get-AuthenticodeSignature $Runtime
if ($RuntimeSignature.Status -ne "Valid" -or $RuntimeSignature.SignerCertificate.Subject -notmatch "Microsoft") {
    throw "The downloaded Visual C++ runtime does not have a valid Microsoft signature."
}

$PortableRuntime = Join-Path $Publish "VC_redist.x64.exe"
Copy-Item $Runtime $PortableRuntime -Force
Copy-Item (Join-Path $Root "README.md") (Join-Path $Publish "README-WINDOWS.md") -Force

$Zip = Join-Path $Artifacts "AgentTrainer-Recorder-$Version-win-x64.zip"
if (Test-Path $Zip) { Remove-Item -Force $Zip }
Compress-Archive -Path (Join-Path $Publish "*") -DestinationPath $Zip -CompressionLevel Optimal

$Installer = $null
if (-not $SkipInstaller) {
    $CompilerCandidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    )
    $Compiler = $CompilerCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($Compiler) {
        & $Compiler "/DMyAppVersion=$Version" (Join-Path $Root "packaging\AgentTrainerRecorder.iss")
        $Installer = Join-Path $Artifacts "AgentTrainer-Recorder-$Version-Setup-x64.exe"
    } else {
        throw "Inno Setup 6 was not found. Install JRSoftware.InnoSetup with winget, or pass -SkipInstaller explicitly for a portable-only build."
    }
}

$HashTargets = @($Zip)
if ($Installer -and (Test-Path $Installer)) { $HashTargets += $Installer }
$HashTargets | ForEach-Object { Get-FileHash -Algorithm SHA256 $_ } |
    ForEach-Object { "{0}  {1}" -f $_.Hash.ToLowerInvariant(), (Split-Path $_.Path -Leaf) } |
    Set-Content -Encoding ascii (Join-Path $Artifacts "SHA256SUMS.txt")

Write-Host "Published: $Publish"
Write-Host "Portable:  $Zip"
Write-Host "Checksums: $(Join-Path $Artifacts 'SHA256SUMS.txt')"
