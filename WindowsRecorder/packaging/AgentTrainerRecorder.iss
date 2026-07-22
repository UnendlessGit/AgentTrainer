#ifndef MyAppVersion
  #define MyAppVersion "1.8.8"
#endif

[Setup]
AppId={{B4A1B0EE-4B4E-4C8B-BA8D-C402AF84EE55}
AppName=AgentTrainer Recorder
AppVersion={#MyAppVersion}
AppPublisher=AgentTrainer
AppPublisherURL=https://github.com/UnendlessGit/AgentTrainer
AppSupportURL=https://github.com/UnendlessGit/AgentTrainer/issues
DefaultDirName={autopf}\AgentTrainer Recorder
DefaultGroupName=AgentTrainer Recorder
DisableProgramGroupPage=yes
OutputDir=..\artifacts
OutputBaseFilename=AgentTrainer-Recorder-{#MyAppVersion}-Setup-x64
SetupIconFile=..\src\AgentTrainer.Recorder\Assets\AppIcon.ico
UninstallDisplayIcon={app}\AgentTrainer Recorder.exe
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
WizardResizable=no
SetupLogging=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
CloseApplications=yes
RestartApplications=no
MinVersion=10.0.19041
AppMutex=AgentTrainerRecorder-B4A1B0EE-4B4E-4C8B-BA8D-C402AF84EE55
VersionInfoVersion={#MyAppVersion}
VersionInfoCompany=AgentTrainer
VersionInfoDescription=AgentTrainer Recorder Setup
VersionInfoProductName=AgentTrainer Recorder

[Files]
Source: "..\artifacts\publish\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "VC_redist.x64.exe"
Source: "..\artifacts\dependencies\VC_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
Name: "{group}\AgentTrainer Recorder"; Filename: "{app}\AgentTrainer Recorder.exe"
Name: "{autodesktop}\AgentTrainer Recorder"; Filename: "{app}\AgentTrainer Recorder.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
Filename: "{tmp}\VC_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Installing Microsoft Visual C++ runtime…"; Flags: waituntilterminated; Check: VCRuntimeRequired
Filename: "{app}\AgentTrainer Recorder.exe"; Description: "Launch AgentTrainer Recorder"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
function VCRuntimeRequired: Boolean;
var
  Installed: Cardinal;
begin
  Result := not (RegQueryDWordValue(HKLM64,
    'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
    'Installed', Installed) and (Installed = 1));
end;
