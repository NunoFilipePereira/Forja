; Instalador da Forja 3D para Windows (Inno Setup 6).
; Compilar depois do PyInstaller:  iscc /DAppVersion=1.0.0 packaging\forja.iss

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{6C1F3D2E-5B7A-4C1E-9D2F-0F0A4B7C9E11}
AppName=Forja 3D
AppVersion={#AppVersion}
AppPublisher=Nuno Filipe Pereira
AppPublisherURL=https://github.com/NunoFilipePereira/Forja-3D
DefaultDirName={autopf}\Forja 3D
DefaultGroupName=Forja 3D
UninstallDisplayIcon={app}\Forja3D.exe
OutputDir=..\dist
OutputBaseFilename=Forja3D-{#AppVersion}-Windows-instalador
SetupIconFile=forja.ico
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern
LicenseFile=..\LICENSE

[Languages]
Name: "portuguese"; MessagesFile: "compiler:Languages\Portuguese.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\Forja3D\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Forja 3D"; Filename: "{app}\Forja3D.exe"
Name: "{group}\{cm:UninstallProgram,Forja 3D}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Forja 3D"; Filename: "{app}\Forja3D.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Forja3D.exe"; Description: "{cm:LaunchProgram,Forja 3D}"; Flags: nowait postinstall skipifsilent
