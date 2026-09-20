#define MyAppName "OpenGazeLink"
#ifndef MyAppVersion
#define MyAppVersion "0.2.0"
#endif
#define MyAppPublisher "OpenGazeLink"
#define MyAppExeName "OpenGazeLink.exe"

[Setup]
AppId={{7B4BBF17-A4AB-4A1E-BBBC-789F2592E114}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\OpenGazeLink
DefaultGroupName=OpenGazeLink
OutputDir=..\release
OutputBaseFilename=OpenGazeLink-Setup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=commandline
UninstallDisplayIcon={app}\{#MyAppExeName}
CloseApplications=yes
RestartApplications=no

[Files]
Source: "..\dist\OpenGazeLink\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
Type: files; Name: "{app}\EyeTracing.exe"
Type: files; Name: "{app}\EyeTracing-Control.cmd"
Type: files; Name: "{app}\EyeTracing-Runtime.cmd"
Type: files; Name: "{app}\EyeTracing-Stop.cmd"
Type: files; Name: "{group}\EyeTracing 控制中心.lnk"
Type: files; Name: "{group}\EyeTracing 后台运行.lnk"
Type: files; Name: "{group}\退出 EyeTracing.lnk"
Type: files; Name: "{autodesktop}\EyeTracing 控制中心.lnk"
Type: filesandordirs; Name: "{commonprograms}\EyeTracing"
Type: files; Name: "{commondesktop}\EyeTracing 控制中心.lnk"

[Icons]
Name: "{group}\OpenGazeLink 控制中心"; Filename: "{app}\{#MyAppExeName}"; Parameters: "control"; WorkingDir: "{app}"
Name: "{group}\OpenGazeLink 后台运行"; Filename: "{app}\{#MyAppExeName}"; Parameters: "runtime"; WorkingDir: "{app}"
Name: "{group}\退出 OpenGazeLink"; Filename: "{app}\{#MyAppExeName}"; Parameters: "stop"; WorkingDir: "{app}"
Name: "{autodesktop}\OpenGazeLink 控制中心"; Filename: "{app}\{#MyAppExeName}"; Parameters: "control"; WorkingDir: "{app}"

[Run]
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""EyeTracing UDP"""; Flags: runhidden waituntilterminated
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""OpenGazeLink UDP"""; Flags: runhidden waituntilterminated
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""OpenGazeLink UDP"" dir=in action=allow protocol=UDP localport=5006,5007 profile=private program=""{app}\{#MyAppExeName}"""; Flags: runhidden waituntilterminated
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""OpenGazeLink TCP"""; Flags: runhidden waituntilterminated
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""OpenGazeLink TCP"" dir=in action=allow protocol=TCP localport=5007 profile=private program=""{app}\{#MyAppExeName}"""; Flags: runhidden waituntilterminated
Filename: "{app}\{#MyAppExeName}"; Parameters: "control"; Description: "启动 OpenGazeLink 控制中心"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{app}\{#MyAppExeName}"; Parameters: "stop"; Flags: runhidden waituntilterminated; RunOnceId: "StopOpenGazeLink"
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""OpenGazeLink UDP"""; Flags: runhidden waituntilterminated; RunOnceId: "RemoveOpenGazeLinkFirewall"
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""OpenGazeLink TCP"""; Flags: runhidden waituntilterminated; RunOnceId: "RemoveOpenGazeLinkTcpFirewall"

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  { User calibration and models live under LocalAppData and are intentionally preserved. }
end;
