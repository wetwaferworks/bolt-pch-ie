' Log view: opens a console that tails one log file. Style 4 shows the window without taking focus,
' and it is never topmost. The tail stops by itself when <log>.done appears; the window stays open until closed.
' Usage: wscript //B //Nologo view_log.vbs <log file>
If WScript.Arguments.Count < 1 Then WScript.Quit 2
Dim fso, here, logf
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)
logf = WScript.Arguments(0)
CreateObject("WScript.Shell").Run "powershell -NoProfile -NoExit -ExecutionPolicy Bypass -File """ & here & "\tail_log.ps1"" -Log """ & logf & """", 4, False
