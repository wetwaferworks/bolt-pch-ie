@echo off
rem Scratch visnav, one CPU driver. The CMake Release build also builds it (build\Release\pchie_lk_visnav.exe).
rem Way-shut, reach, act-blank, and memory injection are runtime flags and default off.
rem Memory (NEML) receipts are on by default; --neml off or PCHIE_NEML=0 turns them off.
rem Links the root Release pchie_core.lib. Does not define PCHIE_SCRATCH_CUDA_VISNAV.
set VIS=%~dp0
set ROOT=%~dp0..
where cl >nul 2>nul
if errorlevel 1 call :vcvars
if errorlevel 1 exit /b 1
if not exist "%ROOT%\build\Release\pchie_core.lib" (
  echo missing %ROOT%\build\Release\pchie_core.lib. Build the root Release target first.
  exit /b 1
)
if not exist "%VIS%build" mkdir "%VIS%build"
cl /nologo /DWIN32 /D_WINDOWS /D_CRT_SECURE_NO_WARNINGS /GR /EHsc /O2 /Ob2 /DNDEBUG -std:c++17 -MD /fp:precise /arch:AVX2 /W3 -I"%ROOT%\include" -I"%VIS%." "%VIS%pchie_lk_visnav.cpp" /Fo"%VIS%build\pchie_lk_visnav.obj" /Fe"%VIS%build\pchie_lk_visnav.exe" /link "%ROOT%\build\Release\pchie_core.lib" user32.lib
set RC=%ERRORLEVEL%
echo BUILD exit=%RC%
exit /b %RC%

:vcvars
rem Find the newest Visual Studio with the C++ tools (Community, Professional, Enterprise, or Build Tools).
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo vswhere.exe not found. Install Visual Studio 2022 with the C++ tools, or run from a Developer Command Prompt.
  exit /b 1
)
set "VSDIR="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
  echo Visual Studio C++ tools not found. Run from a Developer Command Prompt.
  exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
exit /b %ERRORLEVEL%
