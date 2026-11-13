@echo off
rem Scratch visnav, one CPU driver. Way-shut, reach, seeds, and NEML are runtime flags and default off.
rem Links the root Release pchie_core.lib. Does not define PCHIE_SCRATCH_CUDA_VISNAV.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set VIS=%~dp0
set ROOT=%~dp0..
if not exist "%ROOT%\build\Release\pchie_core.lib" (
  echo missing %ROOT%\build\Release\pchie_core.lib. Build the root Release target first.
  exit /b 1
)
if not exist "%VIS%build" mkdir "%VIS%build"
cl /nologo /DWIN32 /D_WINDOWS /D_CRT_SECURE_NO_WARNINGS /GR /EHsc /O2 /Ob2 /DNDEBUG -std:c++17 -MD /fp:precise /arch:AVX2 /W3 -I"%ROOT%\include" -I"%VIS%." "%VIS%pchie_lk_visnav.cpp" /Fo"%VIS%build\pchie_lk_visnav.obj" /Fe"%VIS%build\pchie_lk_visnav.exe" /link "%ROOT%\build\Release\pchie_core.lib" user32.lib
set RC=%ERRORLEVEL%
echo BUILD exit=%RC%
exit /b %RC%
