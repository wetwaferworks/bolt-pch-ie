@echo off
rem Scratch NEML tool. Links the root Release pchie_core.lib.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set VIS=%~dp0
set ROOT=%~dp0..
if not exist "%ROOT%\build\Release\pchie_core.lib" (
  echo missing %ROOT%\build\Release\pchie_core.lib. Build the root Release target first.
  exit /b 1
)
if not exist "%VIS%build" mkdir "%VIS%build"
cl /nologo /D_CRT_SECURE_NO_WARNINGS /EHsc /O2 /DNDEBUG -std:c++17 -MD /fp:precise /arch:AVX2 /W3 -I"%ROOT%\include" -I"%VIS%neml" "%VIS%neml\neml_tool.cpp" /Fo"%VIS%build\neml_tool.obj" /Fe"%VIS%build\neml_tool.exe" /link "%ROOT%\build\Release\pchie_core.lib"
set RC=%ERRORLEVEL%
echo BUILD exit=%RC%
exit /b %RC%
