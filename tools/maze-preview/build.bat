@echo off
if not defined PCHIE_WORKBENCH (
  echo Set PCHIE_WORKBENCH to the corresponding local path before running this lab script. There is no default path.
  exit /b 1
)
if not defined PCHIE_LAB (
  echo Set PCHIE_LAB to the corresponding local path before running this lab script. There is no default path.
  exit /b 1
)
if defined CMAKE_DIR set "PATH=%CMAKE_DIR%;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %PCHIE_LAB%\tools
cl /nologo /EHsc /O2 -std:c++17 -MD /fp:precise /arch:AVX2 -I%PCHIE_WORKBENCH%\pchie-baselines\include maze_preview.cpp /Fe:maze_preview.exe /link %PCHIE_WORKBENCH%\pchie-baselines\build-cpu\pchie_core.lib > build.log 2>&1
echo BUILD exit=%ERRORLEVEL%
