@echo off
rem Scratch visnav, CUDA decide path. Same driver, compiled with PCHIE_SCRATCH_CUDA_VISNAV.
rem Needs CUDA_PATH and the root Release pchie_core.lib. nvcc uses the labeled precision flags.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set VIS=%~dp0
set ROOT=%~dp0..
if "%CUDA_PATH%"=="" (
  echo CUDA_PATH is not set
  exit /b 1
)
if not exist "%ROOT%\build\Release\pchie_core.lib" (
  echo missing %ROOT%\build\Release\pchie_core.lib. Build the root Release target first.
  exit /b 1
)
if not exist "%VIS%build" mkdir "%VIS%build"
"%CUDA_PATH%\bin\nvcc.exe" -c -O3 -std=c++17 -arch=sm_86 -fmad=false -prec-div=true -prec-sqrt=true -ftz=false -Xcompiler=/fp:precise,/MD,/EHsc -I"%ROOT%\include" "%ROOT%\src\pchie_cuda.cu" -o "%VIS%build\pchie_cuda.obj"
if errorlevel 1 (
  echo NVCC exit=%ERRORLEVEL%
  exit /b 1
)
cl /nologo /DWIN32 /D_WINDOWS /D_CRT_SECURE_NO_WARNINGS /DPCHIE_SCRATCH_CUDA_VISNAV /GR /EHsc /O2 /Ob2 /DNDEBUG -std:c++17 -MD /fp:precise /arch:AVX2 /W3 -I"%ROOT%\include" -I"%VIS%." -I"%CUDA_PATH%\include" "%VIS%pchie_lk_visnav.cpp" /Fo"%VIS%build\pchie_lk_visnav_cuda.obj" /Fe"%VIS%build\pchie_lk_visnav_cuda.exe" /link "%VIS%build\pchie_cuda.obj" "%ROOT%\build\Release\pchie_core.lib" "%CUDA_PATH%\lib\x64\cudart_static.lib" user32.lib
set RC=%ERRORLEVEL%
echo BUILD exit=%RC%
exit /b %RC%
