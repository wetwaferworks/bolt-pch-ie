@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
if not exist build mkdir build
cl /nologo /EHsc /O2 /std:c++17 /MP12 /I..\include /Isrc src\pchie_anchor.cpp ..\src\Sha256.cpp /Fobuild\ /Fe:build\pchie_anchor.exe > build\build.log 2>&1 || (echo BUILD FAIL & type build\build.log & exit /b 1)
cl /nologo /EHsc /O2 /std:c++17 /I..\include /Isrc tests\anchor_unit.cpp ..\src\Sha256.cpp /Fobuild\ /Fe:build\anchor_unit.exe >> build\build.log 2>&1 || (echo BUILD FAIL & type build\build.log & exit /b 1)
echo BUILD OK
