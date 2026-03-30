@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul
setlocal enabledelayedexpansion

:: Appending detected paths explicitly
set "LIB=%LIB%;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\lib\x64"
set "LIB=%LIB%;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64"
set "LIB=%LIB%;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64"
set "LIB=%LIB%;C:\Users\FinnK\AppData\Local\Programs\Python\Python38\libs"

echo [Launcher] Environment set. Libs added.
echo [Launcher] Running command: %*
%*