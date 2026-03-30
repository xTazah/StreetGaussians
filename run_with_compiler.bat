@echo off
setlocal enabledelayedexpansion

echo [Setup] Initializing Visual Studio environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1

:: Check if LIB is set
if "%LIB%"=="" (
    echo [Error] vcvars64.bat failed to set LIB environment variable.
    exit /b 1
)

:: Define the libraries we know are missing/problematic
set "MISSING_LIBS="

:: Check for standard MSVC lib
echo %LIB% | find /i "msvcprt.lib" > nul
if errorlevel 1 (
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.36.32532\lib\x64" (
         set "LIB=%LIB%;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.36.32532\lib\x64"
         echo [Fix] Added MSVC lib path explicitly.
    )
)

:: Check for Windows SDK libs (uuid.lib, ucrt.lib, etc)
:: We'll look for the latest SDK version installed
set "SDK_ROOT=C:\Program Files (x86)\Windows Kits\10\Lib"
set "LATEST_SDK="
if exist "%SDK_ROOT%" (
    for /f "delims=" %%I in ('dir "%SDK_ROOT%" /b /ad /o-n') do (
        set "LATEST_SDK=%%I"
        goto :FoundSDK
    )
)
:FoundSDK

if "%LATEST_SDK%"=="" (
    echo [Warning] Could not auto-detect Windows SDK version.
) else (
    echo [Setup] Detected Windows SDK: %LATEST_SDK%
    
    :: Universally check and append essential SDK paths if missing
    
    :: UM (User Mode) libs - uuid.lib, kernel32.lib, user32.lib
    echo !LIB! | find /i "\um\x64" > nul
    if errorlevel 1 (
        set "LIB=!LIB!;%SDK_ROOT%\%LATEST_SDK%\um\x64"
        echo [Fix] Added SDK UM lib path.
    )

    :: UCRT (Universal C Runtime) - ucrt.lib
    echo !LIB! | find /i "\ucrt\x64" > nul
    if errorlevel 1 (
        set "LIB=!LIB!;%SDK_ROOT%\%LATEST_SDK%\ucrt\x64"
        echo [Fix] Added SDK UCRT lib path.
    )
)

:: Add Python libs just in case
set "PYTHON_LIBS=C:\Users\FinnK\AppData\Local\Programs\Python\Python38\libs"
echo !LIB! | find /i "%PYTHON_LIBS%" > nul
if errorlevel 1 (
    set "LIB=!LIB!;%PYTHON_LIBS%"
    echo [Fix] Added Python libs path.
)

echo [Setup] Environment configured.
echo [Run] Executing: %*
%*
