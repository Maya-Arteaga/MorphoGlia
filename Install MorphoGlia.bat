@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%~dp0"
set "LOCAL_PIXI_HOME=%ROOT%.pixi-home"
set "LOCAL_PIXI=%LOCAL_PIXI_HOME%\bin\pixi.exe"
set "GLOBAL_PIXI=%USERPROFILE%\.pixi\bin\pixi.exe"
set "LOG=%ROOT%MorphoGlia_install_log.txt"
set "ARCH=%PROCESSOR_ARCHITECTURE%"
if defined PROCESSOR_ARCHITEW6432 set "ARCH=%PROCESSOR_ARCHITEW6432%"
> "%LOG%" echo MorphoGlia 2.0.0 Windows installer
>> "%LOG%" echo Date: %DATE% %TIME%
>> "%LOG%" echo Architecture: %ARCH%
>> "%LOG%" echo Root: %ROOT%
>> "%LOG%" echo.
echo.
echo ============================================================
echo                MORPHOGLIA 2.0.0 INSTALLER
echo ============================================================
echo.
if /I "%ARCH%"=="ARM64" goto :unsupported
if /I not "%ARCH%"=="AMD64" goto :unsupported
set "PIXI="
if exist "%LOCAL_PIXI%" (
    set "PIXI=%LOCAL_PIXI%"
    goto :pixi_ready
)
if exist "%GLOBAL_PIXI%" (
    set "PIXI=%GLOBAL_PIXI%"
    goto :pixi_ready
)
where pixi.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PIXI=pixi.exe"
    goto :pixi_ready
)
echo Installing the MorphoGlia environment manager...
echo This requires an internet connection.
set "MORPHOGLIA_PIXI_HOME=%LOCAL_PIXI_HOME%"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop'; $env:PIXI_HOME=$env:MORPHOGLIA_PIXI_HOME; irm -useb https://pixi.sh/install.ps1 | iex" ^
  >> "%LOG%" 2>&1
if exist "%LOCAL_PIXI%" (
    set "PIXI=%LOCAL_PIXI%"
    goto :pixi_ready
)
where winget.exe >nul 2>nul
if %ERRORLEVEL% NEQ 0 goto :pixi_failed
winget install --id prefix-dev.pixi -e --accept-source-agreements --accept-package-agreements >> "%LOG%" 2>&1
if exist "%GLOBAL_PIXI%" (
    set "PIXI=%GLOBAL_PIXI%"
    goto :pixi_ready
)
where pixi.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PIXI=pixi.exe"
    goto :pixi_ready
)
goto :pixi_failed
:pixi_ready
echo Pixi found: %PIXI%
"%PIXI%" --version >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 goto :pixi_failed
echo Installing the locked MorphoGlia environment...
"%PIXI%" install --locked >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 goto :environment_failed
echo Checking the installation...
"%PIXI%" run doctor >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 goto :doctor_failed
echo.
echo MorphoGlia 2.0.0 is ready. Opening MorphoGlia...
start "" "%ROOT%MorphoGlia.bat"
exit /b 0
:unsupported
echo.
echo MorphoGlia 2.0.0 currently supports Windows x86-64 Intel/AMD computers.
echo Detected architecture: %ARCH%
>> "%LOG%" echo Unsupported architecture: %ARCH%
pause
exit /b 1
:pixi_failed
echo.
echo ERROR: Pixi installation failed. Log: %LOG%
type "%LOG%"
pause
exit /b 1
:environment_failed
echo.
echo ERROR: MorphoGlia environment installation failed. Log: %LOG%
type "%LOG%"
pause
exit /b 1
:doctor_failed
echo.
echo ERROR: A MorphoGlia compatibility check failed. Log: %LOG%
type "%LOG%"
pause
exit /b 1
