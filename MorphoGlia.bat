@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%~dp0"
set "LOCAL_PIXI=%ROOT%.pixi-home\bin\pixi.exe"
set "GLOBAL_PIXI=%USERPROFILE%\.pixi\bin\pixi.exe"
if exist "%LOCAL_PIXI%" (
    "%LOCAL_PIXI%" run gui
    exit /b %ERRORLEVEL%
)
if exist "%GLOBAL_PIXI%" (
    "%GLOBAL_PIXI%" run gui
    exit /b %ERRORLEVEL%
)
where pixi.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    pixi.exe run gui
    exit /b %ERRORLEVEL%
)
echo.
echo MorphoGlia's environment manager was not found.
echo Run "Install MorphoGlia.bat" first.
echo.
pause
exit /b 1
