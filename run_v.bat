@echo off
setlocal EnableDelayedExpansion

:: ============================================================
::  run_v.bat  —  V_plugin launcher for SOC Ultralight
::
::  What this does:
::    1. Locates SOC_Ultralight on this machine
::    2. Copies v_plugin.py (and this bat) into plugins/
::    3. Checks GGUF Chatbox vision server on port 8082
::
::  After this runs, click [Start V] in SOC Ultralight to
::  activate Agent 4.  SOC detects the file automatically
::  (within ~5 seconds if already running).
::
::  CONFIGURE: set SOC_DIR below if auto-detect fails.
:: ============================================================

set "SOC_DIR="

:: -- Auto-detect SOC_Ultralight location --------------------
if "%SOC_DIR%"=="" if exist "%USERPROFILE%\SOC_Ultralight\plugins\" set "SOC_DIR=%USERPROFILE%\SOC_Ultralight"
if "%SOC_DIR%"=="" if exist "%USERPROFILE%\Desktop\SOC_Ultralight\plugins\" set "SOC_DIR=%USERPROFILE%\Desktop\SOC_Ultralight"
if "%SOC_DIR%"=="" if exist "C:\SOC_Ultralight\plugins\" set "SOC_DIR=C:\SOC_Ultralight"

if "%SOC_DIR%"=="" (
    echo.
    echo  [!] Could not locate SOC_Ultralight automatically.
    echo      Edit this file and set SOC_DIR= near the top.
    echo.
    pause
    exit /b 1
)

set "PLUGINS_DIR=%SOC_DIR%\plugins"

echo.
echo  V_plugin  ^|  run_v.bat
echo  -------------------------
echo  SOC dir  : %SOC_DIR%
echo  Plugins  : %PLUGINS_DIR%
echo.

:: -- Create plugins/ if missing -----------------------------
if not exist "%PLUGINS_DIR%\" mkdir "%PLUGINS_DIR%"

:: -- Copy v_plugin.py and this bat into plugins/ ------------
copy /Y "%~dp0v_plugin.py" "%PLUGINS_DIR%\v_plugin.py" >nul
if %errorlevel%==0 (
    echo  [OK] v_plugin.py  ->  %PLUGINS_DIR%
) else (
    echo  [!] Failed to copy v_plugin.py — check folder permissions.
    pause & exit /b 1
)

copy /Y "%~f0" "%PLUGINS_DIR%\run_v.bat" >nul
if %errorlevel%==0 (
    echo  [OK] run_v.bat    ->  %PLUGINS_DIR%
)

:: -- Check GGUF Chatbox vision server -----------------------
echo.
echo  Checking GGUF Chatbox vision server (port 8082)...
curl -s --max-time 3 http://localhost:8082/v1/models >nul 2>&1
if %errorlevel%==0 (
    echo  [OK] Vision server reachable — Agent 4 ready.
) else (
    echo  [!] Vision server not detected on port 8082.
    echo      Open GGUF Chatbox ^> Server ^> Vision Server
    echo      Set model + mmproj paths, then click Start.
)

echo.
echo  SOC Ultralight will show [Start V] automatically.
echo  Click it to activate Agent 4.
echo.
pause
