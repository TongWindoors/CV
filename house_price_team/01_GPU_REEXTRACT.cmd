@echo off
setlocal
cd /d "%~dp0"
title House Price Team Reproduction
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1" -Mode extract
set "TEAM_RUN_EXIT=%ERRORLEVEL%"
echo.
echo Optional NVIDIA GPU feature re-extraction
if not "%TEAM_RUN_EXIT%"=="0" echo FAILED. Read BOOTSTRAP_ERROR.txt and the latest logs folder entry.
if "%TEAM_RUN_EXIT%"=="0" echo Completed. Read the experiment report and result files.
pause
exit /b %TEAM_RUN_EXIT%
