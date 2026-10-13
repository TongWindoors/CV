@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run 00_START.cmd once to create the environment.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -u run_experiments.py --mode collect
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" echo Collection failed. Check the message above.
pause
exit /b %RC%
