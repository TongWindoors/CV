@echo off
cd /d "%~dp0"
echo stop>STOP_REQUESTED
echo Stop requested. Wait for the current operation to finish.
echo Delete STOP_REQUESTED before a new run.
pause
