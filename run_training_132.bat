@echo off
cd /d "%~dp0"
title LeafGuard 132-class training
echo ============================================
echo  LeafGuard 132-class training (132 classes)
echo  Keep this window OPEN. Plugged in. Do NOT sleep/reboot.
echo ============================================
start "" powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0keep_awake.ps1"
".venv\Scripts\python.exe" "%~dp0train_132_orchestrator.py"
echo.
echo === Finished or interrupted. Window stays open. ===
pause
