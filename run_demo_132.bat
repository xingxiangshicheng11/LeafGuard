@echo off
cd /d "%~dp0"
title LeafGuard 132-class demo
echo ==================================================
echo  LeafGuard 132-class demo (keep this window OPEN)
echo  Wait for a line like:
echo    LeafGuard demo running: http://127.0.0.1:8011
echo  Then open that exact URL in your browser.
echo ==================================================
".venv\Scripts\python.exe" scripts\serve_demo.py --checkpoint results\combined132\seed42\e3\checkpoints\resnet18_best.pt --port 8011
echo.
echo === server stopped or an error occurred above. Press any key to close. ===
pause
