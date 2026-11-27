$root = $PSScriptRoot
Set-Location -LiteralPath $root
Start-Process powershell -ArgumentList @("-NoProfile","-ExecutionPolicy","Bypass","-File","$root\keep_awake.ps1") -WindowStyle Hidden
Write-Host "LeafGuard 132-class training starting. Keep this window open. Do NOT sleep/reboot."
& (Join-Path $root ".venv\Scripts\python.exe") (Join-Path $root "train_132_orchestrator.py")
Write-Host "=== Finished or interrupted. ==="
Read-Host "Press Enter to close"
