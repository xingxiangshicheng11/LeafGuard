$sig = @"
using System;
using System.Runtime.InteropServices;
public class SleepUtil {
  [DllImport("kernel32.dll")]
  public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
Add-Type -TypeDefinition $sig
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED = 0x80000001 (as UInt32)
$flags = [uint32]2147483649
while ($true) {
  [void][SleepUtil]::SetThreadExecutionState($flags)
  Start-Sleep -Seconds 60
}
