# Confirm port is closed
Test-NetConnection 127.0.0.1 -Port 8008 -InformationLevel Quiet

# Is the process even running?
Get-Process server -ErrorAction SilentlyContinue | Select-Object Id, StartTime

# Restart it (CCD1 affinity for 9950X3D):
$env:OMP_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"
$env:OMP_WAIT_POLICY = "PASSIVE"
$env:MKL_DYNAMIC     = "FALSE"

$exe = "C:\Users\Ken\Projects\xVASynth v3.0.0\v3.0.0\resources\app\cpython_cpu\server.exe"
Unblock-File $exe -ErrorAction SilentlyContinue

$xva = Start-Process -FilePath $exe `
                     -WorkingDirectory "C:\Users\Ken\Projects\xVASynth v3.0.0\v3.0.0" `
                     -PassThru -ErrorAction Stop
Start-Sleep -Milliseconds 800
$xva.ProcessorAffinity = [IntPtr]0xFF00
Write-Host "xVASynth PID:" $xva.Id "Affinity: 0x$('{0:X}' -f $xva.ProcessorAffinity.ToInt64())"