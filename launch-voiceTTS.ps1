# Launch voice generation server for Many-Mind Kernel
# Run from any PowerShell terminal on the Gaming PC.

$env:OMP_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"
$env:OMP_WAIT_POLICY = "PASSIVE"
$env:MKL_DYNAMIC     = "FALSE"

cmd /c start "" /affinity FF00 /D "C:\Users\Ken\Projects\xVASynth v3.0.0\v3.0.0" /B "C:\Users\Ken\Projects\xVASynth v3.0.0\v3.0.0\resources\app\cpython_cpu\server.exe"