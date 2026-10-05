param(
    [int]$Size = 12000,
    [int]$BatchN = 3000,
    [int]$BatchSize = 1024,
    [string]$Threads = '1,2,4,8',
    [int]$Reps = 3
)

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Set-Location $project
$env:OMP_PROC_BIND = $null
$env:OMP_PLACES = $null

& .\scripts\build_windows.ps1

$sizes = '1x1','2x3','3x3','4x7','5x5','17x9','37x53','128x1','1x128','257x131'
foreach ($testSize in $sizes) {
    $output = & .\pdc_conv.exe bench --size $testSize --threads 1,3,4 --reps 1 --csv NUL
    if ($LASTEXITCODE -ne 0 -or (($output -join "`n") -notmatch 'ALL VARIANTS MATCH')) {
        throw "Correctness test failed for $testSize."
    }
}

New-Item -ItemType Directory -Force results | Out-Null

& .\pdc_conv.exe bench --size $Size --threads $Threads --reps $Reps --csv results\results_bench.csv |
    Tee-Object results\bench.log
if ($LASTEXITCODE -ne 0) { throw 'Image benchmark failed.' }

& .\pdc_conv.exe batch --n $BatchN --bsize $BatchSize --threads $Threads --kernel blur --reps 2 --csv results\results_batch.csv |
    Tee-Object results\batch.log
if ($LASTEXITCODE -ne 0) { throw 'Batch benchmark failed.' }

python scripts\plot_results.py results\results_bench.csv results\results_batch.csv results\plots

New-Item -ItemType Directory -Force demo | Out-Null
& .\pdc_conv.exe gen --size 1024 --out demo\input.ppm
foreach ($kernel in 'blur','sharpen','sobel') {
    & .\pdc_conv.exe filter --in demo\input.ppm --kernel $kernel --variant omp_tiled --threads 4 --out "demo\$kernel.ppm"
}
@'
from pathlib import Path
from PIL import Image
for name in ('input', 'blur', 'sharpen', 'sobel'):
    Image.open(Path('demo') / f'{name}.ppm').save(Path('demo') / f'{name}.png')
'@ | python -
Remove-Item demo\*.ppm -Force

Write-Host 'Windows project run completed successfully.'
