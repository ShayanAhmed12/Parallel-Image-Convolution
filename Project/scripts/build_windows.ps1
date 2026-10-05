$ErrorActionPreference = 'Stop'

$project = Split-Path -Parent $PSScriptRoot
Set-Location $project

$gcc = (Get-Command gcc -ErrorAction Stop).Source
$mingwBin = Split-Path -Parent $gcc
$temp = Join-Path $env:TEMP ('pdc-pthread-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $temp | Out-Null

try {
    $pthreadDll = Join-Path $mingwBin 'pthreadGC2.dll'
    $dlltool = Join-Path $mingwBin 'dlltool.exe'
    if ((Test-Path $pthreadDll) -and (Test-Path $dlltool)) {
        & $dlltool --export-all-symbols --dllname pthreadGC2.dll --output-lib (Join-Path $temp 'libpthread.a')
        if ($LASTEXITCODE -ne 0) { throw 'dlltool failed while creating the pthread import library.' }
    }

    & $gcc -O3 -march=native -fopenmp -Wall -Wextra -std=gnu11 `
        -o pdc_conv.exe src\pdc_conv.c -fopenmp -lm "-L$temp" -lpthread
    if ($LASTEXITCODE -ne 0) { throw 'gcc failed to build pdc_conv.exe.' }

    foreach ($runtime in 'libgomp-1.dll', 'pthreadGC2.dll') {
        $source = Join-Path $mingwBin $runtime
        if (Test-Path $source) { Copy-Item $source $project -Force }
    }
    Write-Host 'Built pdc_conv.exe.'
}
finally {
    Remove-Item $temp -Recurse -Force -ErrorAction SilentlyContinue
}