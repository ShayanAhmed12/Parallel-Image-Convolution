param(
    # Path to gcc.exe. Default: C:\mingw64\bin\gcc.exe (64-bit WinLibs) if present,
    # otherwise whatever gcc is on PATH.
    [string]$Compiler = '',
    # Allow building with a 32-bit compiler (limited to ~2 GB address space,
    # which is too small for the 12000x12000 blur benchmark).
    [switch]$Allow32Bit
)

$ErrorActionPreference = 'Stop'

$project = Split-Path -Parent $PSScriptRoot
Set-Location $project

if (-not $Compiler) {
    $default64 = 'C:\mingw64\bin\gcc.exe'
    if (Test-Path $default64) { $Compiler = $default64 }
    else { $Compiler = (Get-Command gcc -ErrorAction Stop).Source }
}
if (-not (Test-Path $Compiler)) { throw "Compiler not found: $Compiler" }

$gcc = (Resolve-Path $Compiler).Path
$mingwBin = Split-Path -Parent $gcc
$target = (& $gcc -dumpmachine).Trim()
$is64 = $target -like 'x86_64-*'
Write-Host "Compiler: $gcc ($target)"

$flags = @('-O3', '-march=native', '-fopenmp', '-Wall', '-Wextra', '-std=gnu11')

if ($is64) {
    # mingw-w64: link libgomp/winpthreads statically so the exe is self-contained
    # and can never pick up a 32-bit libgomp-1.dll lying next to it.
    & $gcc @flags -o pdc_conv.exe src\pdc_conv.c -static -lm
    if ($LASTEXITCODE -ne 0) { throw 'gcc failed to build pdc_conv.exe.' }
    # Runtime DLLs left over from an old 32-bit build would only cause confusion.
    foreach ($runtime in 'libgomp-1.dll', 'pthreadGC2.dll') {
        Remove-Item (Join-Path $project $runtime) -Force -ErrorAction SilentlyContinue
    }
    Write-Host 'Built pdc_conv.exe (64-bit, static).'
    return
}

if (-not $Allow32Bit) {
    throw ("$gcc targets $target (32-bit). The 12000x12000 benchmark needs a 64-bit compiler. " +
           "Install a 64-bit MinGW-w64 GCC to C:\mingw64, pass -Compiler <path-to-64-bit-gcc.exe>, " +
           "or pass -Allow32Bit for small runs only.")
}

# Legacy 32-bit MinGW.org toolchain: libgomp depends on pthreadGC2.dll, for which
# no import library ships, so build one with dlltool and copy the runtime DLLs.
$temp = Join-Path $env:TEMP ('pdc-pthread-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $temp | Out-Null
try {
    $pthreadDll = Join-Path $mingwBin 'pthreadGC2.dll'
    $dlltool = Join-Path $mingwBin 'dlltool.exe'
    if ((Test-Path $pthreadDll) -and (Test-Path $dlltool)) {
        & $dlltool --export-all-symbols --dllname pthreadGC2.dll --output-lib (Join-Path $temp 'libpthread.a')
        if ($LASTEXITCODE -ne 0) { throw 'dlltool failed while creating the pthread import library.' }
    }

    & $gcc @flags -o pdc_conv.exe src\pdc_conv.c -lm "-L$temp" -lpthread
    if ($LASTEXITCODE -ne 0) { throw 'gcc failed to build pdc_conv.exe.' }

    foreach ($runtime in 'libgomp-1.dll', 'pthreadGC2.dll') {
        $source = Join-Path $mingwBin $runtime
        if (Test-Path $source) { Copy-Item $source $project -Force }
    }
    Write-Host 'Built pdc_conv.exe (32-bit).'
}
finally {
    Remove-Item $temp -Recurse -Force -ErrorAction SilentlyContinue
}
