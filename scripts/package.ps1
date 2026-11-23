# Build the folder and zip that go to the demo laptop.
#
#   dist\PDC-Convolution\        PDC-Convolution.exe (the app, no Python needed),
#                                pdc_conv.exe, src\, results\, demo\, and the documents
#   dist\PDC-Convolution.zip     the same folder, zipped
#
# Needs: Python 3 (the packages in requirements.txt are installed into build\venv),
# and the compiler at C:\mingw64 for scripts\build.ps1.

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Set-Location $project

& (Join-Path $PSScriptRoot 'build.ps1')

# A private Python environment with only requirements.txt in it, so PyInstaller
# cannot pull in other packages that happen to be installed on this PC.
$venv = Join-Path $project 'build\venv'
if (-not (Test-Path (Join-Path $venv 'Scripts\python.exe'))) { python -m venv $venv }
$python = Join-Path $venv 'Scripts\python.exe'
& $python -m pip install --quiet --disable-pip-version-check -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'pip install failed.' }

# the app: one folder with its own Python inside; --windowed = no console window.
# --paths . lets PyInstaller find the "gui" package that app.pyw imports.
& $python -m PyInstaller --noconfirm --windowed --name PDC-Convolution `
    --distpath dist --workpath build\pyinstaller --specpath build --paths . gui\app.pyw
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }

$out = Join-Path $project 'dist\PDC-Convolution'
Copy-Item pdc_conv.exe $out -Force
foreach ($folder in 'src', 'results', 'demo') {
    $target = Join-Path $out $folder
    if (Test-Path $target) { Remove-Item $target -Recurse -Force }
    Copy-Item $folder $target -Recurse
}
foreach ($file in 'README.md', 'HOW_TO_RUN.md', 'REPORT.md', 'VIVA_GUIDE.md') {
    if (Test-Path $file) { Copy-Item $file $out -Force }
    else { Write-Warning "$file not found; it is not in the package." }
}

$zip = Join-Path $project 'dist\PDC-Convolution.zip'
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $out -DestinationPath $zip
'{0} ({1:N0} MB)' -f $zip, ((Get-Item $zip).Length / 1MB)
