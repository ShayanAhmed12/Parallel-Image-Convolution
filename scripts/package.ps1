# Build the folder and zip that go to the demo laptop.
#
#   dist\PDC-Convolution\        PDC-Convolution.exe (the app, no Python needed),
#                                pdc_conv.exe, src\, results\, demo\, README, REPORT, VIVA_GUIDE
#   dist\PDC-Convolution.zip     the same folder, zipped
#
# Needs: Python with the packages in requirements.txt (including pyinstaller),
# and the compiler at C:\mingw64 for scripts\build.ps1.

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Set-Location $project

& (Join-Path $PSScriptRoot 'build.ps1')

# the app: one folder with its own Python inside; --windowed = no console window.
# --paths . lets PyInstaller find the "gui" package that app.pyw imports.
python -m PyInstaller --noconfirm --windowed --name PDC-Convolution `
    --distpath dist --workpath build\pyinstaller --specpath build --paths . gui\app.pyw
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }

$out = Join-Path $project 'dist\PDC-Convolution'
Copy-Item pdc_conv.exe $out -Force
foreach ($folder in 'src', 'results', 'demo') {
    $target = Join-Path $out $folder
    if (Test-Path $target) { Remove-Item $target -Recurse -Force }
    Copy-Item $folder $target -Recurse
}
foreach ($file in 'README.md', 'REPORT.md', 'VIVA_GUIDE.md') {
    if (Test-Path $file) { Copy-Item $file $out -Force }
    else { Write-Warning "$file not found; it is not in the package." }
}

$zip = Join-Path $project 'dist\PDC-Convolution.zip'
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $out -DestinationPath $zip
'{0} ({1:N0} MB)' -f $zip, ((Get-Item $zip).Length / 1MB)
