$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
if (!(Test-Path $python)) {
    throw "Missing .venv. Run setup-venv.bat first."
}

& $python -m pip install -r (Join-Path $root "requirements-build.txt")
if ($LASTEXITCODE -ne 0) { throw "Installing build dependencies failed." }

Push-Location $root
try {
    & $python -m PyInstaller --noconfirm --clean OpenGazeLink.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }
} finally {
    Pop-Location
}

$releaseDir = Join-Path $root "release"
$portableZip = Join-Path $releaseDir "OpenGazeLink-portable-0.1.0.zip"
New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null
if (Test-Path $portableZip) {
    Remove-Item -LiteralPath $portableZip -Force
}
Compress-Archive -LiteralPath (Join-Path $root "dist\OpenGazeLink") -DestinationPath $portableZip -CompressionLevel Optimal
Write-Host "Portable archive written to $portableZip."

$innoCandidates = @(
    "$env:ProgramFiles\Inno Setup 7\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 7\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
)
$iscc = $innoCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($iscc) {
    & $iscc (Join-Path $root "installer\OpenGazeLink.iss")
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup build failed." }
    Write-Host "Installer written under release."
} else {
    Write-Host "Inno Setup 6 or 7 is not installed; portable builds are ready under dist\OpenGazeLink and release."
}
