param([string]$Version = "0.2.0")
$ErrorActionPreference = "Stop"
if ($Version -notmatch '^\d+\.\d+\.\d+$') { throw "Version must be major.minor.patch." }

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$packageVersion = [regex]::Match((Get-Content -Raw (Join-Path $root 'opengazelink_pc/__init__.py')), '__version__ = "([^"]+)"').Groups[1].Value
if ($Version -ne $packageVersion) { throw "Requested version $Version differs from source version $packageVersion." }
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

# Exercise the frozen optimizer, model export, decoder and fresh control server
# before producing distributable archives. The self-check isolates user data.
$checkDir = Join-Path $root 'build/release-check'
New-Item -ItemType Directory -Force -Path $checkDir | Out-Null
$checkReport = Join-Path $checkDir "self-check-$([guid]::NewGuid().ToString('N')).json"
$check = Start-Process -FilePath (Join-Path $root 'dist/OpenGazeLink/OpenGazeLink.exe') `
    -ArgumentList @('self-check', '--report', ('"' + $checkReport + '"')) `
    -WorkingDirectory $checkDir -WindowStyle Hidden -PassThru
if (-not $check.WaitForExit(60000)) {
    Stop-Process -Id $check.Id -ErrorAction SilentlyContinue
    throw 'Frozen self-check timed out.'
}
if (-not (Test-Path -LiteralPath $checkReport) -or $check.ExitCode -ne 0) {
    throw "Frozen self-check failed; inspect $checkReport."
}
$checkResult = Get-Content -Raw -LiteralPath $checkReport | ConvertFrom-Json
if (-not $checkResult.ok) { throw "Frozen self-check failed; inspect $checkReport." }

$releaseDir = Join-Path $root "release"
$portableZip = Join-Path $releaseDir "OpenGazeLink-portable-$Version.zip"
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
    & $iscc "/DMyAppVersion=$Version" (Join-Path $root "installer\OpenGazeLink.iss")
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup build failed." }
    Write-Host "Installer written under release."
} else {
    Write-Host "Inno Setup 6 or 7 is not installed; portable builds are ready under dist\OpenGazeLink and release."
}
