$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$toolsRoot = Join-Path $env:USERPROFILE ".eyetracing-build-tools"
$sdkRoot = Join-Path $toolsRoot "android-sdk"
$gradleVersion = "8.10.2"
$gradleZip = Join-Path $toolsRoot "gradle-$gradleVersion-bin.zip"
$gradleRoot = Join-Path $toolsRoot "gradle-$gradleVersion"
$cmdlineZip = Join-Path $toolsRoot "commandlinetools-win.zip"
$cmdlineRoot = Join-Path $sdkRoot "cmdline-tools"
$cmdlineLatest = Join-Path $cmdlineRoot "latest"

New-Item -ItemType Directory -Force -Path $toolsRoot | Out-Null

if (!$env:JAVA_HOME) {
    $candidateJdks = @(
        "C:\Program Files\ojdkbuild\java-17-openjdk-17.0.3.0.6-1",
        "C:\Program Files\Eclipse Adoptium\jdk-17*",
        "C:\Program Files\Java\jdk-17*"
    )
    foreach ($candidate in $candidateJdks) {
        $resolved = Get-Item $candidate -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($resolved -and (Test-Path (Join-Path $resolved.FullName "bin/java.exe"))) {
            $env:JAVA_HOME = $resolved.FullName
            break
        }
    }
}

if (!$env:JAVA_HOME) {
    throw "JAVA_HOME is not set. Install JDK 17 and rerun this script."
}

$env:Path = (Join-Path $env:JAVA_HOME "bin") + [IO.Path]::PathSeparator + $env:Path

function Download-IfMissing($uri, $path) {
    if ((Test-Path $path) -and ((Get-Item $path).Length -gt 1024KB)) {
        return
    }
    if (Test-Path $path) {
        Remove-Item -LiteralPath $path -Force
    }
    Write-Host "Downloading $uri"
    curl.exe -L --fail --retry 5 --retry-delay 3 --output $path $uri
}

Download-IfMissing `
    "https://services.gradle.org/distributions/gradle-$gradleVersion-bin.zip" `
    $gradleZip

Download-IfMissing `
    "https://dl.google.com/android/repository/commandlinetools-win-14742923_latest.zip" `
    $cmdlineZip

if (!(Test-Path $gradleRoot)) {
    Expand-Archive -Path $gradleZip -DestinationPath $toolsRoot -Force
}

if (!(Test-Path $cmdlineLatest)) {
    $tempCmdline = Join-Path $toolsRoot "cmdline-expanded"
    Remove-Item -LiteralPath $tempCmdline -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive -Path $cmdlineZip -DestinationPath $tempCmdline -Force
    New-Item -ItemType Directory -Force -Path $cmdlineRoot | Out-Null
    Remove-Item -LiteralPath $cmdlineLatest -Recurse -Force -ErrorAction SilentlyContinue
    Move-Item -LiteralPath (Join-Path $tempCmdline "cmdline-tools") -Destination $cmdlineLatest
}

$env:ANDROID_HOME = $sdkRoot
$env:ANDROID_SDK_ROOT = $sdkRoot
$sdkManager = Join-Path $cmdlineLatest "bin/sdkmanager.bat"

Write-Host "Using JAVA_HOME=$env:JAVA_HOME"
Write-Host "Using ANDROID_HOME=$env:ANDROID_HOME"

1..100 | ForEach-Object { "y" } | & $sdkManager --sdk_root=$sdkRoot --licenses
if ($LASTEXITCODE -ne 0) { throw "Android SDK license acceptance failed." }

& $sdkManager --sdk_root=$sdkRoot "platform-tools" "platforms;android-35" "build-tools;35.0.0" "build-tools;34.0.0"
if ($LASTEXITCODE -ne 0) { throw "Android SDK package install failed." }

$gradleBat = Join-Path $gradleRoot "bin/gradle.bat"
& $gradleBat -p $repoRoot assembleDebug
if ($LASTEXITCODE -ne 0) { throw "Gradle build failed." }

$apk = Join-Path $repoRoot "app/build/outputs/apk/debug/app-debug.apk"
$releaseDir = Join-Path $repoRoot "releases"
New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null
Copy-Item -LiteralPath $apk -Destination (Join-Path $releaseDir "opengazelink-phone-debug.apk") -Force
Write-Host "APK copied to $(Join-Path $releaseDir 'opengazelink-phone-debug.apk')"
