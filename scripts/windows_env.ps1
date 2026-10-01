$ErrorActionPreference = "Stop"

$script:ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script:Python = "D:\CZJ\.envs\dtt5090\Scripts\python.exe"
$script:FfmpegBin = "D:\CZJ\.tools\ffmpeg\ffmpeg-9.0.2-essentials_build\bin"

if (-not (Test-Path -LiteralPath $script:Python)) {
    throw "DTTNet Python environment not found: $script:Python"
}

$env:PATH = "$script:FfmpegBin;$env:PATH"
$env:PYTHONPATH = $script:ProjectRoot
$env:NUMBA_CACHE_DIR = "D:\CZJ\.cache\numba"
$env:TEMP = "D:\CZJ\.tmp"
$env:TMP = "D:\CZJ\.tmp"
$env:HYDRA_FULL_ERROR = "1"

Set-Location -LiteralPath $script:ProjectRoot

