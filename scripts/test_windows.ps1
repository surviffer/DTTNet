. (Join-Path $PSScriptRoot "windows_env.ps1")

& $script:Python -m pytest tests\unit\test_mr_frontend.py -v
if ($LASTEXITCODE -ne 0) {
    throw "DTTNet unit tests failed with exit code $LASTEXITCODE"
}

