param(
    [ValidateSet("test", "valid")][string]$Split = "test",
    [string]$LogDir = "D:\CZJ\experiments\dttnet_drff",
    [string]$OutputDir = "D:\CZJ\experiments\dttnet_drff\results"
)

. (Join-Path $PSScriptRoot "windows_env.ps1")

& $script:Python scripts\summarize.py `
    --eval_dir (Join-Path $LogDir "eval_$Split") `
    --out_dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "Summary failed with exit code $LASTEXITCODE"
}

