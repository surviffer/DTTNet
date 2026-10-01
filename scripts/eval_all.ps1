param(
    [string[]]$Stems = @("drums", "bass"),
    [string[]]$Modes = @("baseline", "learned_static", "dynamic"),
    [int[]]$Seeds = @(2021, 2022),
    [ValidateSet("test", "valid")][string]$Split = "test",
    [switch]$OverlapAdd,
    [int]$PoolWorkers = 8,
    [string]$LogDir = "D:\CZJ\experiments\dttnet_drff"
)

. (Join-Path $PSScriptRoot "windows_env.ps1")
$ErrorActionPreference = "Stop"
$missing = [System.Collections.Generic.List[string]]::new()

foreach ($stem in $Stems) {
    foreach ($seed in $Seeds) {
        foreach ($mode in $Modes) {
            $experiment = "ft_{0}_{1}_s{2}" -f $stem, $mode, $seed
            $runDir = Join-Path $LogDir $experiment
            $outDir = Join-Path $LogDir "eval_$Split\$experiment"

            if (-not (Test-Path -LiteralPath (Join-Path $runDir "DONE"))) {
                $missing.Add("$experiment (training incomplete)")
                continue
            }
            if (Test-Path -LiteralPath (Join-Path $outDir "per_track.csv")) {
                Write-Host "[skip] $experiment already evaluated"
                continue
            }

            $best = Get-ChildItem (Join-Path $runDir "checkpoints\*.ckpt") -ErrorAction SilentlyContinue |
                Where-Object Name -ne "last.ckpt" | Select-Object -First 1
            if (-not $best) {
                $missing.Add("$experiment (best checkpoint missing)")
                continue
            }

            # Hydra 1.4 treats '=' in a checkpoint filename as override syntax.
            # Keep the original checkpoint intact and use a safe evaluation alias.
            $safeBest = Join-Path $runDir "eval_best.ckpt"
            if ((-not (Test-Path -LiteralPath $safeBest)) -or
                ((Get-Item -LiteralPath $safeBest).Length -ne $best.Length)) {
                Copy-Item -LiteralPath $best.FullName -Destination $safeBest -Force
            }

            $arguments = @(
                "run_eval.py", "model=$stem", "model.bn_norm=BN", "model.g=32",
                "model.bandsequence.num_layers=4", "ckpt_path=$($safeBest.Replace('\', '/'))",
                "split=$Split", "seed=$seed", "pool_workers=$PoolWorkers", "logger=[]",
                "hydra.run.dir=$($outDir.Replace('\', '/'))"
            )
            if ($mode -eq "baseline") {
                $arguments += "model.mr_frontend.enabled=false"
            } else {
                $arguments += @("model.mr_frontend.enabled=true", "model.mr_frontend.fusion_mode=$mode")
            }
            # The evaluation config defaults overlap_add to null. Passing the
            # legacy literal `overlap_add=null` breaks Hydra 1.4 parsing.

            Write-Host "[eval] $experiment"
            & $script:Python @arguments
            if ($LASTEXITCODE -ne 0) {
                throw "$experiment evaluation failed with exit code $LASTEXITCODE"
            }
        }
    }
}

if ($missing.Count -gt 0) {
    throw "Expected experiments are unavailable:`n$($missing -join [Environment]::NewLine)"
}
