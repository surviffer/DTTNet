param(
    [string[]]$Stems = @("drums", "bass"),
    [string[]]$Modes = @("baseline", "learned_static", "dynamic"),
    [int[]]$Seeds = @(2021, 2022),
    [int]$TrainSteps = 7000,
    [int]$ValEvery = 200,
    [int]$Batch = 4,
    [int]$Accum = 4,
    [string]$PretrainedDir = "D:\CZJ\DTTNet_pretrained",
    [string]$LogDir = "D:\CZJ\experiments\dttnet_drff",
    [string[]]$ExtraArgs = @()
)

. (Join-Path $PSScriptRoot "windows_env.ps1")
$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$history = Join-Path $LogDir "run_history.tsv"
if (-not (Test-Path -LiteralPath $history)) {
    "time`tevent`texperiment`tdetail" | Set-Content -LiteralPath $history -Encoding utf8
}

function Write-History([string]$Event, [string]$Experiment, [string]$Detail = "") {
    $line = "{0}`t{1}`t{2}`t{3}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Event, $Experiment, $Detail
    Add-Content -LiteralPath $history -Value $line -Encoding utf8
}

foreach ($stem in $Stems) {
    $checkpoint = Join-Path $PretrainedDir "$stem.ckpt"
    if (-not (Test-Path -LiteralPath $checkpoint)) {
        throw "Missing official pretrained checkpoint: $checkpoint"
    }

    foreach ($seed in $Seeds) {
        foreach ($mode in $Modes) {
            $experiment = "ft_{0}_{1}_s{2}" -f $stem, $mode, $seed
            $runDir = Join-Path $LogDir $experiment
            $done = Join-Path $runDir "DONE"
            if (Test-Path -LiteralPath $done) {
                Write-Host "[skip] $experiment already complete"
                continue
            }

            $arguments = @(
                "train.py",
                "experiment=finetune_1gpu",
                "datamodule=musdb_dev14",
                "model=$stem",
                "pretrained_ckpt=$($checkpoint.Replace('\', '/'))",
                "seed=$seed",
                "train_steps=$TrainSteps",
                "val_every_steps=$ValEvery",
                "datamodule.batch_size=$Batch",
                "trainer.accumulate_grad_batches=$Accum",
                "exp_name=$experiment",
                "hydra.run.dir=$($runDir.Replace('\', '/'))"
            )

            if ($mode -eq "baseline") {
                $arguments += "model.mr_frontend.enabled=false"
            } else {
                $arguments += @("model.mr_frontend.enabled=true", "model.mr_frontend.fusion_mode=$mode")
            }

            if (Test-Path -LiteralPath $runDir) {
                $resume = Get-ChildItem (Join-Path $runDir "resume\resume-*.ckpt") -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending | Select-Object -First 1
                if ($resume) {
                    $arguments += "resume_ckpt=$($resume.FullName.Replace('\', '/'))"
                    Write-History "resume" $experiment $resume.Name
                } else {
                    $archive = "$runDir.failed_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
                    Move-Item -LiteralPath $runDir -Destination $archive
                    Write-History "restart" $experiment "archived to $archive"
                }
            }

            $arguments += $ExtraArgs
            Write-History "start" $experiment
            Write-Host "[run ] $experiment"
            & $script:Python @arguments
            if ($LASTEXITCODE -ne 0) {
                Write-History "interrupted" $experiment "exit code $LASTEXITCODE"
                throw "$experiment failed with exit code $LASTEXITCODE"
            }

            New-Item -ItemType File -Force -Path $done | Out-Null
            Write-History "done" $experiment
        }
    }
}
