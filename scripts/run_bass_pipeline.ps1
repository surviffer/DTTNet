param(
    [string]$Stem = "bass",
    [int[]]$Seeds = @(2021, 2022),
    [string[]]$Modes = @("baseline", "fixed", "learned_static", "dynamic"),
    [int]$TrainSteps = 7000,
    [int]$ValEvery = 200,
    [string]$LogDir = "D:\CZJ\experiments\dttnet_drff",
    [string]$Pretrained = ""
)

. (Join-Path $PSScriptRoot "windows_env.ps1")
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$Pretrained = if ($Pretrained) { $Pretrained } else { "D:\CZJ\DTTNet_pretrained\$Stem.ckpt" }
$extra = @(
    "callbacks.model_checkpoint.save_top_k=-1",
    "callbacks.resume_checkpoint.save_top_k=-1",
    "callbacks.resume_checkpoint.every_n_train_steps=200"
)

foreach ($seed in $Seeds) {
    foreach ($mode in $Modes) {
        $exp = "ft_${Stem}_${mode}_s${seed}"
        $run = Join-Path $LogDir $exp
        $done = Join-Path $run "DONE"

        if (-not (Test-Path -LiteralPath $done)) {
            Write-Host "[train] $exp"
            & (Join-Path $PSScriptRoot "finetune_all.ps1") `
                -Stems $Stem -Modes $mode -Seeds $seed `
                -TrainSteps $TrainSteps -ValEvery $ValEvery `
                -PretrainedDir (Split-Path -Parent $Pretrained) `
                -LogDir $LogDir -ExtraArgs $extra
            if ($LASTEXITCODE -ne 0) { throw "training failed: $exp" }
        } else {
            Write-Host "[train skip] $exp already DONE"
        }

        $eval = Join-Path $LogDir "eval_test\$exp"
        if (-not (Test-Path -LiteralPath (Join-Path $eval "eval.csv"))) {
            Write-Host "[select] $exp"
            & $script:Python (Join-Path $PSScriptRoot "select_best_checkpoint.py") $run
            if ($LASTEXITCODE -ne 0) { throw "checkpoint selection failed: $exp" }

            Write-Host "[test] $exp"
            $args = @(
                "run_eval.py", "model=$Stem", "model.bn_norm=BN", "model.g=32",
                "model.bandsequence.num_layers=4",
                "ckpt_path=$((Join-Path $run 'checkpoints\eval_best.ckpt').Replace('\','/'))",
                "split=test", "seed=$seed", "pool_workers=8", "logger=[]",
                "hydra.run.dir=$($eval.Replace('\','/'))"
            )
            if ($mode -eq "baseline") {
                $args += "model.mr_frontend.enabled=false"
            } else {
                $args += @("model.mr_frontend.enabled=true", "model.mr_frontend.fusion_mode=$mode")
            }
            & $script:Python @args
            if ($LASTEXITCODE -ne 0) { throw "test failed: $exp" }
        } else {
            Write-Host "[test skip] $exp already evaluated"
        }
    }
}
