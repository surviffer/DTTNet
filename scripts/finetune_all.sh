#!/usr/bin/env bash
# 单卡顺序跑完全部微调实验；已完成的实验会自动跳过，可随时中断后重新执行。
#
# 用法（在项目根目录）：
#   PRETRAINED_DIR=/path/to/ckpts bash scripts/finetune_all.sh
#
# 可选环境变量（括号内为默认值）：
#   STEMS  ("drums bass")
#   MODES  ("baseline learned_static dynamic")，可加 fixed、mid_only
#   SEEDS  ("2021 2022")
#   TRAIN_STEPS (15000)  VAL_EVERY (2000)  BATCH (4)  ACCUM (4)
#   EXTRA  (额外 Hydra 参数，例如 "trainer.limit_val_batches=2")
#
# 预训练权重需命名为 $PRETRAINED_DIR/{stem}.ckpt，例如 drums.ckpt、bass.ckpt。
# 每个实验输出到 $LOG_DIR/ft_{stem}_{mode}_s{seed}/，完成后写入 DONE 文件。

set -euo pipefail
cd "$(dirname "$0")/.."

# 读取 .env 中的 LOG_DIR 等变量
if [ -f .env ]; then set -a; source .env; set +a; fi

: "${PRETRAINED_DIR:?请设置 PRETRAINED_DIR}"
: "${LOG_DIR:?请在 .env 中设置 LOG_DIR}"
STEMS=${STEMS:-"drums bass"}
MODES=${MODES:-"baseline learned_static dynamic"}
SEEDS=${SEEDS:-"2021 2022"}
TRAIN_STEPS=${TRAIN_STEPS:-15000}
VAL_EVERY=${VAL_EVERY:-2000}
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-4}
EXTRA=${EXTRA:-}

mode_args() {
  if [ "$1" = "baseline" ]; then
    echo "model.mr_frontend.enabled=false"
  else
    echo "model.mr_frontend.enabled=true model.mr_frontend.fusion_mode=$1"
  fi
}

for stem in $STEMS; do
  ckpt="$PRETRAINED_DIR/$stem.ckpt"
  [ -f "$ckpt" ] || { echo "找不到预训练权重 $ckpt"; exit 1; }
  for seed in $SEEDS; do
    for mode in $MODES; do
      exp="ft_${stem}_${mode}_s${seed}"
      run_dir="$LOG_DIR/$exp"
      if [ -f "$run_dir/DONE" ]; then
        echo "[skip] $exp 已完成"
        continue
      fi
      if [ -d "$run_dir" ]; then
        failed="${run_dir}.failed_$(date '+%Y%m%d_%H%M%S')"
        echo "[redo] $exp 上次未完成，旧目录移到 $failed 后重跑"
        mv "$run_dir" "$failed"
      fi
      echo "[run ] $exp  $(date '+%F %T')"
      # shellcheck disable=SC2046
      python train.py experiment=finetune_1gpu datamodule=musdb_dev14 model="$stem" \
        pretrained_ckpt="$ckpt" seed="$seed" \
        train_steps="$TRAIN_STEPS" val_every_steps="$VAL_EVERY" \
        datamodule.batch_size="$BATCH" trainer.accumulate_grad_batches="$ACCUM" \
        exp_name="$exp" hydra.run.dir="$run_dir" \
        $(mode_args "$mode") $EXTRA
      date '+%F %T' > "$run_dir/DONE"
    done
  done
done

echo "全部训练完成"
