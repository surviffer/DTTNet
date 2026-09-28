#!/usr/bin/env bash
# 对 finetune_all.sh 产生的每个实验，用 val/usdr 最高的 checkpoint 在测试集上评估一次。
# 已评估过的实验会自动跳过。全部训练完成、规则冻结之后再运行。
#
# 用法（在项目根目录）：
#   bash scripts/eval_all.sh
#
# 可选环境变量：
#   OVERLAP_ADD (off)  设为 on 时使用 overlap-add（慢约 4 倍）；所有模型必须一致
#   SPLIT (test)       调试时可设为 valid
#   POOL_WORKERS (8)   计算指标的 CPU 进程数

set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; source .env; set +a; fi

: "${LOG_DIR:?请在 .env 中设置 LOG_DIR}"
OVERLAP_ADD=${OVERLAP_ADD:-off}
SPLIT=${SPLIT:-test}
POOL_WORKERS=${POOL_WORKERS:-8}

overlap_arg=""
[ "$OVERLAP_ADD" = "off" ] && overlap_arg="overlap_add=null"

for run_dir in "$LOG_DIR"/ft_*_s*; do
  [ -f "$run_dir/DONE" ] || continue
  exp=$(basename "$run_dir")                 # ft_{stem}_{mode}_s{seed}
  stem=$(echo "$exp" | cut -d_ -f2)
  seed=${exp##*_s}
  mode=${exp#ft_${stem}_}; mode=${mode%_s${seed}}
  out_dir="$LOG_DIR/eval_${SPLIT}/$exp"

  if [ -f "$out_dir/per_track.csv" ]; then
    echo "[skip] $exp 已评估"
    continue
  fi

  # 最优 checkpoint（save_top_k=1，排除 last.ckpt）
  best=$(ls "$run_dir"/checkpoints/*.ckpt 2>/dev/null | grep -v '/last.ckpt$' | head -n 1 || true)
  [ -n "$best" ] || { echo "[warn] $exp 没有最优 checkpoint"; continue; }

  if [ "$mode" = "baseline" ]; then
    margs="model.mr_frontend.enabled=false"
  else
    margs="model.mr_frontend.enabled=true model.mr_frontend.fusion_mode=$mode"
  fi

  echo "[eval] $exp  $(basename "$best")"
  # shellcheck disable=SC2086
  python run_eval.py model="$stem" $margs model.bn_norm=BN \
    ckpt_path="$best" split="$SPLIT" seed="$seed" pool_workers="$POOL_WORKERS" \
    logger=[] $overlap_arg hydra.run.dir="$out_dir"
done

echo "评估完成，汇总：python scripts/summarize.py --eval_dir $LOG_DIR/eval_${SPLIT}"
