#!/usr/bin/env bash
# 对 finetune_all.sh 产生的每个实验，用 val/usdr 最高的 checkpoint 在测试集上评估一次。
# 已评估过的实验会自动跳过。全部训练完成、规则冻结之后再运行。
#
# 用法（在项目根目录）：
#   bash scripts/eval_all.sh
#
# 可选环境变量：
#   STEMS / MODES / SEEDS  与 finetune_all.sh 使用相同的实验组合（默认值也相同）
#   OVERLAP_ADD (off)  设为 on 时使用 overlap-add（慢约 4 倍）；所有模型必须一致
#   SPLIT (test)       调试时可设为 valid
#   POOL_WORKERS (8)   计算指标的 CPU 进程数

set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; source .env; set +a; fi

: "${LOG_DIR:?请在 .env 中设置 LOG_DIR}"
STEMS=${STEMS:-"drums bass"}
MODES=${MODES:-"baseline learned_static dynamic"}
SEEDS=${SEEDS:-"2021 2022"}
OVERLAP_ADD=${OVERLAP_ADD:-off}
SPLIT=${SPLIT:-test}
POOL_WORKERS=${POOL_WORKERS:-8}
# 主干尺寸必须与训练时一致；finetune_1gpu 使用官方尺寸（g=32、4 层 LSTM）
BACKBONE_ARGS=${BACKBONE_ARGS:-"model.g=32 model.bandsequence.num_layers=4"}

overlap_arg=""
missing=()
expected=0
[ "$OVERLAP_ADD" = "off" ] && overlap_arg="overlap_add=null"

for stem in $STEMS; do
  for seed in $SEEDS; do
    for mode in $MODES; do
      expected=$((expected + 1))
      exp="ft_${stem}_${mode}_s${seed}"
      run_dir="$LOG_DIR/$exp"
      out_dir="$LOG_DIR/eval_${SPLIT}/$exp"

      if [ ! -f "$run_dir/DONE" ]; then
        missing+=("$exp (训练未完成)")
        continue
      fi

      if [ -f "$out_dir/per_track.csv" ]; then
        echo "[skip] $exp 已评估"
        continue
      fi

      # 最优 checkpoint（save_top_k=1，排除 last.ckpt）
      best=$(ls "$run_dir"/checkpoints/*.ckpt 2>/dev/null | grep -v '/last.ckpt$' | head -n 1 || true)
      if [ -z "$best" ]; then
        missing+=("$exp (缺少最优 checkpoint)")
        continue
      fi

      if [ "$mode" = "baseline" ]; then
        margs="model.mr_frontend.enabled=false"
      else
        margs="model.mr_frontend.enabled=true model.mr_frontend.fusion_mode=$mode"
      fi

      echo "[eval] $exp  $(basename "$best")"
      # shellcheck disable=SC2086
      python run_eval.py model="$stem" $margs model.bn_norm=BN $BACKBONE_ARGS \
        ckpt_path="'$best'" split="$SPLIT" seed="$seed" pool_workers="$POOL_WORKERS" \
        logger=[] $overlap_arg hydra.run.dir="$out_dir"
    done
  done
done

if [ "$expected" -eq 0 ]; then
  echo "没有指定可评估的实验，请检查 STEMS、MODES 和 SEEDS"
  exit 1
fi

# 缺少任何一个预期实验时以非零状态退出，避免在不完整的结果上做配对比较
if [ ${#missing[@]} -gt 0 ]; then
  echo "以下预期实验尚不可评估，请先补跑："
  printf '  %s\n' "${missing[@]}"
  exit 1
fi

echo "评估完成，汇总：python scripts/summarize.py --eval_dir $LOG_DIR/eval_${SPLIT}"
