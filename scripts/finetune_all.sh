#!/usr/bin/env bash
# 单卡顺序跑完全部微调实验。可随时中断（包括断电），重新执行同一条命令即可：
#   - 已完成的实验跳过
#   - 未完成的实验从最近的续训 checkpoint（每 500 步保存一次）继续
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
#   NOTIFY_URL      失败、续训、完成时推送消息（HTTP POST 纯文本，例如 https://ntfy.sh/<你的主题>）
#   HEARTBEAT_URL   运行期间定时 ping（例如 healthchecks.io 的 ping 地址）；断电后 ping 停止，由外部服务报警
#   HEARTBEAT_EVERY 心跳间隔秒数（300）
#
# 预训练权重需命名为 $PRETRAINED_DIR/{stem}.ckpt，例如 drums.ckpt、bass.ckpt。
# 每个实验输出到 $LOG_DIR/ft_{stem}_{mode}_s{seed}/，完成后写入 DONE 文件。
# 所有开始、续训、完成、失败事件追加记录到 $LOG_DIR/run_history.tsv。

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
NOTIFY_URL=${NOTIFY_URL:-}
HEARTBEAT_URL=${HEARTBEAT_URL:-}
HEARTBEAT_EVERY=${HEARTBEAT_EVERY:-300}

mkdir -p "$LOG_DIR"
HISTORY="$LOG_DIR/run_history.tsv"
[ -f "$HISTORY" ] || printf 'time\tevent\texperiment\tdetail\n' > "$HISTORY"

record() {  # record <event> <experiment> [detail]
  printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" "${3:-}" >> "$HISTORY"
}

notify() {  # 打印并推送；推送失败不影响训练
  local msg="[DTTNet $(hostname)] $1"
  echo "$msg"
  if [ -n "$NOTIFY_URL" ]; then
    curl -fsS -m 10 -H "Title: DTTNet" -d "$msg" "$NOTIFY_URL" >/dev/null 2>&1 || true
  fi
}

ping_heartbeat() {  # ping_heartbeat [/fail]
  if [ -n "$HEARTBEAT_URL" ]; then
    curl -fsS -m 10 "${HEARTBEAT_URL}${1:-}" >/dev/null 2>&1 || true
  fi
}

# 心跳：脚本运行期间定时 ping；机器断电或死机时 ping 停止，由外部服务发出报警
HB_PID=""
if [ -n "$HEARTBEAT_URL" ]; then
  ( while true; do ping_heartbeat; sleep "$HEARTBEAT_EVERY"; done ) &
  HB_PID=$!
fi

CURRENT=""
on_exit() {
  local code=$?
  [ -n "$HB_PID" ] && kill "$HB_PID" 2>/dev/null || true
  if [ $code -ne 0 ] && [ -n "$CURRENT" ]; then
    record interrupted "$CURRENT" "exit code $code"
    notify "$CURRENT 中断（退出码 $code），重新执行脚本即可续训"
    ping_heartbeat /fail
  fi
}
trap on_exit EXIT
trap 'exit 130' INT TERM

mode_args() {
  if [ "$1" = "baseline" ]; then
    echo "model.mr_frontend.enabled=false"
  else
    echo "model.mr_frontend.enabled=true model.mr_frontend.fusion_mode=$1"
  fi
}

total=0; finished=0
for stem in $STEMS; do for seed in $SEEDS; do for mode in $MODES; do total=$((total + 1)); done; done; done

notify "开始批量训练：共 $total 个实验"
record batch_start "-" "stems=$STEMS modes=$MODES seeds=$SEEDS steps=$TRAIN_STEPS"

for stem in $STEMS; do
  ckpt="$PRETRAINED_DIR/$stem.ckpt"
  [ -f "$ckpt" ] || { notify "找不到预训练权重 $ckpt"; exit 1; }
  for seed in $SEEDS; do
    for mode in $MODES; do
      exp="ft_${stem}_${mode}_s${seed}"
      run_dir="$LOG_DIR/$exp"
      if [ -f "$run_dir/DONE" ]; then
        echo "[skip] $exp 已完成"
        finished=$((finished + 1))
        continue
      fi

      resume_arg=""
      if [ -d "$run_dir" ]; then
        # 上次该实验的最后一条记录是 start，说明脚本没来得及记录中断：多半是断电或强制关机
        last_event=$(awk -F'\t' -v e="$exp" '$3 == e {ev = $2} END {print ev}' "$HISTORY")
        if [ "$last_event" = "start" ]; then
          record unclean_stop "$exp" "上次运行没有正常结束记录，疑似断电或强制关机"
          notify "$exp 上次运行异常终止（疑似断电或强制关机）"
        fi
        last=$(ls -t "$run_dir"/resume/resume-*.ckpt 2>/dev/null | head -n 1 || true)
        if [ -n "$last" ]; then
          # 路径加引号，避免 Hydra 解析特殊字符
          resume_arg="resume_ckpt='$last'"
          record resume "$exp" "$(basename "$last")"
          notify "$exp 上次未完成，从 $(basename "$last") 续训"
        else
          failed="${run_dir}.failed_$(date '+%Y%m%d_%H%M%S')"
          record restart "$exp" "no resume checkpoint, moved to $(basename "$failed")"
          notify "$exp 上次未完成且没有续训 checkpoint，从头重跑"
          mv "$run_dir" "$failed"
        fi
      fi

      CURRENT="$exp"
      record start "$exp"
      echo "[run ] $exp  $(date '+%F %T')"
      # shellcheck disable=SC2046,SC2086
      python train.py experiment=finetune_1gpu datamodule=musdb_dev14 model="$stem" \
        pretrained_ckpt="$ckpt" seed="$seed" \
        train_steps="$TRAIN_STEPS" val_every_steps="$VAL_EVERY" \
        datamodule.batch_size="$BATCH" trainer.accumulate_grad_batches="$ACCUM" \
        exp_name="$exp" hydra.run.dir="$run_dir" \
        $(mode_args "$mode") $resume_arg $EXTRA
      date '+%F %T' > "$run_dir/DONE"
      CURRENT=""
      finished=$((finished + 1))
      record done "$exp"
      notify "$exp 完成（$finished/$total）"
    done
  done
done

record batch_done "-" "$finished/$total"
notify "全部训练完成（$finished/$total）。如使用心跳监控，请在服务端暂停该检查，避免之后误报"
