"""Select the validation-best checkpoint for one run.

The training config keeps every validation checkpoint. TensorBoard logs the
validation scalar at step N-1 while Lightning names the checkpoint step N, so
the helper maps that convention and copies the selected file to eval_best.ckpt.
"""
from pathlib import Path
import shutil
import sys
import glob

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: select_best_checkpoint.py RUN_DIR")
    run = Path(sys.argv[1])
    events = glob.glob(str(run / "tensorboard" / "**" / "events.out.tfevents.*"), recursive=True)
    if not events:
        raise SystemExit(f"no TensorBoard event file in {run}")
    acc = EventAccumulator(events[0])
    acc.Reload()
    vals = acc.Scalars("val/usdr")
    if not vals:
        raise SystemExit(f"no val/usdr values in {run}")
    best = max(vals, key=lambda x: x.value)
    step = int(best.step) + 1
    candidates = sorted((run / "checkpoints").glob(f"*step={step}.ckpt"))
    if not candidates:
        raise SystemExit(f"checkpoint for best validation step {step} not found")
    src = candidates[0]
    dst = run / "checkpoints" / "eval_best.ckpt"
    shutil.copy2(src, dst)
    print(f"{src}\t{float(best.value):.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
