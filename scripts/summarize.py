"""
汇总 scripts/eval_all.sh 的输出：均值表、逐曲目配对比较、融合权重统计

用法：
    python scripts/summarize.py --eval_dir $LOG_DIR/eval_test [--out_dir results]

统计规则（在看测试结果前已固定）：
- 每首曲目先在各种子上取平均，再在曲目层面做配对比较，避免把种子当作独立样本
- 95% 置信区间：配对差均值的 bootstrap（10000 次，按曲目重采样）
- 显著性：Wilcoxon 符号秩检验；同一文件内的全部比较一起做 Holm 校正
- 效应量：配对差中位数（dB）与 matched-pairs rank-biserial
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:  # 没有 scipy 时只输出均值和置信区间
    wilcoxon = None

# (A, B, 回答的问题)：报告 A − B
COMPARISONS = [
    ("fixed", "baseline", "多分辨率前端是否有效"),
    ("mid_only", "baseline", "额外容量本身的收益"),
    ("fixed", "mid_only", "扣除容量后多分辨率信息是否有效"),
    ("learned_static", "fixed", "学习全局分辨率偏好是否有效"),
    ("learned_static", "baseline", "静态学习式融合的总体收益"),
    ("dynamic", "learned_static", "输入相关动态权重是否必要"),
    ("dynamic", "baseline", "完整方法的总体收益"),
]


def load(eval_dir: Path, name: str) -> pd.DataFrame:
    files = sorted(eval_dir.glob(f"*/{name}"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def bootstrap_ci(d: np.ndarray, n: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return np.percentile(means, [2.5, 97.5])


def rank_biserial(d: np.ndarray) -> float:
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m = len(p)
    adj, running = {}, 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adj[idx] = running
    return pd.Series(adj).reindex(p.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, type=Path)
    ap.add_argument("--out_dir", default="results", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tracks = load(args.eval_dir, "per_track.csv")
    if tracks.empty:
        raise SystemExit(f"{args.eval_dir} 下没有 per_track.csv")

    # 1. 每个模型 × 音源：种子间均值与标准差（先对曲目取平均）
    per_seed = tracks.groupby(["stem", "model", "seed"])["uSDR"].mean().reset_index()
    summary = per_seed.groupby(["stem", "model"])["uSDR"].agg(["mean", "std", "count"])
    summary = summary.rename(columns={"count": "n_seeds"}).round(3)
    summary.to_csv(args.out_dir / "summary_by_stem.csv")
    print("\n== uSDR（曲目平均后，按种子统计）==")
    print(summary.to_string())

    # 2. 每首曲目在种子上取平均，用于配对比较
    per_track = tracks.groupby(["stem", "model", "track"])["uSDR"].mean().unstack("model")

    rows = []
    for stem, df in per_track.groupby(level="stem"):
        for a, b, question in COMPARISONS:
            if a not in df or b not in df:
                continue
            d = (df[a] - df[b]).dropna().to_numpy()
            if len(d) == 0:
                continue
            lo, hi = bootstrap_ci(d)
            p = wilcoxon(d).pvalue if wilcoxon is not None and np.any(d != 0) else np.nan
            rows.append({
                "stem": stem, "comparison": f"{a} - {b}", "question": question,
                "n_tracks": len(d), "mean_diff": d.mean(), "median_diff": np.median(d),
                "ci95_low": lo, "ci95_high": hi, "p_wilcoxon": p,
                "rank_biserial": rank_biserial(d),
            })
    comp = pd.DataFrame(rows)
    if not comp.empty:
        comp["p_holm"] = holm(comp["p_wilcoxon"].fillna(1.0))
        comp = comp.round(4)
        comp.to_csv(args.out_dir / "paired_comparisons.csv", index=False)
        print("\n== 逐曲目配对比较（A − B，dB）==")
        print(comp.drop(columns="question").to_string(index=False))

    # 3. 融合权重
    w = load(args.eval_dir, "fusion_weights.csv")
    if not w.empty:
        cols = ["alpha_short", "alpha_mid", "alpha_long"]
        a = w[cols].clip(lower=1e-12)
        w["entropy"] = -(a * np.log(a)).sum(1) / np.log(3)   # 1 = 均匀，0 = 完全塌缩

        stem_stats = w.groupby(["stem", "model"])[cols + ["entropy", "fusion_scale"]].agg(["mean", "std"]).round(4)
        stem_stats.to_csv(args.out_dir / "fusion_weights_by_stem.csv")
        track_stats = w.groupby(["stem", "model", "seed", "track"])[cols].agg(["mean", "std"]).round(4)
        track_stats.to_csv(args.out_dir / "fusion_weights_by_track.csv")

        # 片段间变化量：同一曲目内相邻片段权重的平均 L1 变化
        w = w.sort_values(["stem", "model", "seed", "track", "chunk"])
        delta = w.groupby(["stem", "model", "seed", "track"])[cols].diff().abs().sum(1)
        w["chunk_delta_l1"] = delta
        var = w.groupby(["stem", "model"])["chunk_delta_l1"].mean().round(4)
        var.to_csv(args.out_dir / "fusion_weights_chunk_variation.csv")

        print("\n== 融合权重（按音源）==")
        print(stem_stats.to_string())
        print("\n== 相邻片段权重平均 L1 变化 ==")
        print(var.to_string())

    print(f"\n结果已写入 {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
