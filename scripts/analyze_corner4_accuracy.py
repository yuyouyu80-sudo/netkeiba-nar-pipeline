"""JRAの結果データ(passing_order/time/last_3f)から、展開予想(AI展開)のcorner4_rank/
corner4_gap_lengthsとの精度(ずれ)を検証し、data/mark_analysis/配下に保存する。

- 実際の4コーナー通過順位: passing_order(ハイフン区切り)の最後の数字。
- 実際の4コーナー通過時の馬身差(近似): time(走破タイム) - last_3f(後3Fタイム) を
  「残り3F地点通過時の経過タイム」とみなし、レース内最速馬との差を1秒=5馬身で馬身換算した
  近似値。JRAのほとんどのコースは直線(ホームストレッチ)が600m未満のため、この「残り3F地点」は
  物理的には4コーナー途中〜手前になることが多く、厳密には4コーナー通過地点と完全には一致しない
  近似値である点に注意。

このスクリプトはネットワークアクセスなし、ローカルCSVの読み込みのみ。実際の予想モデルへの
組み込み・重み付け判断は「予想用」セッションの領分。

使い方:
    python scripts/analyze_corner4_accuracy.py
"""
import glob
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "mark_analysis"

# 1馬身 ≈ 0.2秒という競馬の一般的な換算則(要検証の近似値)。
SECONDS_TO_LENGTHS = 5.0


def _parse_time(s) -> float | None:
    if pd.isna(s):
        return None
    s = str(s).strip()
    if ":" in s:
        m, rest = s.split(":", 1)
        return int(m) * 60 + float(rest)
    try:
        return float(s)
    except ValueError:
        return None


def _last_corner(s) -> int | None:
    if pd.isna(s):
        return None
    parts = [p for p in str(s).split("-") if p.strip() != ""]
    if not parts:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


def load_actual_corner4() -> pd.DataFrame:
    result_files = glob.glob(str(REPO_ROOT / "data" / "race_results" / "20*" / "*.csv"))  # JRAのみ
    frames = [
        pd.read_csv(f, dtype=str, usecols=["race_id", "umaban", "time", "last_3f", "passing_order"])
        for f in result_files
    ]
    res = pd.concat(frames, ignore_index=True).drop_duplicates(["race_id", "umaban"])

    res["time_s"] = res["time"].apply(_parse_time)
    res["last_3f_f"] = pd.to_numeric(res["last_3f"], errors="coerce")
    res["actual_corner4_rank"] = res["passing_order"].apply(_last_corner)
    res = res.dropna(subset=["time_s", "last_3f_f", "actual_corner4_rank"])

    res["time_to_3f"] = res["time_s"] - res["last_3f_f"]
    res["min_time_to_3f"] = res.groupby("race_id")["time_to_3f"].transform("min")
    res["gap_seconds"] = res["time_to_3f"] - res["min_time_to_3f"]
    res["actual_corner4_gap_lengths"] = (res["gap_seconds"] * SECONDS_TO_LENGTHS).round(2)
    res["umaban"] = res["umaban"].astype(str)

    return res[["race_id", "umaban", "actual_corner4_rank", "actual_corner4_gap_lengths"]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    actual = load_actual_corner4()

    pred = pd.read_csv(REPO_ROOT / "data" / "mark_analysis" / "corner4_result_joined.csv", dtype=str)
    pred["umaban"] = pred["umaban"].astype(str)
    pred["corner4_rank"] = pd.to_numeric(pred["corner4_rank"], errors="coerce")
    pred["corner4_gap_lengths"] = pd.to_numeric(pred["corner4_gap_lengths"], errors="coerce")

    merged = pred.merge(actual, on=["race_id", "umaban"], how="inner")
    merged = merged.dropna(subset=["corner4_rank", "actual_corner4_rank"])

    merged["rank_diff"] = merged["actual_corner4_rank"] - merged["corner4_rank"]
    merged["rank_abs_diff"] = merged["rank_diff"].abs()

    has_gap = merged.dropna(subset=["corner4_gap_lengths", "actual_corner4_gap_lengths"]).copy()
    has_gap["gap_diff_lengths"] = has_gap["actual_corner4_gap_lengths"] - has_gap["corner4_gap_lengths"]
    has_gap["gap_abs_diff_lengths"] = has_gap["gap_diff_lengths"].abs()

    out_cols_rank = [
        "race_id", "umaban", "racecourse", "surface", "running_style",
        "corner4_rank", "actual_corner4_rank", "rank_diff", "rank_abs_diff",
    ]
    merged[out_cols_rank].to_csv(OUT_DIR / "corner4_accuracy_rank.csv", index=False, encoding="utf-8")

    out_cols_gap = out_cols_rank + [
        "corner4_gap_lengths", "actual_corner4_gap_lengths", "gap_diff_lengths", "gap_abs_diff_lengths",
    ]
    has_gap[out_cols_gap].to_csv(OUT_DIR / "corner4_accuracy_gap.csv", index=False, encoding="utf-8")

    print(f"rank comparison: n={len(merged)}")
    print(f"gap comparison: n={len(has_gap)}")
    print(f"saved: {OUT_DIR / 'corner4_accuracy_rank.csv'}")
    print(f"saved: {OUT_DIR / 'corner4_accuracy_gap.csv'}")


if __name__ == "__main__":
    main()
