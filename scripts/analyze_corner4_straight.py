"""4コーナー位置データ(corner4_rank/corner4_gap_lengths/corner4_speedup)と、
コース固有の最終直線特性(直線距離・上り坂の有無)、各馬の脚質(ca_running_style_category_label)
を突き合わせて data/mark_analysis/ 配下に保存する。

直線距離はJRA公式発表の代表値(内外回りの区別なし、近似値)。上り坂の有無は正式な高低差データ
ではなく、コース紹介で一般的に言われる特徴に基づく粗い二値分類(要検証)。
現時点ではJRAの中京・函館・小倉・新潟・札幌・福島のみデータが存在する(2026年夏開催分)。
NARは信頼できる直線距離データを持っていないため対象外。

このスクリプトはネットワークアクセスなし、ローカルCSVの読み込みのみ。実際の予想モデルへの
組み込み・重み付け判断は「予想用」セッションの領分。

使い方:
    python scripts/analyze_corner4_straight.py
"""
import glob
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "mark_analysis"

# JRA公式直線距離(m)の代表値。内外回りがある場合は主に使われる方の値を採用した近似値。
# ※要検証・概数
STRAIGHT_M = {
    ("札幌", "芝"): 266.1, ("札幌", "ダ"): 264.3,
    ("函館", "芝"): 262.1, ("函館", "ダ"): 260.3,
    ("福島", "芝"): 292.0, ("福島", "ダ"): 295.7,
    ("新潟", "芝"): 358.7, ("新潟", "ダ"): 353.9,
    ("東京", "芝"): 525.9, ("東京", "ダ"): 501.6,
    ("中山", "芝"): 310.0, ("中山", "ダ"): 308.0,
    ("中京", "芝"): 412.5, ("中京", "ダ"): 410.7,
    ("京都", "芝"): 328.4, ("京都", "ダ"): 329.1,
    ("阪神", "芝"): 356.5, ("阪神", "ダ"): 352.7,
    ("小倉", "芝"): 293.0, ("小倉", "ダ"): 291.3,
}

# 直線に明確な上り坂特徴があるとされるコース(粗い二値、要検証・一般的なコース紹介ベース)
UPHILL_COURSES = {"中京", "小倉", "中山", "阪神", "東京"}


def load_course_features() -> pd.DataFrame:
    rows = []
    for (course, surface), straight_m in STRAIGHT_M.items():
        rows.append(
            {
                "racecourse": course,
                "surface": surface,
                "straight_m": straight_m,
                "uphill": "上りあり" if course in UPHILL_COURSES else "平坦",
            }
        )
    return pd.DataFrame(rows)


def load_results() -> pd.DataFrame:
    result_files = glob.glob(str(REPO_ROOT / "data" / "race_results" / "20*" / "*.csv"))  # JRAのみ
    frames = [
        pd.read_csv(f, dtype=str, usecols=["race_id", "umaban", "finish_pos", "racecourse", "surface"])
        for f in result_files
    ]
    results = pd.concat(frames, ignore_index=True).drop_duplicates(["race_id", "umaban"])
    results["finish_pos_num"] = pd.to_numeric(results["finish_pos"], errors="coerce")
    results = results.dropna(subset=["finish_pos_num"])
    results["finish_pos_num"] = results["finish_pos_num"].astype(int)
    return results


def load_corner4_and_style() -> pd.DataFrame:
    mark_files = glob.glob(str(REPO_ROOT / "data" / "newspaper" / "*.csv"))
    frames = []
    for f in mark_files:
        cols = pd.read_csv(f, dtype=str, nrows=0).columns
        needed = [
            c
            for c in (
                "race_id",
                "umaban",
                "corner4_rank",
                "corner4_gap_lengths",
                "corner4_speedup",
                "ca_running_style_category_label",
            )
            if c in cols
        ]
        if len(needed) < 6:
            continue
        df = pd.read_csv(f, dtype=str, usecols=needed, keep_default_na=False)
        if "race_id" not in df.columns:
            df["race_id"] = Path(f).stem
        frames.append(df)
    c4 = pd.concat(frames, ignore_index=True)
    c4 = c4[(c4["corner4_rank"] != "") & (c4["corner4_gap_lengths"] != "")]
    c4 = c4.rename(columns={"ca_running_style_category_label": "running_style"})
    c4["running_style"] = c4["running_style"].replace("", "不明")
    return c4


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    course_features = load_course_features()
    course_features.to_csv(OUT_DIR / "course_straight_and_uphill.csv", index=False, encoding="utf-8")

    results = load_results()
    c4 = load_corner4_and_style()

    merged = c4.merge(results, on=["race_id", "umaban"], how="inner")
    for col in ("corner4_rank", "corner4_gap_lengths", "corner4_speedup"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.dropna(subset=["corner4_rank", "corner4_gap_lengths", "corner4_speedup", "finish_pos_num"])

    merged = merged.merge(course_features, on=["racecourse", "surface"], how="inner")

    merged["rank_change"] = merged["corner4_rank"] - merged["finish_pos_num"]
    merged["win"] = (merged["finish_pos_num"] == 1).astype(int)
    merged["place2"] = (merged["finish_pos_num"] <= 2).astype(int)
    merged["place3"] = (merged["finish_pos_num"] <= 3).astype(int)

    out_cols = [
        "race_id",
        "umaban",
        "racecourse",
        "surface",
        "straight_m",
        "uphill",
        "running_style",
        "corner4_rank",
        "corner4_gap_lengths",
        "corner4_speedup",
        "finish_pos_num",
        "rank_change",
        "win",
        "place2",
        "place3",
    ]
    merged[out_cols].to_csv(OUT_DIR / "corner4_result_joined.csv", index=False, encoding="utf-8")

    print(f"対象: {len(merged)}頭")
    print(f"saved: {OUT_DIR / 'course_straight_and_uphill.csv'}")
    print(f"saved: {OUT_DIR / 'corner4_result_joined.csv'}")


if __name__ == "__main__":
    main()
