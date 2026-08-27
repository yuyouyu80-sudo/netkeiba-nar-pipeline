"""3コーナー→4コーナーの位置変化(順位変化・馬身差変化)と、確定着順(勝率・連対率・複勝率)の
関係を検証する。

corner3_rank/corner3_gap_lengths と corner4_rank/corner4_gap_lengths(いずれも newspaper CSV の
AI展開データ)を突き合わせ、「3角→4角でどれだけ順位を上げた/下げたか」「3角→4角で先頭との差が
どれだけ縮んだ/開いたか」を求め、それぞれの区分ごとに実際の勝率・連対率・複勝率を集計する。

JRA・NARの両方を対象(2026-08-27の3コーナーバックフィルでNARにも列がある)。

このスクリプトはネットワークアクセスなし、ローカルCSVの読み込みのみ。実際の予想モデルへの
組み込み・重み付け判断は「予想用」セッションの領分。

使い方:
    python scripts/analyze_corner34_transition.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "mark_analysis"

NEEDED_COLS = [
    "race_id",
    "umaban",
    "corner3_rank",
    "corner3_gap_lengths",
    "corner4_rank",
    "corner4_gap_lengths",
    "ca_running_style_category_label",
]


def load_corner34() -> pd.DataFrame:
    files = glob.glob(str(REPO_ROOT / "data" / "newspaper" / "*.csv")) + glob.glob(
        str(REPO_ROOT / "data" / "newspaper" / "nar" / "*.csv")
    )
    frames = []
    for f in files:
        cols = pd.read_csv(f, dtype=str, nrows=0).columns
        needed = [c for c in NEEDED_COLS if c in cols]
        if not {"race_id", "umaban", "corner3_rank", "corner4_rank"}.issubset(needed):
            continue
        df = pd.read_csv(f, dtype=str, usecols=needed, keep_default_na=False)
        if "race_id" not in df.columns:
            df["race_id"] = Path(f).stem
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out[(out["corner3_rank"] != "") & (out["corner4_rank"] != "")]
    out = out.rename(columns={"ca_running_style_category_label": "running_style"})
    if "running_style" in out.columns:
        out["running_style"] = out["running_style"].replace("", "不明")
    else:
        out["running_style"] = "不明"
    return out


def load_results() -> pd.DataFrame:
    jra_files = glob.glob(str(REPO_ROOT / "data" / "race_results" / "20*" / "*.csv"))
    nar_files = glob.glob(str(REPO_ROOT / "data" / "race_results" / "nar" / "20*" / "*.csv"))
    frames = [
        pd.read_csv(f, dtype=str, usecols=["race_id", "umaban", "finish_pos", "racecourse", "surface"])
        for f in jra_files + nar_files
    ]
    results = pd.concat(frames, ignore_index=True).drop_duplicates(["race_id", "umaban"])
    results["finish_pos_num"] = pd.to_numeric(results["finish_pos"], errors="coerce")
    results = results.dropna(subset=["finish_pos_num"])
    results["finish_pos_num"] = results["finish_pos_num"].astype(int)
    return results


def rank_change_bin(x: int) -> str:
    # 正=3角より4角で順位アップ(前へ進んだ)、負=順位ダウン(後退)
    if x >= 3:
        return "3位以上アップ"
    if x == 2:
        return "2位アップ"
    if x == 1:
        return "1位アップ"
    if x == 0:
        return "変化なし"
    if x == -1:
        return "1位ダウン"
    if x == -2:
        return "2位ダウン"
    return "3位以上ダウン"


RANK_CHANGE_ORDER = [
    "3位以上ダウン", "2位ダウン", "1位ダウン", "変化なし",
    "1位アップ", "2位アップ", "3位以上アップ",
]


def gap_change_bin(x: float) -> str:
    # 正=3角より4角で先頭との差が縮んだ(伸びている)、負=開いた(苦しくなっている)
    if x >= 2:
        return "2馬身以上縮小"
    if x >= 0.5:
        return "0.5〜2馬身縮小"
    if x > -0.5:
        return "ほぼ変化なし"
    if x > -2:
        return "0.5〜2馬身拡大"
    return "2馬身以上拡大"


GAP_CHANGE_ORDER = [
    "2馬身以上拡大", "0.5〜2馬身拡大", "ほぼ変化なし",
    "0.5〜2馬身縮小", "2馬身以上縮小",
]


def summarize(df: pd.DataFrame, group_col: str, order: list[str]) -> pd.DataFrame:
    g = df.groupby(group_col, observed=True)
    out = g.agg(
        n=("win", "size"),
        win_rate=("win", "mean"),
        place2_rate=("place2", "mean"),
        place3_rate=("place3", "mean"),
    ).reindex(order)
    out = out.dropna(subset=["n"])
    out["n"] = out["n"].astype(int)
    out = out.reset_index()
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    c34 = load_corner34()
    results = load_results()

    merged = c34.merge(results, on=["race_id", "umaban"], how="inner")
    for col in ("corner3_rank", "corner3_gap_lengths", "corner4_rank", "corner4_gap_lengths"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.dropna(subset=["corner3_rank", "corner4_rank", "finish_pos_num"])

    merged["rank_delta"] = merged["corner3_rank"] - merged["corner4_rank"]  # 正=前進
    merged["gap_delta"] = merged["corner3_gap_lengths"] - merged["corner4_gap_lengths"]  # 正=差が縮小

    merged["win"] = (merged["finish_pos_num"] == 1).astype(int)
    merged["place2"] = (merged["finish_pos_num"] <= 2).astype(int)
    merged["place3"] = (merged["finish_pos_num"] <= 3).astype(int)

    merged["rank_delta_bin"] = merged["rank_delta"].apply(rank_change_bin)
    merged["gap_delta_bin"] = merged["gap_delta"].dropna().apply(gap_change_bin)

    out_cols = [
        "race_id", "umaban", "racecourse", "surface", "running_style",
        "corner3_rank", "corner4_rank", "rank_delta", "rank_delta_bin",
        "corner3_gap_lengths", "corner4_gap_lengths", "gap_delta", "gap_delta_bin",
        "finish_pos_num", "win", "place2", "place3",
    ]
    merged[out_cols].to_csv(OUT_DIR / "corner34_transition_joined.csv", index=False, encoding="utf-8")

    rank_summary = summarize(merged, "rank_delta_bin", RANK_CHANGE_ORDER)
    rank_summary.to_csv(OUT_DIR / "corner34_rank_delta_hitrate.csv", index=False, encoding="utf-8")

    gap_df = merged.dropna(subset=["gap_delta_bin"])
    gap_summary = summarize(gap_df, "gap_delta_bin", GAP_CHANGE_ORDER)
    gap_summary.to_csv(OUT_DIR / "corner34_gap_delta_hitrate.csv", index=False, encoding="utf-8")

    # 脚質別クロス集計(順位変化 x 脚質)
    style_rank = (
        merged.groupby(["running_style", "rank_delta_bin"], observed=True)
        .agg(n=("win", "size"), win_rate=("win", "mean"), place2_rate=("place2", "mean"), place3_rate=("place3", "mean"))
        .reset_index()
    )
    style_rank.to_csv(OUT_DIR / "corner34_rank_delta_by_style.csv", index=False, encoding="utf-8")

    print(f"対象: {len(merged)}頭 (JRA+NAR)")
    print("\n[3角→4角 順位変化 x 成績]")
    print(rank_summary.to_string(index=False))
    print("\n[3角→4角 先頭との差の変化 x 成績]")
    print(gap_summary.to_string(index=False))
    print(f"\nsaved: {OUT_DIR / 'corner34_transition_joined.csv'}")
    print(f"saved: {OUT_DIR / 'corner34_rank_delta_hitrate.csv'}")
    print(f"saved: {OUT_DIR / 'corner34_gap_delta_hitrate.csv'}")
    print(f"saved: {OUT_DIR / 'corner34_rank_delta_by_style.csv'}")


if __name__ == "__main__":
    main()
