# -*- coding: utf-8 -*-
"""血統(種牡馬)ベースの候補シグナル用に、**時点参照(point-in-time)で安全な**父系統適性の
勝率テーブルを作る。

## リーク防止の設計
jra_sire_aptitude_profile_2026_08_30.py(記述的プロファイル)は2024年〜2026年8月の全期間を
母集団にしており、そのまま予想シグナルとして使うと「そのレース自身の結果を含む統計量で
そのレースを予想する」という未来情報リークになる(NAR側で過去に指摘された「LOBO学習フォールド
限定構築が必須、キャッシュがフォールド依存になり静かなリークを生みうる」という懸念
[[project_nar_box3_remodel_finding]]と同種の問題)。

これを回避するため、各(種牡馬, 区分値)の勝率を**そのレースの開催日より厳密に前の日付の
レースだけ**から計算する(将来を一切参照しない、真の時点参照統計)。実装は「日付×種牡馬×区分値
ごとの日次集計→日付順にcumsumしてから自分の日の分を引く」方式(O(n log n))。

## 4区分
1. distance: 距離帯(短距離<=1400/マイル1401-1800/中距離1801-2200/長距離2201+)
2. surface: 芝/ダート
3. going: 馬場状態(良/稍重/重/不良)
4. course: 競馬場

出力: data/jra_pipeline/pedigree_sire_features_cache.csv
列: race_id, horse_id, ped_sire_{distance,surface,going,course}_win_rate/_runs
(runsは「そのレースより前」の該当区分での産駒延べ出走数。0ならrateはNaN)

新馬・未勝利を含む全レースを入力に使う(過去の集計対象として除外する理由が無い、
jra_dataset.py側の除外はシグナル探索の母集団選定の話であり、統計を作る側とは別の関心事)。
"""
import glob
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.netkeiba_pipeline.storage.paths import load_all_pedigree  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "data" / "race_results"
OUT_PATH = PROJECT_ROOT / "data" / "jra_pipeline" / "pedigree_sire_features_cache.csv"

DISTANCE_BUCKETS = [
    (0, 1400, "短距離"),
    (1401, 1800, "マイル"),
    (1801, 2200, "中距離"),
    (2201, 99999, "長距離"),
]


def distance_bucket(d: float) -> str:
    for lo, hi, label in DISTANCE_BUCKETS:
        if lo <= d <= hi:
            return label
    return None


def build_before_stats(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    """df: sire_id, race_date, is_win, {category_col} の列を持つ全出走行。
    戻り値: sire_id, {category_col}, race_date, wins_before, runs_before
    (そのsire_id×category値×race_dateの組について、race_dateより厳密に前の日付だけの
    累積勝ち数・累積出走数)。"""
    daily = (
        df.groupby(["sire_id", category_col, "race_date"])
        .agg(wins=("is_win", "sum"), runs=("is_win", "size"))
        .reset_index()
        .sort_values(["sire_id", category_col, "race_date"])
    )
    grp = daily.groupby(["sire_id", category_col])
    daily["cum_wins"] = grp["wins"].cumsum()
    daily["cum_runs"] = grp["runs"].cumsum()
    daily["wins_before"] = daily["cum_wins"] - daily["wins"]
    daily["runs_before"] = daily["cum_runs"] - daily["runs"]
    return daily[["sire_id", category_col, "race_date", "wins_before", "runs_before"]]


def main() -> None:
    print("血統データロード中...")
    ped = load_all_pedigree()
    sire_map = ped[["horse_id", "ped_S_horse_id"]].rename(columns={"ped_S_horse_id": "sire_id"})
    sire_map = sire_map.dropna(subset=["sire_id"])
    print(f"pedigreeデータ: {len(ped)}頭(うち父horse_id判明: {len(sire_map)}頭)")

    print("race_resultsロード中(2024-2026年8月、JRA、除外なし)...")
    frames = []
    for year in ("2024", "2025", "2026"):
        for p in sorted(glob.glob(str(RESULTS_DIR / year / "*.csv"))):
            frames.append(pd.read_csv(p, dtype=str, encoding="utf-8"))
    results = pd.concat(frames, ignore_index=True)
    print(f"race_results総行数: {len(results)}")

    results["finish_pos_num"] = pd.to_numeric(results["finish_pos"], errors="coerce")
    results = results.dropna(subset=["finish_pos_num"])
    results["is_win"] = (results["finish_pos_num"] == 1).astype(int)
    results["distance_m_num"] = pd.to_numeric(results["distance_m"], errors="coerce")
    results["distance_bucket"] = results["distance_m_num"].map(distance_bucket)

    merged = results.merge(sire_map, on="horse_id", how="inner")
    print(f"pedigree結合成功: {len(merged)}行 / {merged['sire_id'].nunique()}種牡馬")

    dims = {
        "distance": "distance_bucket",
        "surface": "surface",
        "going": "going",
        "course": "racecourse",
    }

    out = merged[["race_id", "horse_id", "sire_id", "race_date"]].drop_duplicates(
        subset=["race_id", "horse_id"]
    )

    for dim_name, col in dims.items():
        print(f"時点参照統計を計算中: {dim_name} ({col})...")
        before = build_before_stats(merged[["sire_id", "race_date", "is_win", col]].rename(
            columns={col: "cat"}
        ), "cat")
        before = before.rename(columns={"cat": col})
        rate_col = f"ped_sire_{dim_name}_win_rate"
        runs_col = f"ped_sire_{dim_name}_runs"
        before[rate_col] = (before["wins_before"] / before["runs_before"]).where(before["runs_before"] > 0)
        before[runs_col] = before["runs_before"]

        # このレースの区分値(distance_bucket/surface/going/racecourse)自体をoutに付与してから
        # (sire_id, 区分値, race_date)でjoinする。
        cat_vals = merged[["race_id", "horse_id", col]].drop_duplicates(subset=["race_id", "horse_id"])
        out = out.merge(cat_vals, on=["race_id", "horse_id"], how="left")
        out = out.merge(
            before[["sire_id", col, "race_date", rate_col, runs_col]],
            on=["sire_id", col, "race_date"], how="left",
        )
        out = out.drop(columns=[col])

    out_cols = ["race_id", "horse_id"] + [
        c for dim in dims for c in (f"ped_sire_{dim}_win_rate", f"ped_sire_{dim}_runs")
    ]
    out = out[out_cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"\nwrote {OUT_PATH} ({len(out)}行)")
    for dim in dims:
        col = f"ped_sire_{dim}_runs"
        nonzero = (out[col].fillna(0) > 0).sum()
        print(f"  {dim}: 事前実績あり(runs>0) {nonzero}/{len(out)} ({nonzero/len(out)*100:.1f}%)")


if __name__ == "__main__":
    main()
