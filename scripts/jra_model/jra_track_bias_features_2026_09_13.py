# -*- coding: utf-8 -*-
"""「JRAデータ資産棚卸しレビュー」Phase3: 枠番ベースの**同日内トラックバイアス**シグナル用に、
時点参照(point-in-time)で安全な「開催日×競馬場×サーフェス×枠グループ」別の複勝率テーブルを作る。

## 背景・設計意図
`data/race_results`の実測列corner1〜4_rank/group_size/gap_len_est/is_leaderは、モデル・シグナル
計算コードから一切参照されていない(監査「JRAデータ資産棚卸し」B項目、469,779行・参照0件を
Explore agent 3本で再確認済み)。このうちcorner4系はレース中の実測位置そのものであり、
**予想対象レース自身の値は発走前に知りようがない**ため、対象レースの特徴量には使えない。
一方、「その日・その競馬場・そのサーフェスで、枠番のどのあたりが有利/不利だったか」という
トラックバイアスは、**同日ですでに終わった前のレースの実測結果から**推定でき、後続レースの
予想には発走前に使える(実務でも「今日は内枠が伸びる馬場」等と言われる現象そのもの)。

## リーク防止の設計
[[project_jra_pedigree_theory_verification_2026_08_29]]と同型の「日付順にcumsumしてから自分の
分を引く」方式を、**日付単位ではなくレース単位(race_number)** に変えて適用する。
同一(kaisai_date=race_date, racecourse, surface, waku_bucket)の組について、
**race_numberが厳密に小さい(=時刻的に前に確定している)レースだけ**を使って複勝率を集計する。
これにより、対象レースの発走時点で実際に入手可能な情報だけを使う(同日後続レースの結果を
混入させない)。当日の最初のレース(または該当条件の初出現)はruns_before=0となりNaN
(対象外)になるのが正しい挙動(career_form等の「初出走はNaN」と同じ設計)。

## 枠グループ
枠番(waku、1〜8)をそのまま使うと同日内サンプルが薄すぎるため(1レース1〜2頭)、
既存の「予想ファクター」の枠バケット定義(jra_factor_registry.pyの`waku`グループ、
1〜2枠=内/3〜6枠=中/7〜8枠=外)と同じ3区分に集約する。

## 対象外にした設計(将来の拡張候補、本スクリプトでは未実装)
corner4_rank/is_leader(4角の実際の通過順位)を使えば「今日は逃げ・先行馬が残る馬場か」という
別のペース系バイアス信号も理論上作れるが、対象馬自身の脚質(逃げ/先行/差し/追込)を発走前の
どの列と対応付けるかの設計が別途必要になるため、本Phaseのスコープ外とする(V2候補として
将来検討)。

出力: data/jra_pipeline/track_bias_features_cache.csv
列: race_id, horse_id, track_waku_bias_place3_rate, track_waku_bias_runs
(runsは「同日・同競馬場・同サーフェス・同枠グループで、このレースより前のrace_number」の
延べ出走数。0またはこの組み合わせの初出現ならrateはNaN)

新馬・未勝利を含む全レースを入力に使う(過去の集計対象として除外する理由が無い、
jra_pedigree_features_2026_08_30.pyと同じ方針)。
"""
import sys
from pathlib import Path

import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
sys.path.insert(0, str(LIB_DIR))
import jra_history as JH  # noqa: E402

OUT_PATH = PROJECT_ROOT / "data" / "jra_pipeline" / "track_bias_features_cache.csv"

# jra_factor_registry.pyの"waku"グループ(waku_1_2/waku_3_6/waku_7_8)と同一の区分。
WAKU_GROUPS = [
    ({1, 2}, "uchi"),
    ({3, 4, 5, 6}, "naka"),
    ({7, 8}, "soto"),
]


def waku_group(w) -> str:
    try:
        wi = int(w)
    except (TypeError, ValueError):
        return None
    for members, label in WAKU_GROUPS:
        if wi in members:
            return label
    return None


def build_before_stats(df: pd.DataFrame) -> pd.DataFrame:
    """df: race_date, racecourse, surface, waku_group, race_number, is_place3 の列を持つ
    全出走行。戻り値: race_date, racecourse, surface, waku_group, race_number,
    hits_before, runs_before (その(race_date, racecourse, surface, waku_group)の組について、
    race_numberより厳密に小さいレースだけの累積複勝数・出走数)。"""
    key = ["race_date", "racecourse", "surface", "waku_group"]
    daily = (
        df.groupby(key + ["race_number"])
        .agg(hits=("is_place3", "sum"), runs=("is_place3", "size"))
        .reset_index()
        .sort_values(key + ["race_number"])
    )
    grp = daily.groupby(key)
    daily["cum_hits"] = grp["hits"].cumsum()
    daily["cum_runs"] = grp["runs"].cumsum()
    daily["hits_before"] = daily["cum_hits"] - daily["hits"]
    daily["runs_before"] = daily["cum_runs"] - daily["runs"]
    return daily[key + ["race_number", "hits_before", "runs_before"]]


def main() -> None:
    print("race_resultsロード中(JH.load_results()、2016-2026年)...")
    results = JH.load_results()
    print(f"race_results総行数: {len(results)}")

    results = results.copy()
    results["finish_pos_num"] = pd.to_numeric(results["finish_pos"], errors="coerce")
    results["race_number_num"] = pd.to_numeric(results["race_number"], errors="coerce")
    results["waku_group"] = results["waku"].map(waku_group)
    before_drop = len(results)
    results = results.dropna(
        subset=["finish_pos_num", "race_number_num", "waku_group", "race_date", "racecourse", "surface"]
    )
    print(f"有効行(着順・レース番号・枠・日付/競馬場/サーフェス全て揃う): "
          f"{len(results)}/{before_drop}")
    results["race_number"] = results["race_number_num"].astype(int)
    results["is_place3"] = (results["finish_pos_num"] <= 3).astype(int)

    print("時点参照統計を計算中(開催日×競馬場×サーフェス×枠グループ、race_number順)...")
    before = build_before_stats(results)

    out = results[
        ["race_id", "horse_id", "race_date", "racecourse", "surface", "waku_group", "race_number"]
    ].drop_duplicates(subset=["race_id", "horse_id"])
    out = out.merge(
        before, on=["race_date", "racecourse", "surface", "waku_group", "race_number"], how="left"
    )
    out["track_waku_bias_place3_rate"] = (
        out["hits_before"] / out["runs_before"]
    ).where(out["runs_before"] > 0)
    out["track_waku_bias_runs"] = out["runs_before"]

    out_final = out[["race_id", "horse_id", "track_waku_bias_place3_rate", "track_waku_bias_runs"]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_final.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"\nwrote {OUT_PATH} ({len(out_final)}行)")

    nonzero = (out_final["track_waku_bias_runs"].fillna(0) > 0).sum()
    print(f"事前実績あり(runs>0): {nonzero}/{len(out_final)} ({nonzero/len(out_final)*100:.1f}%)")

    # race_number別のfill率(当日のレースが進むほど増えるはずという設計上の期待の確認用)
    by_num = out.groupby("race_number").apply(
        lambda g: (g["track_waku_bias_runs"].fillna(0) > 0).mean()
    )
    print("\nrace_number別fill率(当日の何R目かによる充足率の変化、増加傾向なら設計通り):")
    print(by_num.round(3).to_string())


if __name__ == "__main__":
    main()
