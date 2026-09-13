# -*- coding: utf-8 -*-
"""発走前(pending)レースの単勝オッズ推移パネルに添える参考データを1枚のCSVにまとめる
サイドカー(2026-08-29、ユーザー依頼)。

対象は「33ラップ理論」「凡走予測モデル(K=3〜8参考)」「予想印・展開データ」の3種。
いずれも過去の検証(jra_stage2_review_log_2026_08_28.json・jra_lap33関連レポート)で
本番モデルへの採用は見送られている(統計的に有意な市場超過を示せなかった)ため、
predict_pattern29.py・jra_signals.py・winner_v3.json 等の本番ファイルは一切改造せず、
このスクリプトはあくまで「参考表示専用」の値をbuild_artifact.pyへ渡すだけに留める。
_score/pred_rank(本番の予想順位)には一切影響しない。

- 33ラップ理論: jra_lap33_signals.py(既存、無改造)をそのまま呼び出し、型スコア
  (瞬発力型/持久力型の度合い)・今回コースの理論参照値・適合度(lap33_fit)を出す。
- 凡走予測モデル(K=3〜8参考): NARで検証・不採用となったnar_bottom_signals.pyの設計
  思想(=既存シグナルの向き反転流用が本体)をJRAへ移植したもの。JRA固有の新規4シグナル
  (pace_clash_risk等、NARのnewspaperスキーマ依存)は移植せず、jra_signals.ALL_SIGNALS_V4
  (既に本番score算出に使っている40本、mark/corner系のV4含む)を等重みでcombineし、
  値が低い(=「良い」方向のシグナルが弱い)馬ほど凡走リスクが高いとみなして順位付けする。
  K=3〜8という呼称は、この凡走リスク順位が1〜8位の馬について「何頭を径Kのフォーメーション/
  BOXから除外候補とみなすか」を読者が自分で使い分けられるようにするための表示上の目安であり、
  固定のK別モデルが8個あるわけではない(1本の順位表をK=3〜8の各カットオフで読み替える方式)。
- 予想印・展開データ: newspaper CSVのmark_honshi/mark_cp/mark_other(◎○▲等)と
  corner3/4_rank・corner3/4_gap_lengths・corner4_speedupを生値のまま転記する
  (jra_signals.pyのCANDIDATE_SIGNALS_V4は0..1正規化済みスコアだが、参考表示としては
  人間が読める生値の方が有用なため、ここでは正規化前のnewspaper CSV列を直接使う)。

出力: {SCRATCHPAD}/reference_panel_{date}.csv
  race_id, umaban, horse_id, horse_name, n_field,
  flop_score(0..1、高いほど凡走しにくい=combine_signals生値), flop_rank(1=最も凡走リスク高),
  lap33_type_score, lap33_course_ref, lap33_fit(0..1、コース理論値との適合度),
  mark_honshi, mark_cp, mark_other,
  corner3_rank, corner3_gap_lengths, corner4_rank, corner4_gap_lengths, corner4_speedup

対象: 当日の全レース(新馬・未勝利・障害含む)。新馬(初出走)は過去走が無いためflop_score・
lap33_type_scoreは自然にNaN/0.0になり(データ不足の明示、中立値としての0.0を除きクラッシュは
しない)、build_artifact.py側で「-」「対象外」表示になる。予想印・展開データはモデル非依存の
newspaper生値なので全レースで等しく出せる。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
SCRATCHPAD = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)

sys.path.insert(0, str(LIB_DIR))
import jra_signals as JS  # noqa: E402
import jra_dataset as JD  # noqa: E402
import jra_lap33_signals as L33  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from src.netkeiba_pipeline.storage.paths import newspaper_csv_path  # noqa: E402

RAW_MARK_CORNER_COLS = [
    "mark_honshi", "mark_cp", "mark_other",
    "corner3_rank", "corner3_gap_lengths",
    "corner4_rank", "corner4_gap_lengths", "corner4_speedup",
]


def _col_or_nan(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def load_pending_races(date: str) -> list:
    """race_names_{date}.csv(pending日の対戦表)+ newspaper CSVから、jra_dataset.build()と
    近い形の races リスト({"race_id","kaisai_date","racecourse","race_name","df",...})を作る。
    jra_dataset.build()と異なり新馬・未勝利・障害も除外しない(このスクリプトは参考表示専用で
    本番母集団ではないため、全レース対象にできる)。"""
    meta_path = SCRATCHPAD / f"race_names_{date}.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} が見つかりません(先に該当日のpredict_pattern29.pyを実行)。")
    meta = pd.read_csv(meta_path, dtype=str).drop_duplicates("race_id")
    # 2026-08-29、ユーザーから「各レースの単勝オッズ推移の右側に表示」と明示依頼があったため、
    # 新馬・未勝利・障害も含め全レースを対象にする(初出走馬は過去走が無くcompute_signals/
    # horse_type_scoreが自然にNaN・0.0を返し、build_artifact.py側で「-」「対象外」表示になる
    # だけで、計算自体はクラッシュしない。予想印・展開データは元々モデル非依存の生値なので
    # 全レースで問題なく出せる)。

    races = []
    for _, row in meta.iterrows():
        path = newspaper_csv_path(row["race_id"])
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
        df = JS._drop_scratched(df)
        if df.empty:
            continue
        races.append({
            "race_id": row["race_id"], "kaisai_date": date, "racecourse": row["racecourse"],
            "race_name": row["race_name"], "df": df,
            "surface": row.get("surface"), "distance_m": row.get("distance_m"),
        })
    return races


def build_reference_panel(date: str, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print("母集団・priors読み込み中(初回は時間がかかる)...")
    hist = JD.load(rebuild=False)
    priors_v4 = JS.make_priors([r["df"] for r in hist["races"]])
    weights_equal = {n: 1.0 for n in JS.ALL_SIGNALS_V4}

    lap33_lookup = L33.load_lap33_lookup()
    history_index = L33.build_history_index()

    races = load_pending_races(date)
    if verbose:
        print(f"対象レース: {len(races)}件(全レース、{date})")

    iso_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    out_rows = []
    for r in races:
        df = r["df"]
        current_class = JS._class_ordinal(r["race_name"], JS.CLASS_ORDINAL)
        sig = JS.compute_signals(df, current_class, priors_v4, JS.CLASS_ORDINAL)
        flop_score = JS.combine_signals(sig, weights_equal)
        # 値が低い(=良い方向シグナルが弱い)ほど凡走リスク高 -> 昇順rankで1位が最もリスク高
        flop_rank = flop_score.rank(method="min", ascending=True, na_option="bottom")
        n_field = len(df)

        surface = r.get("surface")
        distance_m = r.get("distance_m")
        ref = None
        if pd.notna(surface) and pd.notna(distance_m) and str(distance_m).strip() not in ("", "nan"):
            try:
                ref = L33.theory_lookup(r["racecourse"], surface, int(float(distance_m)))
            except (TypeError, ValueError):
                ref = None

        type_scores = pd.Series(
            [L33.horse_type_score(hid, iso_date, history_index, lap33_lookup, L33.N_LOOKBACK)
             for hid in df["horse_id"].astype(str)],
            index=df.index,
        )
        fit = pd.Series(np.nan, index=df.index) if ref is None else JS._minmax(type_scores * ref)

        raw_cols = {name: _col_or_nan(df, name) for name in RAW_MARK_CORNER_COLS}

        for idx in df.index:
            out_rows.append({
                "race_id": r["race_id"],
                "umaban": df.at[idx, "umaban"],
                "horse_id": df.at[idx, "horse_id"],
                "horse_name": df.at[idx, "horse_name"],
                "n_field": n_field,
                "flop_score": flop_score.at[idx],
                "flop_rank": flop_rank.at[idx],
                "lap33_type_score": type_scores.at[idx],
                "lap33_course_ref": ref,
                "lap33_fit": fit.at[idx],
                **{name: raw_cols[name].at[idx] for name in RAW_MARK_CORNER_COLS},
            })

    return pd.DataFrame(out_rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYYMMDD(pending日、race_names_{date}.csvが必要)")
    args = parser.parse_args()

    out = build_reference_panel(args.date)
    out_path = SCRATCHPAD / f"reference_panel_{args.date}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"wrote {out_path} ({len(out)} rows, {out['race_id'].nunique()} races)")
    n_ref = out.drop_duplicates("race_id")["lap33_course_ref"].notna().sum()
    print(f"33ラップ理論参照値あり: {n_ref}/{out['race_id'].nunique()}レース")
    n_flagged = (out["flop_rank"] <= 8).sum()
    print(f"凡走予測 K<=8警戒フラグ該当: {n_flagged}/{len(out)}頭")
