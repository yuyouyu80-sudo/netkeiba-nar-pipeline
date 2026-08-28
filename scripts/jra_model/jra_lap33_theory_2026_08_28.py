# -*- coding: utf-8 -*-
"""「33ラップ理論」(鈴木ショータ氏、2020年4月著、非公開個人PDF)の記述的検証(Part 1)。

理論の定義: 33ラップ = (残り1200m〜600m地点の3F) − (残り600m〜ゴールの3F=上がり3F)。
プラスが大きいほど「瞬発力勝負」、マイナスが大きいほど「持久力勝負」。

data/lap_times/(先頭馬基準200m区間タイム、race_result_parser.parse_lap_timesが抽出)を
race_id単位に集約し、race_resultsのdistance_m/surface/racecourseと結合して実測する。

区間の切り出しルール: 距離が200mの倍数でないレース(1900m等)は実データ確認済みで
「先頭(1本目)の区間が短い/長い」形で端数を吸収し、末尾側は常にクリーンな200m区間になる
(2025年データで実測: 1900m×2レースとも10区間・1区間目のみ他と明確に異なる長さ)。
よって「末尾3区間」「その直前3区間」を取れば、距離の端数に関わらず正しく上がり3F・
その前の3Fに対応する。distance_m<1200相当(区間数<6)は理論の適用外として除外する。

理論のPDF原本p4「コース別！平均33ラップ一覧表」の値は jra_lap33_theory_table.py
(このスクリプトと jra_lap33_signals.py の両方から参照される、定数のみの共有モジュール)
に集約している。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jra_lap33_theory_table import THEORY_TABLE, THEORY_TABLE_AMBIGUOUS_AVG, AMBIGUOUS_CELLS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LAP_DIR = PROJECT_ROOT / "data" / "lap_times"
RESULTS_DIR = PROJECT_ROOT / "data" / "race_results"
OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
OUT_JSON = OUT_DIR / "jra_lap33_theory_2026_08_28_result.json"
OUT_TXT = OUT_DIR / "jra_lap33_theory_2026_08_28_report.txt"

MIN_N_TRUST = 20  # このN未満のセルは参考値として明記する閾値

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


# ===================================================================== ロード
def load_lap_times() -> pd.DataFrame:
    frames = []
    for year_dir in sorted(LAP_DIR.glob("[0-9][0-9][0-9][0-9]")):
        for p in sorted(year_dir.glob("*.csv")):
            df = pd.read_csv(p, dtype=str)
            if len(df):
                frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["segment_index"] = pd.to_numeric(out["segment_index"], errors="coerce")
    out["lap_time_sec"] = pd.to_numeric(out["lap_time_sec"], errors="coerce")
    return out


def load_race_meta() -> pd.DataFrame:
    frames = []
    for year_dir in sorted(RESULTS_DIR.glob("[0-9][0-9][0-9][0-9]")):
        for p in sorted(year_dir.glob("*.csv")):
            df = pd.read_csv(p, dtype=str, usecols=["race_id", "surface", "distance_m", "racecourse"])
            frames.append(df)
    out = pd.concat(frames, ignore_index=True).drop_duplicates("race_id")
    out["distance_m"] = pd.to_numeric(out["distance_m"], errors="coerce")
    return out


log("ロード中...")
lap = load_lap_times()
meta = load_race_meta()
log(f"lap_times行数: {len(lap)}  対象race_id数: {lap['race_id'].nunique()}")
log(f"race_results race_id数: {len(meta)}")


# ===================================================================== 33ラップ計算
def compute_lap33(lap: pd.DataFrame) -> pd.DataFrame:
    """race_idごとに33ラップ値を計算する。末尾3区間=上がり3F、その直前3区間=残り
    1200-600m。距離が200mの倍数でない場合、端数は先頭区間側に吸収される(実データ確認済み、
    docstring参照)ため、区間数が7以上であれば末尾側の切り出しは常に正しい(先頭の変則区間は
    計算対象の外側に位置するため)。

    区間数がちょうど6の場合のみ例外: 距離が1200mちょうど(200mの倍数)なら全区間が
    クリーンな200mで問題無いが、distance_m%200!=0(例: 1150m)の場合は先頭区間
    (=「残り1200-600m」の一部として使われてしまう区間)が変則的に短い/長いため、
    「前3F」が実質600m相当になっていない(実測確認: 福島ダ1150mで先頭区間9.7秒 vs
    他区間10.9-12.7秒、n=73レース平均で理論値-2.9に対し実測-5.49という理論からの
    突出した乖離として発現)。この判定にはdistance_mが必要なため、ここでは全て計算した
    上でrace_id・n_segmentsを返し、呼び出し側(meta結合後)でdistance_m%200!=0かつ
    n_segments==6の行を除外する。"""
    rows = []
    for race_id, g in lap.groupby("race_id"):
        g = g.dropna(subset=["segment_index", "lap_time_sec"]).sort_values("segment_index")
        n = len(g)
        if n < 6:
            continue
        vals = g["lap_time_sec"].to_numpy()
        final_3f = float(vals[-3:].sum())
        preceding_3f = float(vals[-6:-3].sum())
        rows.append({"race_id": race_id, "n_segments": n,
                     "final_3f": final_3f, "preceding_3f": preceding_3f,
                     "lap33": preceding_3f - final_3f})
    return pd.DataFrame(rows)


lap33 = compute_lap33(lap)
log(f"33ラップ計算成功: {len(lap33)}/{lap['race_id'].nunique()} races(区間数>=6のみ)")

merged = lap33.merge(meta, on="race_id", how="left")
merged = merged[merged["surface"].isin(["芝", "ダ"]) & merged["distance_m"].notna()]
log(f"surface/distance_m結合後: {len(merged)}件")

# n_segments==6かつ distance_m が200mの倍数でない(例: 1150m)レースは、先頭区間に距離の
# 端数が吸収されるため「前3F」が実質600m相当になっていない(実測: 福島ダ1150mで理論値-2.9
# に対し実測-5.49という突出した乖離、docstring参照)。これらは計算対象から除外する。
contaminated = (merged["n_segments"] == 6) & (merged["distance_m"] % 200 != 0)
n_contaminated = int(contaminated.sum())
if n_contaminated:
    log(f"\n除外: 距離が200mの倍数でない6区間レース(前3F汚染、実測で確認済みの既知の限界): "
        f"{n_contaminated}件  対象距離: {sorted(merged.loc[contaminated, 'distance_m'].unique())}")
    merged = merged[~contaminated]

# ===================================================================== コース別集計 vs 理論
log("\n" + "=" * 72)
log("コース×距離×芝ダ別 平均33ラップ: 実データ vs 理論PDF")
log("=" * 72)

grp = merged.groupby(["racecourse", "surface", "distance_m"])["lap33"].agg(["mean", "std", "count"])
rows_out = []
for (course, surf, dist), r in grp.iterrows():
    key = (course, surf, int(dist))
    theory_val = THEORY_TABLE.get(key)
    ambiguous = key in AMBIGUOUS_CELLS
    if theory_val is None and ambiguous:
        theory_val = THEORY_TABLE_AMBIGUOUS_AVG[key]
    if theory_val is None:
        continue  # 理論表に掲載の無い組合せは比較対象外(黙って除外、件数は末尾に記録)
    diff = r["mean"] - theory_val
    rows_out.append({
        "racecourse": course, "surface": surf, "distance_m": int(dist),
        "n_races": int(r["count"]), "actual_mean": float(r["mean"]), "actual_std": float(r["std"]),
        "theory_value": float(theory_val), "diff": float(diff), "ambiguous_inout": ambiguous,
        "low_n": bool(r["count"] < MIN_N_TRUST),
    })

result_df = pd.DataFrame(rows_out).sort_values(["surface", "distance_m", "racecourse"])
for _, r in result_df.iterrows():
    flag = []
    if r["low_n"]:
        flag.append(f"N={r['n_races']}要注意")
    if r["ambiguous_inout"]:
        flag.append("内外統合値と比較")
    flagstr = f"  [{', '.join(flag)}]" if flag else ""
    log(f"  {r['surface']}{r['distance_m']:.0f} {r['racecourse']}: "
        f"実測={r['actual_mean']:+.2f}(N={r['n_races']}, sd={r['actual_std']:.2f})  "
        f"理論={r['theory_value']:+.2f}  差={r['diff']:+.2f}{flagstr}")

n_cells = len(result_df)
n_covered = int((~result_df["low_n"]).sum())
sign_agree = float((np.sign(result_df["actual_mean"]) == np.sign(result_df["theory_value"])).mean())
sign_agree_trusted = float((np.sign(result_df[~result_df["low_n"]]["actual_mean"]) ==
                            np.sign(result_df[~result_df["low_n"]]["theory_value"])).mean())
rho, pval = stats.spearmanr(result_df["actual_mean"], result_df["theory_value"])
rho_trusted, pval_trusted = stats.spearmanr(
    result_df[~result_df["low_n"]]["actual_mean"], result_df[~result_df["low_n"]]["theory_value"])
mae = float((result_df["diff"]).abs().mean())
mae_trusted = float((result_df[~result_df["low_n"]]["diff"]).abs().mean())

log("\n" + "=" * 72)
log("まとめ")
log("=" * 72)
log(f"比較対象セル数: {n_cells}(うちN>={MIN_N_TRUST}の信頼できるセル: {n_covered})")
log(f"符号一致率(全セル): {sign_agree*100:.1f}%   符号一致率(N>={MIN_N_TRUST}のみ): {sign_agree_trusted*100:.1f}%")
log(f"Spearman順位相関(全セル): rho={rho:+.3f} p={pval:.4f}")
log(f"Spearman順位相関(N>={MIN_N_TRUST}のみ): rho={rho_trusted:+.3f} p={pval_trusted:.4f}")
log(f"平均絶対誤差(全セル): {mae:.2f}pt   平均絶対誤差(N>={MIN_N_TRUST}のみ): {mae_trusted:.2f}pt")

theory_keys_covered = {(r["racecourse"], r["surface"], r["distance_m"]) for _, r in result_df.iterrows()}
theory_keys_all = set(THEORY_TABLE) | set(THEORY_TABLE_AMBIGUOUS_AVG)
missing = theory_keys_all - theory_keys_covered
log(f"\n理論表に載っているがこちらのデータで未計算(該当レース無しまたはlap_times未取得)"
    f"のセル数: {len(missing)}")
if missing:
    log(f"  内訳(距離小さい順): {sorted(missing, key=lambda x: (x[1], x[2]))}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps({
    "n_lap_races": len(lap33), "n_merged": len(merged),
    "cells": result_df.to_dict(orient="records"),
    "n_cells": n_cells, "n_cells_trusted": n_covered,
    "sign_agreement_all": sign_agree, "sign_agreement_trusted": sign_agree_trusted,
    "spearman_rho_all": float(rho), "spearman_p_all": float(pval),
    "spearman_rho_trusted": float(rho_trusted), "spearman_p_trusted": float(pval_trusted),
    "mae_all": mae, "mae_trusted": mae_trusted,
    "missing_theory_cells": sorted([list(m) for m in missing], key=lambda x: (x[1], x[2])),
}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")

# race_id -> lap33値を後続(Part2)から再利用できるよう別途保存
lap33_export = merged[["race_id", "lap33", "racecourse", "surface", "distance_m", "n_segments"]]
lap33_export.to_csv(OUT_DIR / "jra_lap33_by_race_2026_08_28.csv", index=False, encoding="utf-8")
log(f"wrote jra_lap33_by_race_2026_08_28.csv ({len(lap33_export)}行、Part2用)")
