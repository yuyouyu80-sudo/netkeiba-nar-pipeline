# -*- coding: utf-8 -*-
"""NAR予想3頭BOXモデル(winner_box3_nar.json・predict_box3_nar.py)の確信度フィルタ
リング(5~12レース/日)。JRAのconfidence_sweep_box3.pyと同じ設計を踏襲し、以下を守る:

  - 確信度指標(gap_pct)はNごとに再計算せず、常に固定の1つの統計量を使う。N_RANGE=[5..12]は、
    その1つの固定ランキング上で「その日何レースを採用するか」というカットオフ件数としてのみ
    使う。これによりn=5の選出集合はn=6の真部分集合...という入れ子関係が保証され、Nを増やす
    ほど的中数は単調非減少になる(JRA側で過去に発生した非単調バグの再発防止)。
  - 2026-07-29、nar_confidence_calibrate.pyでのLOBO較正検証を反映し、確信度指標を
    「BOX3の賭け目位置(3位-4位差)」から「1位-2位のスコア差」(gap_top2)に変更した。
    3位-4位差は実際の複勝的中率とほぼ無相関(Spearman-0.049)だったのに対し、gap_top2は
    +0.238の正相関があり、自明な基準(常に平均を予測)よりOOF Brier scoreが改善した
    唯一の候補だった(詳細はconfidence_calibration_nar.json参照)。
  - NARはJRAよりレース数の多い開催日(1場あたり最大12R)があるため、N_RANGEは
    JRAの5~10ではなく5~12まで対象とする(ユーザー要望)。
  - 対象レースはdata/race_results/nar配下の検証済み日付から自動検出する
    (search_patterns_nar.py・box_return_nar.pyと同じ自動検出ロジック)。
"""
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nar_pipeline"
UNIT = 100
BET_TYPES = ["単勝", "複勝", "馬連", "ワイド", "馬単", "3連複", "3連単"]  # 枠連は除外(NARに枠番データ無し)
N_RANGE = [5, 6, 7, 8, 9, 10, 11, 12]

sys_path_added = str(PROJECT_ROOT)
import sys as _sys  # noqa: E402
if sys_path_added not in _sys.path:
    _sys.path.insert(0, sys_path_added)
from src.netkeiba_pipeline.storage.paths import newspaper_csv_path  # noqa: E402

spec = importlib.util.spec_from_file_location("predict_box3_nar_mod", LIB_DIR / "predict_box3_nar.py")
predict_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_mod)

BOX_N = predict_mod.BOX_N  # 3

# --- 対象レース一覧: data/race_results/nar配下の検証済み(payoutsも存在する)日付から自動検出 ---
_results_dir = PROJECT_ROOT / "data" / "race_results" / "nar" / "2026"
_payouts_dir = PROJECT_ROOT / "data" / "payouts" / "nar" / "2026"
DATES = sorted(p.stem for p in _results_dir.glob("2026*.csv") if (_payouts_dir / p.name).exists())

race_name_rows = []
for d in DATES:
    df = pd.read_csv(_results_dir / f"{d}.csv", dtype=str)
    names = df[["race_id", "race_name", "racecourse"]].drop_duplicates("race_id").copy()
    names["kaisai_date"] = d
    race_name_rows.append(names)
race_names = pd.concat(race_name_rows, ignore_index=True)
target = race_names[~race_names["race_name"].str.contains("新馬|未勝利", regex=True, na=False)]
print(f"DATES={DATES}  target races={len(target)}  BOX_N={BOX_N}  pattern={predict_mod.PATTERN_ID}")

payout_frames = [pd.read_csv(_payouts_dir / f"{d}.csv", dtype=str) for d in DATES]
payouts = pd.concat(payout_frames, ignore_index=True)
payouts["payout"] = payouts["payout"].astype(int)


def parse_combo(bet_type: str, combo_text: str):
    if bet_type in ("単勝", "複勝"):
        return int(combo_text)
    if bet_type in ("馬単", "3連単"):
        return tuple(int(x) for x in combo_text.split("→"))
    # NARの一部レース(金沢 202646072603-607)の「枠単」誤ラベル行対策(他のNARスクリプトと同じ)。
    if "→" in combo_text:
        return None
    return frozenset(int(x) for x in combo_text.split("-"))


ACTUAL_MAPS = {}
for race_id, g in payouts.groupby("race_id"):
    per_bt = {}
    for bt in BET_TYPES:
        rows = g[g["bet_type"] == bt]
        m = {}
        for c, p in zip(rows["combination"], rows["payout"]):
            key = parse_combo(bt, c)
            if key is None:
                continue
            m[key] = m.get(key, 0) + p
        per_bt[bt] = m
    ACTUAL_MAPS[race_id] = per_bt


def box_return(pred_df: pd.DataFrame) -> pd.DataFrame:
    results = {bt: {"stake": 0, "return": 0, "hit_races": 0, "race_count": 0} for bt in BET_TYPES}
    for race_id, g in pred_df.groupby("race_id", sort=False):
        umabans = g["umaban"].astype(int).tolist()
        actual = ACTUAL_MAPS.get(race_id, {bt: {} for bt in BET_TYPES})

        def settle(bt, combos):
            stake = len(combos) * UNIT
            ret = sum(actual[bt].get(c, 0) for c in combos)
            results[bt]["stake"] += stake
            results[bt]["return"] += ret
            results[bt]["race_count"] += 1
            results[bt]["hit_races"] += 1 if ret > 0 else 0

        settle("単勝", umabans)
        settle("複勝", umabans)
        settle("馬連", [frozenset(c) for c in itertools.combinations(umabans, 2)])
        settle("ワイド", [frozenset(c) for c in itertools.combinations(umabans, 2)])
        settle("馬単", list(itertools.permutations(umabans, 2)))
        settle("3連複", [frozenset(c) for c in itertools.combinations(umabans, 3)])
        settle("3連単", list(itertools.permutations(umabans, 3)))

    rows = []
    for bt in BET_TYPES:
        r = results[bt]
        rate = (r["return"] / r["stake"] * 100) if r["stake"] else 0.0
        rows.append({
            "bet_type": bt, "races": r["race_count"], "hit_races": r["hit_races"],
            "hit_rate_pct": round(r["hit_races"] / r["race_count"] * 100, 1) if r["race_count"] else 0,
            "total_stake": r["stake"], "total_return": r["return"], "return_rate_pct": round(rate, 1),
        })
    return pd.DataFrame(rows)


def analyze_model(model_name: str, score_fn):
    conf_rows = []
    pred_rows = []
    for _, row in target.iterrows():
        race_id = row["race_id"]
        path = newspaper_csv_path(race_id)
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
        if df.empty:
            continue
        df = predict_mod._drop_scratched(df)
        if df.empty:
            continue
        current_class = predict_mod._class_ordinal(row["race_name"])
        scored = score_fn(df, current_class)
        score = scored["_score"].to_numpy(dtype=float)
        df = scored
        order = np.argsort(-np.where(np.isnan(score), -1e18, score), kind="stable")
        sorted_scores = np.where(np.isnan(score), -1e18, score)[order]
        field_size = len(df)
        top_score, bottom_score = sorted_scores[0], sorted_scores[-1]
        spread = top_score - bottom_score

        # gap_top2(1位-2位差): LOBO較正で実際の複勝的中率と正相関が確認された統計量
        gap_pct = (sorted_scores[0] - sorted_scores[1]) / spread if (field_size > 1 and spread > 0) else 0.0
        conf_rows.append({"race_id": race_id, "kaisai_date": row["kaisai_date"], "gap_pct": gap_pct})

        topN = df.iloc[order[:BOX_N]]
        for _, r in topN.iterrows():
            pred_rows.append({"race_id": race_id, "kaisai_date": row["kaisai_date"],
                               "umaban": int(r["umaban"]), "waku": r["waku"]})

    conf_df = pd.DataFrame(conf_rows)
    pred_df = pd.DataFrame(pred_rows)

    full_summary = box_return(pred_df)
    full_summary.insert(0, "scope", f"全{len(conf_df)}レース")
    full_summary.insert(0, "model", model_name)

    all_summaries = [full_summary]
    conf_sorted = conf_df.sort_values("gap_pct", ascending=False, kind="stable")
    for n in N_RANGE:
        selected = conf_sorted.groupby("kaisai_date", group_keys=False).head(n)
        selected_ids = set(selected["race_id"])
        sub_pred = pred_df[pred_df["race_id"].isin(selected_ids)]
        s = box_return(sub_pred)
        s.insert(0, "scope", f"高確信度{n}レース/日(計{len(selected_ids)}レース)")
        s.insert(0, "model", model_name)
        all_summaries.append(s)

    return pd.concat(all_summaries, ignore_index=True)


if __name__ == "__main__":
    result = analyze_model(f"通常戦モデル(BOX3, {predict_mod.MODEL_LABEL})", predict_mod.score_race)
    result.to_csv(DATA_DIR / "confidence_sweep_box3_nar.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
