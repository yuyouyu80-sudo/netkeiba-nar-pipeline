# -*- coding: utf-8 -*-
"""NAR通常戦モデル(予想5頭BOX)の確信度フィルタリング(5~12レース/日)。
confidence_sweep_box4_nar.py/confidence_sweep_box3_nar.pyと全く同じ設計
(N_RANGE=5~12)を、BOX_N=5に戻したもの。

2026-07-29、予想4頭BOX回収率検証で採用した等重み14シグナルモデル(nar_signals.py +
winner_box4_nar.json、predict_box4_nar.py)にスコアリングを統一した。旧版は
scripts/predict_pattern29.pyのWEIGHTS_NAR(pattern24、40レースのみで探索・以後
レビュー対象外)を動的importして使っていた。

2026-07-30、確信度の基準を「BOXの賭け目位置(5位-6位差)」から「1位-2位のスコア差
(gap_top2)」に切替え。nar_confidence_calibrate.pyの再検証で、5位-6位差は的中率と
逆相関(-0.136、値が大きいほど的中率が下がる)であることが判明したため
(box4/box3は既に2026-07-29にgap_top2へ切替済みで、box5だけ旧基準が残っていた)。
gap_top2は正の相関(+0.104)で、box4(+0.185)・box3(+0.199)と同じ方向性。
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nar_pipeline"
UNIT = 100
BET_TYPES = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "3連複", "3連単"]
N_RANGE = [5, 6, 7, 8, 9, 10, 11, 12]
BOX_N = 5

sys.path.insert(0, str(LIB_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
import predict_box4_nar as predict_mod  # noqa: E402

PATTERN_ID = predict_mod.PATTERN_ID

from src.netkeiba_pipeline.storage.paths import newspaper_csv_path  # noqa: E402

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
print(f"DATES={DATES}  target races={len(target)}  BOX_N={BOX_N}  pattern={PATTERN_ID}")

payout_frames = [pd.read_csv(_payouts_dir / f"{d}.csv", dtype=str) for d in DATES]
payouts = pd.concat(payout_frames, ignore_index=True)
payouts["payout"] = payouts["payout"].astype(int)


def parse_combo(bet_type: str, combo_text: str):
    if bet_type in ("単勝", "複勝"):
        return int(combo_text)
    if bet_type in ("馬単", "3連単"):
        return tuple(int(x) for x in combo_text.split("→"))
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
        wakus = sorted(set(pd.to_numeric(g["waku"], errors="coerce").dropna().astype(int).tolist()))
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
        settle("枠連", [frozenset(c) for c in itertools.combinations(wakus, 2)])
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

    all_summaries = [full_summary]
    conf_sorted = conf_df.sort_values("gap_pct", ascending=False, kind="stable")
    for n in N_RANGE:
        selected = conf_sorted.groupby("kaisai_date", group_keys=False).head(n)
        selected_ids = set(selected["race_id"])
        sub_pred = pred_df[pred_df["race_id"].isin(selected_ids)]
        s = box_return(sub_pred)
        s.insert(0, "scope", f"高確信度{n}レース/日(計{len(selected_ids)}レース)")
        all_summaries.append(s)

    result = pd.concat(all_summaries, ignore_index=True)
    result.insert(0, "model", model_name)
    return result


if __name__ == "__main__":
    result = analyze_model(f"通常戦モデル(BOX5, {predict_mod.MODEL_LABEL})", predict_mod.score_race)
    result.to_csv(DATA_DIR / "confidence_sweep_baseline_nar.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
