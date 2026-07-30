# -*- coding: utf-8 -*-
"""地方競馬(NAR)通常戦モデルのレース単位確信度指標。confidence_per_race.py(JRA)を移植。

スコアリングは predict_box4_nar.py(BOX4回収率検証で採用した等重み14シグナル
モデル、nar_signals.py + winner_box4_nar.json)と同一にする。レースカードの
「予想5頭」(predict_top5_nar.pyが生成するpredictions_nar_{date}.csv)と確信度
バッジの順位付けが食い違わないようにするため。

2026-07-30、nar_confidence_calibrate.pyの再検証で、旧来の「BOXの賭け目位置での
スコア差(box5ならN=5位置)」は的中率と逆相関(-0.136)であることが判明したため、
box4/box3と同じ「1位-2位のスコア差(gap_top2)」に統一した(selected_5の順位付けも
gap_top2ベースに変更)。あわせて、上位K頭picksに実際の1着馬が入っている確率を
K=5,4,3,2,1のはしご状に表示する`topk_ladder`列を追加した
(confidence_calibration_nar.jsonのtopk_ladderセクションを参照)。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nar_pipeline"
LADDER_KS = [5, 4, 3, 2, 1]

sys.path.insert(0, str(LIB_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
import predict_box4_nar as predict_mod  # noqa: E402

from src.netkeiba_pipeline.storage.paths import newspaper_csv_path  # noqa: E402


def _load_targets() -> pd.DataFrame:
    cols = ["race_id", "kaisai_date", "race_name"]
    frames = []
    for path in sorted(DATA_DIR.glob("race_names_nar_*.csv")):
        df = pd.read_csv(path, dtype=str)
        frames.append(df[~df["race_name"].str.contains("新馬|未勝利", regex=True, na=False)][cols])
    return pd.concat(frames, ignore_index=True)


target = _load_targets()
assert target["race_id"].is_unique, "race_names_nar_*.csv の間でrace_idが重複しています。"

rows = []
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
    scored = predict_mod.score_race(df, current_class)
    score = scored["_score"].to_numpy(dtype=float)
    score = np.where(np.isnan(score), -1e18, score)
    order = np.argsort(-score, kind="stable")
    sorted_scores = score[order]
    field_size = len(scored)
    spread = sorted_scores[0] - sorted_scores[-1]

    gap_top2 = (sorted_scores[0] - sorted_scores[1]) / spread if (field_size > 1 and spread > 0) else 0.0

    rec = {
        "race_id": race_id, "kaisai_date": row["kaisai_date"],
        "field_size": field_size, "gap_top2": gap_top2,
    }
    for k in LADDER_KS:
        if field_size > k and spread > 0:
            rec[f"gap_boundary_{k}"] = (sorted_scores[k - 1] - sorted_scores[k]) / spread
        else:
            rec[f"gap_boundary_{k}"] = 1.0  # 全頭カバー、またはスコア差が無い場合は最大確信として扱う
    rows.append(rec)

conf = pd.DataFrame(rows)
conf["selected_5"] = False
for _date, g in conf.groupby("kaisai_date"):
    top_idx = g.sort_values("gap_top2", ascending=False, kind="stable").head(5).index
    conf.loc[top_idx, "selected_5"] = True

# 後方互換のため、5位確信度のはしご値をgap_pct_5という旧列名でも保持する。
conf["gap_pct_5"] = conf["gap_boundary_5"]


def _apply_calibration(values: pd.Series, table: dict) -> pd.Series:
    edges = table["edges"]
    bucket_rate = {b["bucket"]: b["hit_rate_pct"] for b in table["buckets"]}
    idx = np.digitize(values.to_numpy(dtype=float), edges)
    return pd.Series([bucket_rate.get(int(b), np.nan) for b in idx], index=values.index)


_calib_path = DATA_DIR / "confidence_calibration_nar.json"
if _calib_path.exists():
    _calib = json.loads(_calib_path.read_text(encoding="utf-8"))

    # --- box5(予想5頭)本体の確信度%: results_by_box_n."5"."place" ---
    _tbl = _calib["results_by_box_n"]["5"]["place"]["calibration_table"]
    _feat_col = {"gap_pct": "gap_pct_5", "spread": None, "gap_top2": "gap_top2"}.get(_tbl["feature"])
    assert _feat_col is not None, f"box5較正の対象特徴量に未対応です: {_tbl['feature']}"
    conf["confidence_calibrated_pct"] = _apply_calibration(conf[_feat_col], _tbl)

    # --- 段階的的中確率のはしご(5頭確信度〜1頭確信度) ---
    _ladder = _calib.get("topk_ladder", {}).get("results_by_k", {})
    for k in LADDER_KS:
        entry = _ladder.get(str(k))
        col = f"ladder_conf_{k}_pct"
        if entry is None:
            conf[col] = np.nan
            continue
        feat = entry["chosen_feature"]
        src_col = "gap_top2" if feat == "gap_top2" else f"gap_boundary_{k}"
        conf[col] = _apply_calibration(conf[src_col], entry["calibration_table"])
        conf[f"ladder_conf_{k}_beats_trivial"] = bool(entry["beats_trivial_baseline"])
else:
    conf["confidence_calibrated_pct"] = np.nan
    for k in LADDER_KS:
        conf[f"ladder_conf_{k}_pct"] = np.nan
        conf[f"ladder_conf_{k}_beats_trivial"] = False

conf.to_csv(DATA_DIR / "confidence_per_race_nar.csv", index=False, encoding="utf-8-sig")
print(conf.to_string(index=False))
print("races:", len(conf), "selected_5 count:", int(conf["selected_5"].sum()))
