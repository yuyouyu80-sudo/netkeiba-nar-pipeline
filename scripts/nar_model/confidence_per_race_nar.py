# -*- coding: utf-8 -*-
"""地方競馬(NAR)通常戦モデルのレース単位確信度指標。

スコアリングは predict_box5_nar.py(box5専用に300パターン探索で選んだ独立重み
モデル、nar_signals.py + winner_box5_nar.json)と同一にする。レースカードの
「予想5頭」(predict_top5_nar.pyが生成するpredictions_nar_{date}.csv)と確信度
バッジの順位付けが食い違わないようにするため。

2026-07-30、`gap_pct`(旧来のBOX賭け目位置でのスコア差)が的中率と逆相関(-0.136)である
ことが判明したため、`selected_5`の順位付けは`gap_top2`(1位-2位のスコア差)に統一済み
(このロジックは維持)。

2026-08-12、JRA/NAR確信度統一の一環で、較正済みパーセンテージの算出部分を全面書き換えた。
旧来の`confidence_calibrated_pct`(box5×place較正)を廃止し、JRAと同じ単勝ベース
`ladder_conf_{k}_pct`(k=5..1)に統一した。較正パラメータの適用は
`scripts/common/confidence_calibrate.py`の`apply_calibration`を使う。
`ladder_conf_{k}_show_pct`がFalseのK/レースはレポート側で3段階(高/中/低)表示に
フォールバックする(較正のブロック数不足・統計的有意性不足の場合)。
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "common"))
sys.path.insert(0, str(PROJECT_ROOT))
import confidence_calibrate as CC  # noqa: E402
import predict_box5_nar as predict_mod  # noqa: E402

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

    gaps = CC.gap_features(sorted_scores, LADDER_KS)
    rec = {"race_id": race_id, "kaisai_date": row["kaisai_date"], "field_size": field_size, **gaps}
    rows.append(rec)

conf = pd.DataFrame(rows)
conf["selected_5"] = False
for _date, g in conf.groupby("kaisai_date"):
    top_idx = g.sort_values("gap_top2", ascending=False, kind="stable").head(5).index
    conf.loc[top_idx, "selected_5"] = True

# 後方互換のため、5位確信度のはしご値をgap_pct_5という旧列名でも保持する。
conf["gap_pct_5"] = conf["gap_boundary_5"]

_calib_path = DATA_DIR / "confidence_calibration_nar.json"
if _calib_path.exists():
    _calib = json.loads(_calib_path.read_text(encoding="utf-8"))
    _ladder = _calib.get("topk_ladder", {}).get("results_by_k", {})
    for k in LADDER_KS:
        entry = _ladder.get(str(k))
        col = f"ladder_conf_{k}_pct"
        if entry is None:
            conf[col] = np.nan
            conf[f"ladder_conf_{k}_show_pct"] = False
            continue
        conf[col] = CC.apply_calibration(conf[f"gap_boundary_{k}"], entry)
        conf[f"ladder_conf_{k}_show_pct"] = bool(entry["show_pct"])
else:
    for k in LADDER_KS:
        conf[f"ladder_conf_{k}_pct"] = np.nan
        conf[f"ladder_conf_{k}_show_pct"] = False

conf.to_csv(DATA_DIR / "confidence_per_race_nar.csv", index=False, encoding="utf-8-sig")
print(conf.to_string(index=False))
print("races:", len(conf), "selected_5 count:", int(conf["selected_5"].sum()))
