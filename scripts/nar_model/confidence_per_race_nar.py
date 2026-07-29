# -*- coding: utf-8 -*-
"""地方競馬(NAR)通常戦モデルのレース単位確信度指標(N=5でのパーセンタイル正規化
ギャップ)。confidence_per_race.py(JRA)を移植したもの。

スコアリングは predict_box4_nar.py(BOX4回収率検証で採用した等重み14シグナル
モデル、nar_signals.py + winner_box4_nar.json)と同一にする。レースカードの
「予想5頭」(predict_top5_nar.pyが生成するpredictions_nar_{date}.csv)と確信度
バッジの順位付けが食い違わないようにするため。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nar_pipeline"
N = 5  # 「高確信度5レース/日」タブと同じ基準に揃える(JRAと統一)

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

    if field_size > N:
        gap = sorted_scores[N - 1] - sorted_scores[N]
        gap_pct = gap / spread if spread > 0 else 0.0
    else:
        gap_pct = np.inf  # 出走頭数がN以下ならBOXが全頭を覆うため比較不能(=最大確信扱い)

    rows.append({
        "race_id": race_id,
        "kaisai_date": row["kaisai_date"],
        "field_size": field_size,
        "gap_pct_5": gap_pct,
    })

conf = pd.DataFrame(rows)
conf["selected_5"] = False
for _date, g in conf.groupby("kaisai_date"):
    top_idx = g.sort_values("gap_pct_5", ascending=False, kind="stable").head(N).index
    conf.loc[top_idx, "selected_5"] = True

# --- 較正済み確信度(2026-07-29、nar_confidence_calibrate.pyのLOBO較正結果を反映) ---
# gap_pct_5をそのまま「確信度%」として表示していたが、実際の複勝的中率との対応は
# 未検証だった。box_n=5ではgap_pct(=gap_pct_5と同一統計量)がLOBO OOFで自明な基準を
# 上回る唯一の候補だったため、ランキング(selected_5)はそのまま維持しつつ、表示する
# %だけを「そのバケットの実測複勝的中率」に置き換える。
import json as _json  # noqa: E402
_calib_path = DATA_DIR / "confidence_calibration_nar.json"
if _calib_path.exists():
    _calib = _json.loads(_calib_path.read_text(encoding="utf-8"))
    _tbl = _calib["results_by_box_n"]["5"]["place"]["calibration_table"]
    assert _tbl["feature"] == "gap_pct", f"box5較正の対象特徴量がgap_pctではありません: {_tbl['feature']}"
    _edges = _tbl["edges"]
    _bucket_rate = {b["bucket"]: b["hit_rate_pct"] for b in _tbl["buckets"]}
    _finite = conf["gap_pct_5"].replace([np.inf], 1.0)
    _bucket_idx = np.digitize(_finite.to_numpy(dtype=float), _edges)
    conf["confidence_calibrated_pct"] = [_bucket_rate.get(int(b), np.nan) for b in _bucket_idx]
else:
    conf["confidence_calibrated_pct"] = np.nan

conf.to_csv(DATA_DIR / "confidence_per_race_nar.csv", index=False, encoding="utf-8-sig")
print(conf.to_string(index=False))
print("races:", len(conf), "selected_5 count:", int(conf["selected_5"].sum()))
