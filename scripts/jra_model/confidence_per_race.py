# -*- coding: utf-8 -*-
"""レース単位の確信度指標を算出し、レポートの各レースカード用CSVを出力する。

2026-08-12、JRA/NAR確信度統一の一環でscratchpadから昇格し、`BASE / "predict.py"`への動的import
依存を解消した(jra_signals.py+winner_v3.jsonを直接使用)。対象は検証済みレースに限らず、
race_names_YYYYMMDD.csv(scratchpad、日次予想生成が書き出すpending日を含む全レース一覧)にある
新馬・未勝利以外の全レース。pending日は実績が無くてもモデルのスコアだけで確信度は計算できるため
含める。newspaper CSVが無いレースはスキップする。

gap_pct_5/selected_5(その日の「高確信度5レース」選定用)は較正とは独立の生スコア指標のため
変更していない。ladder_conf_{k}_pct(k=5..1)は`confidence_calibration.json`
(jra_confidence_calibrate.pyが生成)の較正パラメータを`confidence_calibrate.apply_calibration`で
適用した値。`ladder_conf_{k}_show_pct`がFalseのK/レースはレポート側で3段階(高/中/低)表示に
フォールバックする(較正のブロック数不足・統計的有意性不足の場合)。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
SCRATCHPAD = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
sys.path.insert(0, str(LIB_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "common"))
import confidence_calibrate as CC  # noqa: E402
import jra_signals as JS  # noqa: E402

LADDER_KS = [5, 4, 3, 2, 1]
N = 5  # 「高確信度5レース/日」タブと同じ基準

winner = json.loads((DATA_DIR / "winner_v3.json").read_text(encoding="utf-8"))
WEIGHTS, PRIORS, CLASS_ORDINAL = winner["weights"], winner["priors"], winner["class_ordinal"]


def _load_targets() -> pd.DataFrame:
    """race_names_[0-9]*.csv(pending日含む全検証対象)のみから構築する。同じscratchpadに
    置かれているrace_names_nar_*.csv(NAR用、別スキーマ)を誤って拾わないよう絞り込む。"""
    cols = ["race_id", "kaisai_date", "race_name"]
    frames = []
    for path in sorted(SCRATCHPAD.glob("race_names_[0-9]*.csv")):
        df = pd.read_csv(path, dtype=str)
        frames.append(df[~df["race_name"].str.contains("新馬|未勝利", regex=True, na=False)][cols])
    return pd.concat(frames, ignore_index=True)


target = _load_targets()
assert target["race_id"].is_unique, (
    "race_names_*.csv(pending日含む)間でrace_idが重複しています。"
)

rows = []
for _, row in target.iterrows():
    race_id = row["race_id"]
    path = PROJECT_ROOT / "data" / "newspaper" / f"{race_id}.csv"
    if not path.exists():
        continue
    df = pd.read_csv(path, dtype=str, encoding="utf-8")
    if df.empty:
        continue
    df = JS._drop_scratched(df)
    if df.empty:
        continue
    current_class = JS._class_ordinal(row["race_name"], CLASS_ORDINAL)
    scored = JS.score_race(df, current_class, WEIGHTS, PRIORS, CLASS_ORDINAL)
    score = scored["_score"].to_numpy(dtype=float)
    score = np.where(np.isnan(score), -1e18, score)
    order = np.argsort(-score, kind="stable")
    sorted_scores = score[order]
    field_size = len(scored)
    spread = sorted_scores[0] - sorted_scores[-1] if field_size else 0.0

    if field_size > N:
        gap = sorted_scores[N - 1] - sorted_scores[N]
        gap_pct = gap / spread if spread > 0 else 0.0
    else:
        gap_pct = np.inf  # 出走頭数がN以下ならBOXが全頭を覆うため比較不能(=最大確信扱い)

    gaps = CC.gap_features(sorted_scores, LADDER_KS)
    rec = {"race_id": race_id, "kaisai_date": row["kaisai_date"], "field_size": field_size,
          "gap_pct_5": gap_pct, **gaps}
    rows.append(rec)

conf = pd.DataFrame(rows)
conf["selected_5"] = False
for _date, g in conf.groupby("kaisai_date"):
    top_idx = g.sort_values("gap_pct_5", ascending=False, kind="stable").head(N).index
    conf.loc[top_idx, "selected_5"] = True

_calib_path = DATA_DIR / "confidence_calibration.json"
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

conf.to_csv(DATA_DIR / "confidence_per_race.csv", index=False, encoding="utf-8-sig")
print(conf.to_string(index=False))
print("races:", len(conf), "selected_5 count:", int(conf["selected_5"].sum()))
