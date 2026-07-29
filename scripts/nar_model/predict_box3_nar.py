# -*- coding: utf-8 -*-
"""NAR 予想3頭BOX モデルのスコアリング。

旧版は scripts/predict_pattern29.py を動的importして重みだけ差し替えていたが(BOX4の旧版と
同じ設計)、探索側と本番側で priors が食い違うバグの温床だった。predict_box4_nar.py と同じく
nar_signals.py を単一の真実の源として使い、重みも priors も winner_box3_nar.json から読む。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import nar_signals as NS

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "nar_pipeline"

_winner = json.loads((DATA_DIR / "winner_box3_nar.json").read_text(encoding="utf-8"))
BOX_N = _winner["box_n"]
PATTERN_ID = _winner.get("pattern_id", 0)
MODEL_LABEL = _winner.get("model_label", f"pattern{PATTERN_ID}")
WEIGHTS = _winner["weights"]
DEAD_SIGNALS = _winner.get("dead_signals", [])
ALIVE_SIGNALS = _winner.get("alive_signals", [])
PRIORS = {k: (np.nan if v is None else float(v)) for k, v in _winner["priors"].items()}


def _drop_scratched(df: pd.DataFrame) -> pd.DataFrame:
    """出走取消・除外馬(単勝オッズと人気の両方が空)を落とす。"""
    odds = NS._num(df["bias_win_odds"])
    ninki = NS._num(df["bias_ninki"])
    return df[odds.notna() & ninki.notna()].reset_index(drop=True)


def _class_ordinal(text) -> float:
    return NS.class_ordinal(text)


def score_race(df: pd.DataFrame, current_class: float) -> pd.DataFrame:
    """馬柱DataFrameに _score 列を足して返す。"""
    sig = NS.build_signals(df, current_class, PRIORS)
    out = df.copy()
    out["_score"] = NS.combine(sig, WEIGHTS, df.index)
    return out


def top_n(df: pd.DataFrame, current_class: float, n: int = None) -> pd.DataFrame:
    n = n or BOX_N
    scored = score_race(df, current_class)
    s = scored["_score"].to_numpy(dtype=float)
    order = np.argsort(-np.where(np.isnan(s), -1e18, s), kind="stable")
    return scored.iloc[order[:n]]
