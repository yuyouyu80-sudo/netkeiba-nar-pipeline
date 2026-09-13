# -*- coding: utf-8 -*-
"""レースタイプ(通常戦/新馬戦/未勝利戦)ごとに本番と同じロジックで`_score`/`pred_rank`を
再現するモジュール。`jra_signals.py`は無改造、既存のヘルパー(`_shrink`/`_minmax`/`_col`/
`_pct`/`_num`/`compute_signals`/`combine_signals`)を最大限再利用する。

3レースタイプでpriors計算式が異なる点に注意(スクラッチパッドの`predict_shinba.py`/
`predict_mishoubi.py`から検証済みのロジックを移植、レビューA-2で確定した仕様):
  - 通常戦: `JS.make_priors()`(単純平均)。
  - 新馬戦・未勝利戦: `compute_priors_runs_weighted()`(runsで露出加重した平均。
    小標本outlierに引っ張られないための意図的な設計、`JS.make_priors()`とは別物)。

「本番予想スコア」の基本はBOX5(`winner_v3.json`、日次メインレポートが表示するモデル)。
BOX4/BOX3(`winner_box4.json`/`winner_box3.json`)はレビューA-3でv1スコープ外としたが、
2026-09-02にユーザー依頼で追加した(`normal_box4`/`normal_box3`という擬似race_typeとして
扱う。シグナル構成(LEGACY_SIGNALS 10本)・priorsは通常戦(`normal`)と完全に同一で、
重みだけが異なる。通常戦(246レース)にのみ存在し、新馬戦・未勝利戦にBOX4/BOX3モデルは
無い)。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"

sys.path.insert(0, str(LIB_DIR))
import jra_signals as JS  # noqa: E402

WINNER_PATHS = {
    "normal": DATA_DIR / "winner_v3.json",
    "normal_box4": DATA_DIR / "winner_box4.json",
    "normal_box3": DATA_DIR / "winner_box3.json",
    "shinba": DATA_DIR / "winner_shinba.json",
    "mishoubi": DATA_DIR / "winner_mishoubi.json",
}

# predict_shinba.py L39を移植(厩舎コメント評価コード。01=最良〜03=最悪、実測で単調性確認済み)。
COMMENT_RATING_MAP = {"01": 3, "02": 2, "03": 1}

SIGNAL_NAMES = {
    "normal": list(JS.LEGACY_SIGNALS),
    "normal_box4": list(JS.LEGACY_SIGNALS),
    "normal_box3": list(JS.LEGACY_SIGNALS),
    "mishoubi": list(JS.LEGACY_SIGNALS),
    "shinba": ["jt", "waku", "train", "sire", "bms", "comment"],
}

# predict_shinba.py/predict_mishoubi.py の SHRINK_SPECS 部分集合。キー定義は
# JS.SHRINK_SPECSと同一(実測確認済み)なのでフィルタで抽出するだけでよい。
_SHINBA_SHRINK_KEYS = ["jockey_win", "trainer_win", "waku_win",
                      "sire_win", "sire_place3", "sire_return",
                      "bms_win", "bms_place3", "bms_return"]
_MISHOUBI_SHRINK_KEYS = list(JS.SHRINK_SPECS.keys())  # 未勝利は通常戦と同じ全キーを使う


def _shrink_specs_for(keys: list) -> dict:
    return {k: JS.SHRINK_SPECS[k] for k in keys}


def compute_priors_runs_weighted(dfs: list, shrink_specs: dict) -> dict:
    """predict_shinba.compute_priors / predict_mishoubi.compute_priors と数式完全一致で
    移植(runsで露出加重した平均、Σ(rate×runs)/Σruns)。JS.make_priors()の単純平均とは
    意図的に異なる式なので絶対に代替しないこと(レビューA-2参照)。"""
    priors = {}
    for key, (rate_col, runs_col) in shrink_specs.items():
        rates = pd.concat([JS._pct(JS._col(df, rate_col)) for df in dfs], ignore_index=True)
        runs = pd.concat([JS._num(JS._col(df, runs_col)) for df in dfs], ignore_index=True).fillna(0.0)
        valid = rates.notna() & (runs > 0)
        total_runs = runs[valid].sum()
        priors[key] = (float((rates[valid] * runs[valid]).sum() / total_runs)
                      if total_runs > 0 else float(rates.mean(skipna=True)))
    return priors


def compute_comment_signal(df: pd.DataFrame) -> pd.Series:
    rating = JS._col(df, "stable_comment_rating_code").astype(str).str.strip().map(COMMENT_RATING_MAP)
    return JS._minmax(rating)


def build_shinba_signals(df: pd.DataFrame, priors: dict) -> dict:
    """predict_shinba.build_shinba_signals の移植(jt/waku/train/sire/bms/comment の6本)。"""
    specs = _shrink_specs_for(_SHINBA_SHRINK_KEYS)
    sig = {}
    jt_rate = pd.concat(
        [JS._shrink(df, "jockey_win", priors, specs), JS._shrink(df, "trainer_win", priors, specs)], axis=1
    ).mean(axis=1)
    sig["jt"] = JS._minmax(jt_rate)
    sig["waku"] = JS._minmax(JS._shrink(df, "waku_win", priors, specs))
    training = JS._col(df, "training_rank").astype(str).str.strip().str.upper().map(JS.TRAIN_RANK_MAP)
    sig["train"] = JS._minmax(training)
    sig["sire"] = JS._blend_minmax(
        JS._shrink(df, "sire_win", priors, specs), JS._shrink(df, "sire_place3", priors, specs),
        JS._shrink(df, "sire_return", priors, specs)
    )
    sig["bms"] = JS._blend_minmax(
        JS._shrink(df, "bms_win", priors, specs), JS._shrink(df, "bms_place3", priors, specs),
        JS._shrink(df, "bms_return", priors, specs)
    )
    sig["comment"] = compute_comment_signal(df)
    return sig


def load_weights(race_type: str) -> dict:
    """重みファイルを読み込み、キー集合を検証する(レビューA-1: 将来のモデル取り違えを
    防ぐ安価な防御、predict_mishoubi.pyのfail-fast方針を踏襲)。"""
    path = WINNER_PATHS[race_type]
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = payload["weights"]
    expected = set(SIGNAL_NAMES[race_type])
    if set(weights.keys()) != expected:
        raise ValueError(
            f"{path}: weights key mismatch, got {sorted(weights.keys())}, expected {sorted(expected)}"
        )
    return weights


def score_race(df: pd.DataFrame, race_name: str, race_type: str, priors: dict, weights: dict) -> pd.Series:
    """1レース分の_scoreを返す(index=df.index、高いほど有利)。"""
    if race_type in ("normal", "mishoubi", "normal_box4", "normal_box3"):
        current_class = JS._class_ordinal(race_name, JS.CLASS_ORDINAL)
        sig = JS.compute_signals(df, current_class, priors, JS.CLASS_ORDINAL)
        return JS.combine_signals({k: sig[k] for k in SIGNAL_NAMES[race_type]}, weights)
    if race_type == "shinba":
        sig = build_shinba_signals(df, priors)
        return JS.combine_signals({k: sig[k] for k in SIGNAL_NAMES["shinba"]}, weights)
    raise ValueError(f"unknown race_type: {race_type}")


def pred_rank_of(score: pd.Series) -> pd.Series:
    """本番(jra_eval.score_picks)と同じ並べ替え規則(np.argsort(-score, kind='stable')、
    NaNは最下位)でpred_rankを付ける。pandas.rank()は使わない(タイの扱いが本番と異なるため)。"""
    order = np.argsort(-score.fillna(-1e18).to_numpy(), kind="stable")
    rank = np.empty(len(score), dtype=int)
    rank[order] = np.arange(1, len(score) + 1)
    return pd.Series(rank, index=score.index)


def compute_priors_for_population(races_by_type: dict) -> dict:
    """race_type別のpriors辞書を一括計算する。races_by_type: {race_type: [df, ...]}。"""
    priors = {}
    priors["normal"] = JS.make_priors(races_by_type.get("normal", []))
    if races_by_type.get("shinba"):
        priors["shinba"] = compute_priors_runs_weighted(
            races_by_type["shinba"], _shrink_specs_for(_SHINBA_SHRINK_KEYS))
    if races_by_type.get("mishoubi"):
        priors["mishoubi"] = compute_priors_runs_weighted(
            races_by_type["mishoubi"], _shrink_specs_for(_MISHOUBI_SHRINK_KEYS))
    return priors
