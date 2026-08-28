# -*- coding: utf-8 -*-
"""JRA 2頭軸流し買いの決済エンジン(2026-08-28新設、JRA Stage2 Phase J3)。
scripts/jra_model/jra_axis_backtest.py(1頭軸)の2頭軸版。settle()のdictルックアップ加算
コアはjra_backtest.settle()と同じ設計を踏襲する。

1頭軸(jra_axis_backtest.py)は「軸1頭を固定し、残りK-1頭(相手)との組合せのみ」を買うが、
2頭軸は「軸2頭を固定し、残りの相手候補との組合せのみ」を買う。対象は3連複・3連単の2券種
(軸2頭を選ぶという構造上、2頭で完結する馬連・ワイド・馬単・枠連には「2頭軸」という
概念自体が無い——軸2頭を固定した時点でベット自体が1点に定まってしまうため、ユーザー
依頼の「2頭軸」はexoticな3頭系券種にのみ意味を持つ)。

3連単はさらに2方式に分ける(ユーザー依頼「軸2頭box+相手Kを3着に、またはaxis順序固定+
相手K」通り):
  * 2軸box: 軸2頭のどちらが1着・2着でもよい(2通り)×相手が3着、の2K通り
  * 2軸固定: 軸1が1着・軸2が2着に固定×相手が3着、のK通り(点数はboxの半分)

賭け金は1点100円固定(UNIT、他のbacktestモジュールと同一)。
"""
import itertools
from typing import Dict, Sequence, Tuple

import numpy as np

UNIT = 100
BET_TYPES_2AXIS = ["3連複", "3連単_2軸box", "3連単_2軸固定"]
UNDERLYING_BET_TYPE = {
    "3連複": "3連複", "3連単_2軸box": "3連単", "3連単_2軸固定": "3連単",
}


def combos_for_2axis(axis1_umaban: int, axis2_umaban: int,
                     partner_umabans: Sequence[int]) -> Dict[str, list]:
    """軸2頭+相手複数頭から、券種ごとに賭ける組合せのリストを返す。"""
    p = list(int(x) for x in partner_umabans)
    a1, a2 = int(axis1_umaban), int(axis2_umaban)
    return {
        "3連複": [frozenset({a1, a2, x}) for x in p],
        "3連単_2軸box": [t for x in p for t in ((a1, a2, x), (a2, a1, x))],
        "3連単_2軸固定": [(a1, a2, x) for x in p],
    }


def settle_2axis(actual_race: dict, axis1_umaban: int, axis2_umaban: int,
                 partner_umabans: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """1レース分。(stake[券種], return[券種]) を BET_TYPES_2AXIS の順で返す。"""
    combos = combos_for_2axis(axis1_umaban, axis2_umaban, partner_umabans)
    stake = np.empty(len(BET_TYPES_2AXIS), dtype=np.int64)
    ret = np.empty(len(BET_TYPES_2AXIS), dtype=np.int64)
    for i, bt in enumerate(BET_TYPES_2AXIS):
        cs = combos[bt]
        m = actual_race.get(UNDERLYING_BET_TYPE[bt], {})
        stake[i] = len(cs) * UNIT
        ret[i] = sum(m.get(c, 0) for c in cs)
    return stake, ret


class MultiAxisSettler:
    """レースごとに「(軸1の行インデックス, 軸2の行インデックス, 相手の行インデックス集合)
    →(stake, return)」を遅延キャッシュするルックアップ表(jra_axis_backtest.AxisSettlerと
    同一設計)。umaban配列はレース内の行順(馬柱CSVの並び)を保った整数配列で渡す。"""

    def __init__(self, races: list, actual: dict):
        self.race_ids = [r["race_id"] for r in races]
        self.umabans = [r["df"]["umaban"].astype(int).to_numpy() for r in races]
        self.actuals = [actual.get(rid, {}) for rid in self.race_ids]
        self._memo = [dict() for _ in races]

    def returns_for(self, picks: list) -> Tuple[np.ndarray, np.ndarray]:
        """picks: レースごとの「選んだ行インデックスの配列」。picks[i][0]が軸1、picks[i][1]が
        軸2、picks[i][2:]が相手。picksは必ずコンストラクタに渡したracesと同じ全レース分・
        同じ並び順で渡すこと(AxisSettlerと同じ契約)。"""
        s = np.empty((len(picks), len(BET_TYPES_2AXIS)), dtype=np.int64)
        r = np.empty((len(picks), len(BET_TYPES_2AXIS)), dtype=np.int64)
        for i, idx in enumerate(picks):
            s[i], r[i] = self._settle_one(i, idx)
        return s, r

    def returns_for_at(self, race_idx_pick_pairs: list) -> Tuple[np.ndarray, np.ndarray]:
        """race_idx_pick_pairs: [(元のレースindex, 選んだ行インデックスの配列), ...]のリスト。
        任意の部分集合・任意の順序で問い合わせられる(jra_axis_backtest.AxisSettlerと同一設計、
        2026-08-21のバグ修正を踏まえ最初からこちらを部分集合評価用に用意する)。"""
        n = len(race_idx_pick_pairs)
        s = np.empty((n, len(BET_TYPES_2AXIS)), dtype=np.int64)
        r = np.empty((n, len(BET_TYPES_2AXIS)), dtype=np.int64)
        for k, (race_idx, idx) in enumerate(race_idx_pick_pairs):
            s[k], r[k] = self._settle_one(race_idx, idx)
        return s, r

    def _settle_one(self, race_idx: int, idx) -> Tuple[np.ndarray, np.ndarray]:
        a1_idx, a2_idx = int(idx[0]), int(idx[1])
        partner_idx = tuple(sorted(int(x) for x in idx[2:]))
        key = (a1_idx, a2_idx, partner_idx)
        memo = self._memo[race_idx]
        if key not in memo:
            u = self.umabans[race_idx]
            a1_u, a2_u = int(u[a1_idx]), int(u[a2_idx])
            partner_u = [int(u[j]) for j in partner_idx]
            memo[key] = settle_2axis(self.actuals[race_idx], a1_u, a2_u, partner_u)
        return memo[key]


def picks_2axis_from_scores(scores: np.ndarray, ranges: list, race_indices, k_partners: int) -> list:
    """スコア降順で上位2頭を軸1・軸2、続くk_partners頭を相手としたpicksを作る
    (jra_logistic.picks_from_scoresと同じ「レース内行インデックス配列」形式だが、
    先頭2要素が軸という意味づけがMultiAxisSettlerの契約)。"""
    picks = []
    for ri in race_indices:
        start, end = ranges[ri]
        s = scores[start:end]
        order = np.argsort(-s, kind="stable")
        k = min(2 + k_partners, len(order))
        picks.append(order[:k])
    return picks


def summarize(stake: np.ndarray, ret: np.ndarray) -> dict:
    """券種別の 的中レース数 / 的中率 / 総賭金 / 総払戻 / 回収率 を返す。"""
    out = {}
    n = stake.shape[0]
    for i, bt in enumerate(BET_TYPES_2AXIS):
        st, rt = int(stake[:, i].sum()), int(ret[:, i].sum())
        hits = int((ret[:, i] > 0).sum())
        out[bt] = {
            "races": n, "hit_races": hits,
            "hit_rate_pct": round(hits / n * 100, 1) if n else 0.0,
            "stake": st, "return": rt,
            "return_rate_pct": round(rt / st * 100, 1) if st else 0.0,
        }
    return out
