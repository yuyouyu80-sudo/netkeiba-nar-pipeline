# -*- coding: utf-8 -*-
"""JRAフォーメーション買いの決済エンジン(2026-08-28新設、JRA Stage2 Phase J3)。
scripts/jra_model/jra_axis_backtest.py/jra_multi_axis_backtest.pyと同じ設計思想
(settle()のdictルックアップ加算コアを踏襲)。

軸流し(1頭軸・2頭軸)は「軸を固定し相手を広げる」買い方だが、フォーメーションは
「着順ごとに独立した候補集合を割り当てる」買い方(1着候補A×2着候補B[×3着候補C]の
全組合せ、同じ馬の重複は除く)。対象は馬単・3連単の2券種(方向性の無い券種にはそもそも
「何着候補か」という区別が無いため対象外、jra_axis_backtest.pyが馬単・3連単のみ
軸流し/マルチの2方式を持つのと同じ理由)。

候補集合A/B[/C]は「スコア上位A_size/B_size[/C_size]頭」という設計を想定し、通常は
A_size<=B_size[<=C_size](1着候補を絞り、2着・3着候補を広げる、標準的なフォーメーション
買いの発想)。呼び出し側(fit_fn)がスコアから候補集合を作る。

賭け金は1点100円固定(UNIT、他のbacktestモジュールと同一)。
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

UNIT = 100
BET_TYPES_FORMATION = ["馬単_フォーメーション", "3連単_フォーメーション"]


def combos_for_formation(top_a: Sequence[int], top_b: Sequence[int],
                         top_c: Optional[Sequence[int]] = None) -> Dict[str, list]:
    """候補集合(umaban)から、券種ごとに賭ける組合せのリストを返す。top_cがNone/空なら
    3連単フォーメーションは0点(賭けない)。"""
    a, b = list(int(x) for x in top_a), list(int(x) for x in top_b)
    c = list(int(x) for x in top_c) if top_c else []
    out = {
        "馬単_フォーメーション": [(x, y) for x in a for y in b if x != y],
        "3連単_フォーメーション": [(x, y, z) for x in a for y in b for z in c
                              if x != y and y != z and x != z] if c else [],
    }
    return out


def settle_formation(actual_race: dict, top_a: Sequence[int], top_b: Sequence[int],
                     top_c: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """1レース分。(stake[券種], return[券種]) を BET_TYPES_FORMATION の順で返す。"""
    combos = combos_for_formation(top_a, top_b, top_c)
    stake = np.empty(len(BET_TYPES_FORMATION), dtype=np.int64)
    ret = np.empty(len(BET_TYPES_FORMATION), dtype=np.int64)
    for i, bt in enumerate(BET_TYPES_FORMATION):
        cs = combos[bt]
        underlying = "馬単" if bt == "馬単_フォーメーション" else "3連単"
        m = actual_race.get(underlying, {})
        stake[i] = len(cs) * UNIT
        ret[i] = sum(m.get(c, 0) for c in cs)
    return stake, ret


class FormationSettler:
    """レースごとに「(A候補の行インデックスtuple, B候補の行インデックスtuple,
    C候補の行インデックスtuple)→(stake, return)」を遅延キャッシュするルックアップ表
    (jra_axis_backtest.AxisSettlerと同一設計)。umaban配列はレース内の行順を保った
    整数配列で渡す。"""

    def __init__(self, races: list, actual: dict):
        self.race_ids = [r["race_id"] for r in races]
        self.umabans = [r["df"]["umaban"].astype(int).to_numpy() for r in races]
        self.actuals = [actual.get(rid, {}) for rid in self.race_ids]
        self._memo = [dict() for _ in races]

    def returns_for(self, picks: list) -> Tuple[np.ndarray, np.ndarray]:
        """picks: レースごとの (A候補の行インデックス配列, B候補の行インデックス配列,
        C候補の行インデックス配列またはNone) の3要素タプル。picksは必ずコンストラクタに
        渡したracesと同じ全レース分・同じ並び順で渡すこと(AxisSettlerと同じ契約)。"""
        s = np.empty((len(picks), len(BET_TYPES_FORMATION)), dtype=np.int64)
        r = np.empty((len(picks), len(BET_TYPES_FORMATION)), dtype=np.int64)
        for i, idx in enumerate(picks):
            s[i], r[i] = self._settle_one(i, idx)
        return s, r

    def returns_for_at(self, race_idx_pick_pairs: list) -> Tuple[np.ndarray, np.ndarray]:
        """任意の部分集合・任意の順序で問い合わせる版(jra_axis_backtest.AxisSettlerと同一設計)。"""
        n = len(race_idx_pick_pairs)
        s = np.empty((n, len(BET_TYPES_FORMATION)), dtype=np.int64)
        r = np.empty((n, len(BET_TYPES_FORMATION)), dtype=np.int64)
        for k, (race_idx, idx) in enumerate(race_idx_pick_pairs):
            s[k], r[k] = self._settle_one(race_idx, idx)
        return s, r

    def _settle_one(self, race_idx: int, idx) -> Tuple[np.ndarray, np.ndarray]:
        a_idx, b_idx, c_idx = idx
        a_idx = tuple(sorted(int(x) for x in a_idx))
        b_idx = tuple(sorted(int(x) for x in b_idx))
        c_idx = tuple(sorted(int(x) for x in c_idx)) if c_idx is not None and len(c_idx) else None
        key = (a_idx, b_idx, c_idx)
        memo = self._memo[race_idx]
        if key not in memo:
            u = self.umabans[race_idx]
            top_a = [int(u[j]) for j in a_idx]
            top_b = [int(u[j]) for j in b_idx]
            top_c = [int(u[j]) for j in c_idx] if c_idx is not None else None
            memo[key] = settle_formation(self.actuals[race_idx], top_a, top_b, top_c)
        return memo[key]


def picks_formation_from_scores(scores: np.ndarray, ranges: list, race_indices,
                                a_size: int, b_size: int, c_size: int = 0) -> list:
    """スコア降順の上位a_size/b_size/c_size頭をそれぞれ候補A/B/Cとしたpicksを作る
    (c_size=0なら馬単フォーメーション専用、Cはlen0のtupleでFormationSettlerが自動的に
    3連単を0点扱いする)。"""
    picks = []
    for ri in race_indices:
        start, end = ranges[ri]
        s = scores[start:end]
        order = np.argsort(-s, kind="stable")
        n = len(order)
        a = order[:min(a_size, n)]
        b = order[:min(b_size, n)]
        c = order[:min(c_size, n)] if c_size else np.array([], dtype=order.dtype)
        picks.append((a, b, c))
    return picks


def summarize(stake: np.ndarray, ret: np.ndarray) -> dict:
    """券種別の 的中レース数 / 的中率 / 総賭金 / 総払戻 / 回収率 を返す。"""
    out = {}
    n = stake.shape[0]
    for i, bt in enumerate(BET_TYPES_FORMATION):
        st, rt = int(stake[:, i].sum()), int(ret[:, i].sum())
        hits = int((ret[:, i] > 0).sum())
        out[bt] = {
            "races": n, "hit_races": hits,
            "hit_rate_pct": round(hits / n * 100, 1) if n else 0.0,
            "stake": st, "return": rt,
            "return_rate_pct": round(rt / st * 100, 1) if st else 0.0,
        }
    return out
