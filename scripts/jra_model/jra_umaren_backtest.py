# -*- coding: utf-8 -*-
"""馬連1点買いの決済エンジン(2026-08-22新設、Step1: 馬連特化モデリング)。

`jra_singles_backtest.py`(単勝/複勝、1頭=1点)と対になる、馬連(1着-2着の組・順不同)専用の
決済層。`jra_dataset.parse_combo("馬連", ...)`が返す`frozenset({u1,u2})`キーをそのまま
`actual[race_id]["馬連"]`から引く(`jra_archive_dataset.py`も同じ`parse_combo`を再利用済み
のため、`actual`辞書の形は単勝/複勝と共通)。

賭け金は1点100円固定(UNIT、他のbacktestモジュールと同一)。
"""
from typing import Optional, Sequence, Tuple

import numpy as np

UNIT = 100
BET_TYPES_UMAREN = ["馬連"]


def settle_umaren(actual_race: dict, umaban_pair: Optional[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """1レース分。(stake[1], return[1])を返す。umaban_pairはNoneまたは要素2の馬番配列
    (見送りレース=stake/return全0)。"""
    stake = np.zeros(len(BET_TYPES_UMAREN), dtype=np.int64)
    ret = np.zeros(len(BET_TYPES_UMAREN), dtype=np.int64)
    if umaban_pair is None or len(umaban_pair) == 0:
        return stake, ret
    m = actual_race.get("馬連", {})
    key = frozenset(int(u) for u in umaban_pair)
    stake[0] = UNIT
    ret[0] = m.get(key, 0)
    return stake, ret


class UmarenSettler:
    """レースごとに「選んだペア(行インデックス2つ)→(stake, return)」を都度計算する。"""

    def __init__(self, races: list, actual: dict):
        self.race_ids = [r["race_id"] for r in races]
        self.umabans = [r["df"]["umaban"].astype(int).to_numpy() for r in races]
        self.actual = actual

    def returns_for(self, picks: list) -> Tuple[np.ndarray, np.ndarray]:
        """picks: レースごとの「選んだ行インデックスの(i,j)タプル」(Noneはそのレース見送り)。
        戻り値: (stake[n_races, 1], return[n_races, 1])"""
        n = len(picks)
        s = np.zeros((n, len(BET_TYPES_UMAREN)), dtype=np.int64)
        r = np.zeros((n, len(BET_TYPES_UMAREN)), dtype=np.int64)
        for i, pair in enumerate(picks):
            if pair is None:
                continue
            i1, i2 = pair
            umaban_pair = (self.umabans[i][i1], self.umabans[i][i2])
            a = self.actual.get(self.race_ids[i], {})
            st, rt = settle_umaren(a, umaban_pair)
            s[i], r[i] = st, rt
        return s, r
