# -*- coding: utf-8 -*-
"""単勝/複勝1点買いの決済エンジン(2026-08-22新設、Step1: 市場アンカー型条件付きロジット用)。

jra_backtest.BoxSettler(固定N頭の全組合せを事前キャッシュ)・jra_axis_backtest.AxisSettler
(軸+相手の組合せを遅延キャッシュ)と異なり、単勝/複勝は「1頭=1点」の辞書引きだけで済むため
組合せ列挙は不要。期待値(EV)ベース選抜は各レースで条件を満たす馬の数が0〜n頭と可変
(BOX買いのような固定box_nではない)ため、`returns_for(picks)`は可変長のpicks(空配列=
そのレースは見送り)を受け付ける設計にする。

賭け金は1点100円固定(UNIT、jra_backtest.pyと同一)。
"""
from typing import Sequence, Tuple

import numpy as np

UNIT = 100
BET_TYPES_SINGLES = ["単勝", "複勝"]


def settle_singles(actual_race: dict, umabans: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """1レース分。(stake[券種], return[券種]) をBET_TYPES_SINGLESの順で返す。
    umabansは0頭でもよい(見送りレース=stake/return全0)。"""
    stake = np.zeros(len(BET_TYPES_SINGLES), dtype=np.int64)
    ret = np.zeros(len(BET_TYPES_SINGLES), dtype=np.int64)
    for i, bt in enumerate(BET_TYPES_SINGLES):
        m = actual_race.get(bt, {})
        for u in umabans:
            stake[i] += UNIT
            ret[i] += m.get(int(u), 0)
    return stake, ret


class SinglesSettler:
    """レースごとに「選んだ馬の行インデックス集合→(stake, return)」を都度計算する
    (BoxSettlerと違い1レースあたりの選択数が小さい[0〜数頭]ため事前キャッシュ不要)。"""

    def __init__(self, races: list, actual: dict):
        self.race_ids = [r["race_id"] for r in races]
        self.umabans = [r["df"]["umaban"].astype(int).to_numpy() for r in races]
        self.actual = actual

    def returns_for(self, picks: list) -> Tuple[np.ndarray, np.ndarray]:
        """picks: レースごとの「選んだ行インデックスの配列」(空配列/Noneのそのレースは見送り)。
        戻り値: (stake[n_races, n_bets], return[n_races, n_bets])"""
        n = len(picks)
        s = np.zeros((n, len(BET_TYPES_SINGLES)), dtype=np.int64)
        r = np.zeros((n, len(BET_TYPES_SINGLES)), dtype=np.int64)
        for i, idx in enumerate(picks):
            if idx is None or len(idx) == 0:
                continue
            sel_umaban = self.umabans[i][np.asarray(list(idx), dtype=int)]
            a = self.actual.get(self.race_ids[i], {})
            st, rt = settle_singles(a, sel_umaban)
            s[i], r[i] = st, rt
        return s, r
