# -*- coding: utf-8 -*-
"""JRA BOX買いの決済エンジン(高速版)。scripts/nar_model/nar_backtest.pyのJRA移植版。

重み探索では同じ重みベクトル空間を何千回も走査するため、レース単位で「上位K頭の組合せ →
券種別払戻」を事前計算しておき、以後はルックアップだけで済ませる。

NAR版との違い: NARはwaku列が常に空のため枠連を対象外にしているが、JRAは枠連が有効なので
8券種すべてを扱う。枠連は選択した馬の「枠番の集合」(頭数分ではなく重複除去した枠の組合せ)
から作る(confidence_sweep_v2.pyのbox_return()と同じ規約)。

賭け金は1点100円固定(UNIT)。
"""
import itertools
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

UNIT = 100
BET_TYPES = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "3連複", "3連単"]


def combos_for(umabans: Sequence[int], wakus: Sequence[int]) -> Dict[str, list]:
    u = list(umabans)
    w = sorted(set(int(x) for x in wakus))
    return {
        "単勝": u,
        "複勝": u,
        "枠連": [frozenset(c) for c in itertools.combinations(w, 2)],
        "馬連": [frozenset(c) for c in itertools.combinations(u, 2)],
        "ワイド": [frozenset(c) for c in itertools.combinations(u, 2)],
        "馬単": list(itertools.permutations(u, 2)),
        "3連複": [frozenset(c) for c in itertools.combinations(u, 3)],
        "3連単": list(itertools.permutations(u, 3)),
    }


def settle(actual_race: dict, umabans: Sequence[int], wakus: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """1レース分。(stake[券種], return[券種]) を BET_TYPES の順で返す。"""
    combos = combos_for(umabans, wakus)
    stake = np.empty(len(BET_TYPES), dtype=np.int64)
    ret = np.empty(len(BET_TYPES), dtype=np.int64)
    for i, bt in enumerate(BET_TYPES):
        cs = combos[bt]
        m = actual_race.get(bt, {})
        stake[i] = len(cs) * UNIT
        ret[i] = sum(m.get(c, 0) for c in cs)
    return stake, ret


class BoxSettler:
    """レースごとに「選んだK頭の組合せ→(stake, return)」を全列挙してキャッシュする。

    umaban配列はレース内の行順(馬柱CSVの並び)を保った整数配列で渡す。
    lookup は行インデックスのタプル(昇順にソートしたもの)をキーにする。
    """

    def __init__(self, races: list, actual: dict, box_n: int = 5):
        self.box_n = box_n
        self.race_ids = [r["race_id"] for r in races]
        self.dates = np.array([r["kaisai_date"] for r in races])
        self.umabans = [r["df"]["umaban"].astype(int).to_numpy() for r in races]
        self.wakus = [pd.to_numeric(r["df"]["waku"], errors="coerce").to_numpy() for r in races]
        self.tables = []
        for r, u, w in zip(races, self.umabans, self.wakus):
            a = actual.get(r["race_id"], {})
            n = len(u)
            k = min(box_n, n)
            table = {}
            for idx in itertools.combinations(range(n), k):
                sel_u = u[list(idx)]
                sel_w = [int(x) for x in w[list(idx)] if not np.isnan(x)]
                table[idx] = settle(a, sel_u, sel_w)
            self.tables.append(table)

    def returns_for(self, picks: list) -> Tuple[np.ndarray, np.ndarray]:
        """picks: レースごとの「選んだ行インデックスの配列」。
        戻り値: (stake[n_races, n_bets], return[n_races, n_bets])"""
        s = np.empty((len(picks), len(BET_TYPES)), dtype=np.int64)
        r = np.empty((len(picks), len(BET_TYPES)), dtype=np.int64)
        for i, idx in enumerate(picks):
            key = tuple(sorted(int(x) for x in idx))
            s[i], r[i] = self.tables[i][key]
        return s, r


def summarize(stake: np.ndarray, ret: np.ndarray) -> dict:
    """券種別の 的中レース数 / 的中率 / 総賭金 / 総払戻 / 回収率 を返す。"""
    out = {}
    n = stake.shape[0]
    for i, bt in enumerate(BET_TYPES):
        st, rt = int(stake[:, i].sum()), int(ret[:, i].sum())
        hits = int((ret[:, i] > 0).sum())
        out[bt] = {
            "races": n, "hit_races": hits,
            "hit_rate_pct": round(hits / n * 100, 1) if n else 0.0,
            "stake": st, "return": rt,
            "return_rate_pct": round(rt / st * 100, 1) if st else 0.0,
        }
    return out
