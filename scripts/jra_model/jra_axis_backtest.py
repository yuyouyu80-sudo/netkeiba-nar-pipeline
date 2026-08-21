# -*- coding: utf-8 -*-
"""JRA 1頭軸流し買いの決済エンジン(高速版)。scripts/jra_model/jra_backtest.pyの軸流し版。

BOX買い(jra_backtest.py)は「選んだK頭の全通り」を買うが、軸流しは「軸1頭を固定し、
残りK-1頭(相手)との組合せのみ」を買う。的中判定コア(dictルックアップして加算)は
jra_backtest.settle()と全く同じ設計を踏襲する(券種による分岐は組合せの型(frozenset/tuple)の
違いだけで、settle自体は券種に依存しない)。

対象は馬連・ワイド・3連複・馬単・3連単の5券種(単勝・複勝・枠連は「軸+相手」という構造を
持たないため対象外、ユーザー依頼通り)。馬単・3連単はさらに2方式に分ける:
  * 軸流し: 軸が指定着順(1着)に来たときのみ的中(標準的な軸流し)
  * マルチ: 軸が絡む着順ならどこでも的中(点数は軸流しの2倍/6倍、賭金もその分増える)
方向性の無い馬連・ワイド・3連複は1方式のみ。

賭け金は1点100円固定(UNIT、jra_backtest.pyと同一)。
"""
import itertools
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

UNIT = 100
BET_TYPES_AXIS = [
    "馬連", "ワイド", "3連複",
    "馬単_軸流し", "馬単_マルチ",
    "3連単_軸流し", "3連単_マルチ",
]
# レポート表示用: 実際の払戻データ上の券種名(data/payouts/*.csvのbet_type列)への対応。
# 軸流し/マルチは同じ元の券種(馬単/3連単)の払戻データを別の組合せ集合で引くだけ。
UNDERLYING_BET_TYPE = {
    "馬連": "馬連", "ワイド": "ワイド", "3連複": "3連複",
    "馬単_軸流し": "馬単", "馬単_マルチ": "馬単",
    "3連単_軸流し": "3連単", "3連単_マルチ": "3連単",
}


def combos_for_axis(axis_umaban: int, partner_umabans: Sequence[int]) -> Dict[str, list]:
    """軸1頭+相手複数頭から、券種ごとに賭ける組合せのリストを返す。"""
    p = list(int(x) for x in partner_umabans)
    a = int(axis_umaban)
    return {
        "馬連": [frozenset({a, x}) for x in p],
        "ワイド": [frozenset({a, x}) for x in p],
        "3連複": [frozenset({a, x, y}) for x, y in itertools.combinations(p, 2)],
        # 軸1着固定流し: 軸が1着のときのみ的中
        "馬単_軸流し": [(a, x) for x in p],
        "3連単_軸流し": [(a, x, y) for x, y in itertools.permutations(p, 2)],
        # 軸マルチ: 軸+相手の集合で作れる着順を全通り(軸がどの着順でも的中)
        "馬単_マルチ": [t for x in p for t in ((a, x), (x, a))],
        "3連単_マルチ": [
            t for x, y in itertools.combinations(p, 2)
            for t in itertools.permutations((a, x, y), 3)
        ],
    }


def settle_axis(actual_race: dict, axis_umaban: int, partner_umabans: Sequence[int]
                ) -> Tuple[np.ndarray, np.ndarray]:
    """1レース分。(stake[券種], return[券種]) を BET_TYPES_AXIS の順で返す。"""
    combos = combos_for_axis(axis_umaban, partner_umabans)
    stake = np.empty(len(BET_TYPES_AXIS), dtype=np.int64)
    ret = np.empty(len(BET_TYPES_AXIS), dtype=np.int64)
    for i, bt in enumerate(BET_TYPES_AXIS):
        cs = combos[bt]
        m = actual_race.get(UNDERLYING_BET_TYPE[bt], {})
        stake[i] = len(cs) * UNIT
        ret[i] = sum(m.get(c, 0) for c in cs)
    return stake, ret


class AxisSettler:
    """レースごとに「(軸の行インデックス, 相手の行インデックス集合)→(stake, return)」を
    遅延キャッシュするルックアップ表。BoxSettlerと違い、軸流しは組合せ空間が
    n * C(n-1, k-1) と大きくなる(軸の選び方の分だけ増える)ため全列挙はせず、
    実際に returns_for() で問い合わせが来たキーだけをその都度計算してメモ化する
    (重み探索で実際に選ばれる(軸, 相手集合)の種類は候補パターン数程度に収まるため、
    全列挙するより遥かに少ない計算量で済む)。

    umaban配列はレース内の行順(馬柱CSVの並び)を保った整数配列で渡す。
    """

    def __init__(self, races: list, actual: dict, box_n: int = 5):
        self.box_n = box_n
        self.race_ids = [r["race_id"] for r in races]
        self.dates = np.array([r["kaisai_date"] for r in races])
        self.umabans = [r["df"]["umaban"].astype(int).to_numpy() for r in races]
        self.actuals = [actual.get(rid, {}) for rid in self.race_ids]
        self._memo = [dict() for _ in races]

    def returns_for(self, picks: list) -> Tuple[np.ndarray, np.ndarray]:
        """picks: レースごとの「選んだ行インデックスの配列」。picks[i][0]が軸、
        picks[i][1:]が相手。**picksは必ずコンストラクタに渡したracesと同じ全レース分・
        同じ並び順で渡すこと**(BoxSettlerと同じ契約。i番目のpicksがi番目のレースに
        対応する前提でself.umabans[i]等をenumerateの位置でそのまま引く)。開催日の一部
        レースだけを部分集合として渡したい場合は、代わりにreturns_for_at()を使うこと。"""
        s = np.empty((len(picks), len(BET_TYPES_AXIS)), dtype=np.int64)
        r = np.empty((len(picks), len(BET_TYPES_AXIS)), dtype=np.int64)
        for i, idx in enumerate(picks):
            s[i], r[i] = self._settle_one(i, idx)
        return s, r

    def returns_for_at(self, race_idx_pick_pairs: list) -> Tuple[np.ndarray, np.ndarray]:
        """race_idx_pick_pairs: [(元のレースindex, 選んだ行インデックスの配列), ...]の
        リスト。returns_for()と違い、任意の部分集合・任意の順序で問い合わせられる
        (confidence_sweepの「高確信度Nレース/日」のような部分集合評価向け)。"""
        n = len(race_idx_pick_pairs)
        s = np.empty((n, len(BET_TYPES_AXIS)), dtype=np.int64)
        r = np.empty((n, len(BET_TYPES_AXIS)), dtype=np.int64)
        for k, (race_idx, idx) in enumerate(race_idx_pick_pairs):
            s[k], r[k] = self._settle_one(race_idx, idx)
        return s, r

    def _settle_one(self, race_idx: int, idx) -> Tuple[np.ndarray, np.ndarray]:
        axis_idx = int(idx[0])
        partner_idx = tuple(sorted(int(x) for x in idx[1:]))
        key = (axis_idx, partner_idx)
        memo = self._memo[race_idx]
        if key not in memo:
            u = self.umabans[race_idx]
            axis_u = int(u[axis_idx])
            partner_u = [int(u[j]) for j in partner_idx]
            memo[key] = settle_axis(self.actuals[race_idx], axis_u, partner_u)
        return memo[key]


def summarize(stake: np.ndarray, ret: np.ndarray) -> dict:
    """券種別の 的中レース数 / 的中率 / 総賭金 / 総払戻 / 回収率 を返す。"""
    out = {}
    n = stake.shape[0]
    for i, bt in enumerate(BET_TYPES_AXIS):
        st, rt = int(stake[:, i].sum()), int(ret[:, i].sum())
        hits = int((ret[:, i] > 0).sum())
        out[bt] = {
            "races": n, "hit_races": hits,
            "hit_rate_pct": round(hits / n * 100, 1) if n else 0.0,
            "stake": st, "return": rt,
            "return_rate_pct": round(rt / st * 100, 1) if st else 0.0,
        }
    return out
