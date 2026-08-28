# -*- coding: utf-8 -*-
"""lap33_fit型判定カバレッジ(64.2%)のボトルネックが何かを試算する(ユーザー依頼「採用
できる状況を考えて」への追補)。jra_lap33_signals.py本体は無改造、L33.MIN_GROUP_Nを
一時的に書き換えて比較するだけ。

背景: horse_type_score()は好走群/凡走群それぞれMIN_GROUP_N本(既定2)未満なら型不明
(0.0)を返す。n_lookback(既定20)を増やせば「データ蓄積を待てば自然にカバレッジが
上がる」のか、それともMIN_GROUP_N自体がボトルネックなのかを切り分ける。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_dataset as JD
import jra_lap33_signals as L33

data = JD.load(rebuild=False)
races = data["races"]
lap33_lookup = L33.load_lap33_lookup()
history_index = L33.build_history_index()


def coverage_for(min_group_n: int, n_lookback: int) -> tuple:
    orig = L33.MIN_GROUP_N
    L33.MIN_GROUP_N = min_group_n
    try:
        n_horses = n_known = 0
        for r in races:
            iso_date = L33.kaisai_date_to_iso(r["kaisai_date"])
            for hid in r["df"]["horse_id"].astype(str):
                score = L33.horse_type_score(hid, iso_date, history_index, lap33_lookup, n_lookback=n_lookback)
                n_horses += 1
                if score != 0.0:
                    n_known += 1
        return n_known, n_horses
    finally:
        L33.MIN_GROUP_N = orig


for min_n, lb in [(2, 20), (1, 20), (2, 40), (1, 40), (2, 9999)]:
    k, n = coverage_for(min_n, lb)
    print(f"MIN_GROUP_N={min_n} n_lookback={lb}: {k}/{n} ({k / n * 100:.1f}%)")
