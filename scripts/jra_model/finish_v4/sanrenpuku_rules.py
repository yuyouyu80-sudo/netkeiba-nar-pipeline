# -*- coding: utf-8 -*-
"""finish_v4 F4: 3連複の買い目(tickets)と印、精算(2026-10-11、プラン第3版 F4)。オッズは読まない(テストで確認)。

- 主のルール(開発で決定、事前登録に書く): 組の確率の高い順に **20点**(N20)。
  決め方: 現行の印(P2×Q1×No.644)の点数は 10/3〜10/10 の発走前の記録 28 レースで平均 17.4 点・中央値 16 点。点数をそろえて比べるため、
  候補 N=10・20・30 のうちいちばん近い 20 を主にする。累積確率で止める方式は F0 の開発で固定点数より上積みが悪かったので主にしない。
  N10・N30 は副次として記録する。
- 印(表示用、買い目から作る): ◎=全ての買い目に入る馬、〇=半数以上の買い目に入る馬、▲=それ以外で買い目に入る馬。
- 印の JSON(正本は tickets): {"meta": {...}, "races": {race_id: {"umaban", "horse_name", "tickets", "p_ticket", "p_top3", "marks"}}}。
  marks は post_my_marks.py の --source でそのまま読める形({"馬番": "◎"})。
- 精算(settle): 同着で当たり組が複数あれば全部を数える。出走取消・競走除外の馬を含む買い目は返還(賭け金にも払戻にも数えない)。
  競走中止の馬を含む買い目は外れ。
"""
import itertools

import numpy as np

import pl_ext as P

MAIN_RULE = "N20"
SUB_RULES = ("N10", "N30")


def pick(p: np.ndarray, rule: str) -> np.ndarray:
    o = np.argsort(-p, kind="stable")
    assert rule[0] == "N"
    return o[:int(rule[1:])]


def race_tickets(umaban, u, lam, rule: str = MAIN_RULE) -> dict:
    """1レース: 効用 u(出走馬の並び)→ 買い目・その確率・P(3着内)・印。"""
    um = np.asarray(umaban, int)
    combos, p = P.triple_probs(np.asarray(u, float), "pl", tuple(lam))
    ii = pick(p, rule)
    tk = [sorted(int(x) for x in um[combos[i]]) for i in ii]
    top3 = P.top3_probs(combos, p, len(um))
    return {"tickets": tk, "p_ticket": [round(float(p[i]), 6) for i in ii],
            "p_top3": {str(int(a)): round(float(b), 4) for a, b in zip(um, top3)}, "marks": marks_from_tickets(tk),
            "p_covered": round(float(p[ii].sum()), 4)}


def marks_from_tickets(tickets) -> dict:
    n = len(tickets)
    cnt = {}
    for t in tickets:
        for h in t:
            cnt[h] = cnt.get(h, 0) + 1
    out = {}
    for h, c in sorted(cnt.items()):
        out[str(h)] = "◎" if c == n else ("〇" if 2 * c >= n else "▲")
    return out


def settle(tickets, winners, scratched=()) -> dict:
    """tickets: [[a,b,c],...]。winners: [(frozenset 3頭, 払戻円/100円)](同着なら複数)。scratched: 取消・除外の馬番。

    戻り値: 点数・返還の点数・賭け金(円)・払戻(円)・的中。"""
    sc = set(int(x) for x in scratched)
    win = {frozenset(s): v for s, v in winners}
    live = [t for t in tickets if not (set(t) & sc)]
    ret = sum(win.get(frozenset(t), 0) for t in live)
    return {"n": len(tickets), "n_refund": len(tickets) - len(live), "stake": 100 * len(live), "ret": int(ret),
            "hit": bool(ret > 0)}


def all_combos(n: int):
    return list(itertools.combinations(range(n), 3))
