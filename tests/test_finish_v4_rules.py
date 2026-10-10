# -*- coding: utf-8 -*-
"""finish_v4 F4(sanrenpuku_rules.py): 買い目・印・精算(同着・返還・中止)。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "jra_model" / "finish_v4"))
import sanrenpuku_rules as R  # noqa: E402


def test_tickets_are_top_combos_and_marks():
    um = [1, 2, 3, 4, 5, 6]
    u = [2.0, 1.5, 1.0, 0.0, -0.5, -1.0]
    r = R.race_tickets(um, u, (1.0, 0.9, 0.8), rule="N10")
    assert len(r["tickets"]) == 10
    assert r["tickets"][0] == [1, 2, 3]
    assert r["p_ticket"] == sorted(r["p_ticket"], reverse=True)
    assert abs(sum(r["p_top3"].values()) - 3.0) < 1e-3   # 4桁に丸めた値の和
    # 馬1は10組中の多くに入る。印は買い目の数え上げどおり
    cnt = sum(1 in t for t in r["tickets"])
    assert r["marks"]["1"] == ("◎" if cnt == 10 else ("〇" if 2 * cnt >= 10 else "▲"))
    assert set(r["marks"]) <= {str(x) for x in um}


def test_marks_rule():
    t = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [1, 2, 5]]
    m = R.marks_from_tickets(t)
    assert m == {"1": "◎", "2": "〇", "3": "〇", "4": "〇", "5": "▲"}


def test_settle_ties_refunds_and_dnf():
    tickets = [[1, 2, 3], [1, 2, 4], [1, 2, 5], [3, 4, 5]]
    # 同着で当たり組が2つ(1-2-3 と 1-2-4)
    s = R.settle(tickets, [(frozenset({1, 2, 3}), 1200), (frozenset({1, 2, 4}), 900)])
    assert s == {"n": 4, "n_refund": 0, "stake": 400, "ret": 2100, "hit": True}
    # 5番が取消: 5を含む2点は返還
    s = R.settle(tickets, [(frozenset({1, 2, 3}), 1200)], scratched=[5])
    assert s["n_refund"] == 2 and s["stake"] == 200 and s["ret"] == 1200
    # 中止の馬(4番)を含む組は外れのまま(返還ではない)
    s = R.settle([[1, 2, 4]], [(frozenset({1, 2, 3}), 1200)])
    assert s["stake"] == 100 and s["ret"] == 0 and not s["hit"]


def test_rules_module_does_not_read_odds():
    import ast
    src = (ROOT / "scripts" / "jra_model" / "finish_v4" / "sanrenpuku_rules.py").read_text(encoding="utf-8")
    names = {a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) and n.module}
    assert not names & {"market_yardstick", "f0_market_eval", "fmark_weekend", "load_score5"}
    np.testing.assert_equal(R.MAIN_RULE, "N20")
