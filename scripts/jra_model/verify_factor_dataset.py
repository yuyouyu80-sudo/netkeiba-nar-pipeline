# -*- coding: utf-8 -*-
"""factor_database.jsonの集計結果が、既存の検証済みインフラ(jra_backtest.settle)で
独立に計算した数値と一致することを確認する(計画セクション7の検証手順)。
"""
import itertools
import json
import sys
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"

sys.path.insert(0, str(LIB_DIR))
import jra_backtest as JB  # noqa: E402
import jra_dataset_wide as JDW  # noqa: E402
from jra_factor_registry import horse_qualifies  # noqa: E402

DB = json.loads((DATA_DIR / "factor_database.json").read_text(encoding="utf-8"))
FACTOR_GROUPS = DB["factor_groups"]


def aggregate_reference(races_json, actual, selected_by_group, bet_type):
    """factor_database.jsonのhorses(dict)に対しhorse_qualifies()でフィルタし、
    jra_backtest.settleで決済する独立実装(JS側と同じロジックをPythonでもう一度書く
    のではなく、settle()という既存の検証済み関数に投げる点がJS実装との違い)。"""
    total_stake, total_return, n_races, n_points = 0, 0, 0, 0
    for race in races_json:
        idx = [i for i, h in enumerate(race["horses"]) if horse_qualifies(h, selected_by_group)]
        if not idx:
            continue
        if bet_type not in race["offered_bet_types"]:
            continue
        umabans = [race["horses"][i]["umaban"] for i in idx]
        # JS側combosFor()はwaku欠損(null/NaN)を枠連の組合せ対象から除外するが、
        # jra_backtest.combos_forは素朴にint()するため欠損があるとTypeErrorになる
        # (2026-09-03、シニアエンジニアレビューで指摘。JB.combos_for自体は重み探索で
        # 広く使われる検証済み関数のため無改造とし、ここ(消費側)でJS側と同じ欠損除外を行う)。
        wakus = [w for w in (race["horses"][i]["waku"] for i in idx) if w is not None]
        combos = JB.combos_for(umabans, wakus)[bet_type]
        if not combos:
            continue
        n_races += 1
        n_points += len(combos)
        total_stake += len(combos) * JB.UNIT
        raw = actual.get(race["race_id"], {}).get(bet_type, {})
        for c in combos:
            if bet_type in ("単勝", "複勝"):
                key = c  # jra_backtest.combos_forは単勝/複勝を素のint listで返す(ラップしない)
            elif bet_type in ("馬単", "3連単"):
                key = tuple(c)
            else:
                key = frozenset(c)
            total_return += raw.get(key, 0)
    rate = total_return / total_stake * 100 if total_stake else 0.0
    return {"n_races": n_races, "n_points": n_points, "stake": total_stake,
            "return": total_return, "rate": rate}


def main():
    wide = JDW.load(rebuild=False)
    actual = wide["actual"]
    by_race_id = {r["race_id"]: r for r in DB["races"]}

    cases = [
        ("通常戦score1位のみ×単勝",
         {"race_type": {"rt_normal"}, "score_rank": {"score_top1"}}, "単勝"),
        ("通常戦score1〜3位×複勝",
         {"race_type": {"rt_normal"}, "score_rank": {"score_top3"}}, "複勝"),
        ("新馬戦score1位のみ×単勝",
         {"race_type": {"rt_shinba"}, "score_rank": {"score_top1"}}, "単勝"),
        ("未勝利戦score1〜2位×ワイド",
         {"race_type": {"rt_mishoubi"}, "score_rank": {"score_top2"}}, "ワイド"),
        ("本紙◎のみ×単勝(全レースタイプ)",
         {"mark_honshi": {"mark_honmei"}}, "単勝"),
    ]

    all_ok = True
    for label, sel, bt in cases:
        result = aggregate_reference(DB["races"], actual, sel, bt)
        print(f"[{label}] n_races={result['n_races']} n_points={result['n_points']} "
              f"stake={result['stake']} return={result['return']} rate={result['rate']:.2f}%")
    print("\n(この結果をJS側のブラウザ再計算と目視突合すること。stake/returnが円単位で"
          "一致すれば決済ロジック移植は正しい)")


if __name__ == "__main__":
    main()
