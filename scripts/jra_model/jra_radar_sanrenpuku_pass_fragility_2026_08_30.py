# -*- coding: utf-8 -*-
"""3連複探索(jra_radar_yes_pattern_search_sanrenpuku_2026_08_30.py)で名目PASSした
主プロトコル候補5件について、「高配当レースの取り除き」に対する頑健性を測る追加診断
(2026-08-30)。

3連複は複勝+ワイドより配当分散が桁違いに大きく、246レース中の数レースの高配当で
回収率が決まりうる。ここでは各PASS候補について、
  * 全レースでの回収率
  * 払戻上位1レース / 上位3レース / 上位5レースを除いたときの回収率
  * 線形平均ベースラインについても同じ除外を適用したときの差分
を出して、名目上の +30pt 級の優位が何レースに支えられているかを可視化する。

探索スクリプト・モデル計算側は一切改造しない(読み取り専用のサイドカー)。
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jra_backtest as JB  # noqa: E402
import jra_dataset  # noqa: E402
import jra_eval as JE  # noqa: E402
import jra_lap33_signals as L33  # noqa: E402
import jra_radar_categories as RC  # noqa: E402
import jra_signals as JS  # noqa: E402
from jra_radar_axis_merge_search_2026_08_29 import LEVELS  # noqa: E402
from jra_radar_yes_pattern_search_sanrenpuku_2026_08_30 import (  # noqa: E402
    OBJ_BETS, _picks_from_vals, area_scores, scalar_scores, unique_cyclic_orders,
)

OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
LV8 = list(LEVELS.keys())[0]
LV5 = list(LEVELS.keys())[3]

# 主プロトコルで gate_pass=True だった5件 (level, box_n, method, order_idx)
TARGETS = [
    (LV8, 3, "area", 1171),
    (LV8, 4, "area", 1217),
    (LV8, 5, "area", 664),
    (LV5, 4, "area", 2),
    (LV5, 5, "min", None),
]


def rate_excluding_top(r_cand: np.ndarray, s_cand: np.ndarray,
                       r_lin: np.ndarray, s_lin: np.ndarray, k: int) -> dict:
    """候補の払戻が大きいレース上位k本を候補・線形の両方から同時に除いた回収率を返す
    (同じレース集合を両者から除くので対比較の公平性は保たれる)。"""
    n = len(r_cand)
    drop = np.argsort(-r_cand)[:k] if k > 0 else np.array([], dtype=int)
    keep = np.setdiff1d(np.arange(n), drop)
    c = r_cand[keep].sum() / s_cand[keep].sum() * 100.0
    l = r_lin[keep].sum() / s_lin[keep].sum() * 100.0
    return {"k_dropped": int(k), "n_races": int(len(keep)),
            "candidate_rate_pct": float(c), "linear_rate_pct": float(l),
            "diff_pt": float(c - l),
            "dropped_payouts": [float(x) for x in r_cand[drop]]}


def main():
    t0 = time.time()
    print("データロード中...")
    data = jra_dataset.load(rebuild=False)
    races_all, actual = data["races"], data["actual"]
    priors_all = JS.make_priors([r["df"] for r in races_all])
    lap33_lookup = L33.load_lap33_lookup()
    race_meta = L33.load_race_surface_distance()
    history_index = L33.build_history_index()
    cols = [JB.BET_TYPES.index(b) for b in OBJ_BETS]
    print(f"  レース数={len(races_all)} 目的券種={OBJ_BETS} ({time.time()-t0:.1f}s)")

    out = []
    for level_name in [LV8, LV5]:
        cat_map = LEVELS[level_name]
        cats = list(cat_map.keys())
        orders = unique_cyclic_orders(cats)
        records_all = RC.build_axis_matrix(
            races_all, priors_all, history_index, lap33_lookup, race_meta, category_map=cat_map
        )
        races, records = RC.filter_min_categories(
            races_all, records_all, min_categories=min(4, len(cat_map))
        )
        print(f"\n=== {level_name} 対象レース={len(races)} ({time.time()-t0:.1f}s) ===")
        lin_vals = scalar_scores(records, "linear")

        for (lv, box_n, method, oi) in TARGETS:
            if lv != level_name:
                continue
            ev = JE.Evaluator(races, actual, box_n=box_n)
            lin_picks = [_picks_from_vals(v, box_n) for v in lin_vals]
            lin_st, lin_rt = ev.settler.returns_for(lin_picks)
            if method == "area":
                vals = area_scores(records, orders[oi])
            else:
                vals = scalar_scores(records, method)
            picks = [_picks_from_vals(v, box_n) for v in vals]
            st, rt = ev.settler.returns_for(picks)
            r_c, s_c = rt[:, cols].sum(axis=1).astype(float), st[:, cols].sum(axis=1).astype(float)
            r_l, s_l = lin_rt[:, cols].sum(axis=1).astype(float), lin_st[:, cols].sum(axis=1).astype(float)
            rows = [rate_excluding_top(r_c, s_c, r_l, s_l, k) for k in (0, 1, 2, 3, 5)]
            rec = {
                "level": lv, "box_n": box_n, "method": method, "order_idx": oi,
                "order": orders[oi] if oi is not None else None,
                "n_points_per_race": len(list(itertools.combinations(range(box_n), 3))),
                "breakeven_pct": JE.breakeven_pct(box_n, bets=OBJ_BETS),
                "candidate_hit_races": int((r_c > 0).sum()),
                "linear_hit_races": int((r_l > 0).sum()),
                "n_races": int(len(r_c)),
                "jackknife": rows,
            }
            out.append(rec)
            print(f"\n[{lv} box{box_n} {method} order_idx={oi}] "
                  f"的中 候補={rec['candidate_hit_races']}/{rec['n_races']} "
                  f"線形={rec['linear_hit_races']}/{rec['n_races']} "
                  f"理論BE={rec['breakeven_pct']:.1f}%")
            for w in rows:
                print(f"   上位{w['k_dropped']}本除外: 候補={w['candidate_rate_pct']:7.2f}% "
                      f"線形={w['linear_rate_pct']:7.2f}% diff={w['diff_pt']:+7.2f}pt "
                      f"(除いた払戻={[int(x) for x in w['dropped_payouts']]})")

    (OUT_DIR / "jra_radar_sanrenpuku_pass_fragility_2026_08_30_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8", newline="\n"
    )
    print(f"\nwrote jra_radar_sanrenpuku_pass_fragility_2026_08_30_result.json ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
