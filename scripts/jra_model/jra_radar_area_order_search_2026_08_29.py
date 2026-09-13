# -*- coding: utf-8 -*-
"""面積(レーダー順序)の隣接構造を総当たり探索し、「良い順序」が実在するか
選択バイアス込みで検証するサイドカー(ユーザー依頼、2026-08-29)。

**重要な位置づけ**: jra_radar_area_gate_2026_08_29.py(主検定・確証的結論)とは別物。
あちらは「1本の事前指定した比較」を主検定としたが、本スクリプトは真逆の
「多数の順序から事後的に最良を選ぶ」という、まさに計画時に禁じた
garden-of-forking-pathsを意図的に実行し、それがどれだけ過大評価を生むかを
selection_optimism(jra_eval.py)と同じ発想(ブロックを2分割し、片側で選んだ
「最良」をもう片側で検証する)で定量化する。**最良順序をそのまま採用する
ための探索ではない**。

対象は面積のみ(box4)。他4集約方法(線形/幾何平均/最小値/上位3軸和)は
順序に依存しないため「隣接シグナルを変える」という操作自体が定義できない。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_backtest as JB
import jra_dataset
import jra_eval as JE
import jra_lap33_signals as L33
import jra_radar_categories as RC
import jra_signals as JS

OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
BOX_N = 4
N_REP = 200
SEED = 17


def unique_cyclic_orders(categories: list) -> list:
    """8要素の巡回順序を回転・反転の重複を除いて列挙する(先頭を固定し、残り7要素の
    順列5040通りを辺集合でグルーピングして代表を1つずつ取る。厳密に7!/2=2520通りになる)。"""
    anchor = categories[0]
    rest = categories[1:]
    seen = {}
    for perm in __import__("itertools").permutations(rest):
        order = [anchor] + list(perm)
        edges = RC._cyclic_edges(order)
        key = frozenset(edges)
        if key not in seen:
            seen[key] = order
    return list(seen.values())


def main():
    print("データロード中...")
    data = jra_dataset.load(rebuild=False)
    races_all, actual = data["races"], data["actual"]
    priors_all = JS.make_priors([r["df"] for r in races_all])
    lap33_lookup = L33.load_lap33_lookup()
    race_meta = L33.load_race_surface_distance()
    history_index = L33.build_history_index()

    records_all = RC.build_axis_matrix(races_all, priors_all, history_index, lap33_lookup, race_meta)
    races, records = RC.filter_min_categories(races_all, records_all)
    print(f"対象レース数: {len(races)}")

    orders = unique_cyclic_orders(RC.CATEGORIES)
    print(f"探索する順序数(回転・反転の重複除去後): {len(orders)}")

    ev = JE.Evaluator(races, actual, box_n=BOX_N)
    bet_cols = [JB.BET_TYPES.index(b) for b in JE.OBJ_BETS]

    n_orders = len(orders)
    n_races = len(races)
    all_st = np.empty((n_orders, n_races, len(bet_cols)), dtype=np.int64)
    all_rt = np.empty((n_orders, n_races, len(bet_cols)), dtype=np.int64)
    for i, order in enumerate(orders):
        picks = RC.radar_area_picks(records, order, BOX_N)
        st, rt = ev.settler.returns_for(picks)
        all_st[i] = st[:, bet_cols]
        all_rt[i] = rt[:, bet_cols]

    market_rate = JE.cost_weighted_rate(ev.mkt_stake, ev.mkt_ret, bets=JE.OBJ_BETS)

    # ============================================================ in-sample分布
    in_sample_rate = all_rt.sum(axis=(1, 2)) / all_st.sum(axis=(1, 2)) * 100
    in_sample_excess = in_sample_rate - market_rate
    best_idx = int(np.argmax(in_sample_excess))
    best_order = orders[best_idx]

    print("\n" + "=" * 72)
    print(f"in-sample分布({n_orders}順序、box{BOX_N}、市場超過pt)")
    print("=" * 72)
    print(f"平均={in_sample_excess.mean():+.2f}pt  標準偏差={in_sample_excess.std():.2f}pt  "
          f"最小={in_sample_excess.min():+.2f}pt  最大={in_sample_excess.max():+.2f}pt")
    print(f"最良順序(in-sample): excess={in_sample_excess[best_idx]:+.2f}pt")
    print("  " + " → ".join(best_order))
    diff_best = ev.block_bootstrap_diff(
        RC.radar_area_picks(records, best_order, BOX_N),
        RC.linear_picks(records, BOX_N),
    )
    print(f"  この最良順序 vs 線形平均、ペア差分95%CI=[{diff_best['lo']:+.2f},{diff_best['hi']:+.2f}]pt "
          "(← in-sampleでの見かけ上のCI。次の分割検証と比較すること)")

    # 参考: 検証レポートで報告した順序1がこの分布のどこに位置するか
    order1_key = frozenset(RC._cyclic_edges(RC.ORDER_1))
    order1_idx = next(i for i, o in enumerate(orders) if frozenset(RC._cyclic_edges(o)) == order1_key)
    pct = float((in_sample_excess < in_sample_excess[order1_idx]).mean() * 100)
    print(f"\n参考: 既報告の順序1(物語的順序)はexcess={in_sample_excess[order1_idx]:+.2f}pt、"
          f"{n_orders}順序中下位{pct:.1f}パーセンタイル")

    # ============================================================ 選択バイアス診断
    # (selection_optimism()と同じ発想: ブロックを半分に割り、片側で選んだ「最良順序」を
    # もう片側で検証する。n_rep回繰り返して安定させる)
    print("\n" + "=" * 72)
    print(f"選択バイアス診断(ブロック半分割、n_rep={N_REP}、selection_optimism()と同発想)")
    print("=" * 72)
    ids = list(ev.block_ids)
    by_block = {b: np.where(ev.blocks == b)[0] for b in ids}
    rng = np.random.default_rng(SEED)
    sel, unseen, unseen_mean = [], [], []
    for _ in range(N_REP):
        perm = rng.permutation(len(ids))
        a = np.concatenate([by_block[ids[i]] for i in perm[: len(ids) // 2]])
        b = np.concatenate([by_block[ids[i]] for i in perm[len(ids) // 2:]])
        va = all_rt[:, a, :].sum(axis=(1, 2)) / all_st[:, a, :].sum(axis=(1, 2)) * 100
        vb = all_rt[:, b, :].sum(axis=(1, 2)) / all_st[:, b, :].sum(axis=(1, 2)) * 100
        best = int(np.argmax(va))
        sel.append(va[best])
        unseen.append(vb[best])
        unseen_mean.append(vb.mean())
    sel, unseen, unseen_mean = map(np.array, (sel, unseen, unseen_mean))
    optimism_pt = float(sel.mean() - unseen.mean())
    true_edge_pt = float((unseen - unseen_mean).mean())
    true_edge_sd = float((unseen - unseen_mean).std())
    win_rate = float((unseen > unseen_mean).mean())

    print(f"選抜側(train半分で最良、その側の値): {sel.mean():.2f}%")
    print(f"未使用側(test半分での同じ順序の値): {unseen.mean():.2f}%")
    print(f"未使用側の全順序平均(何も選ばなかった場合の期待値): {unseen_mean.mean():.2f}%")
    print(f"選択の楽観バイアス(selected - unseen): {optimism_pt:+.2f}pt")
    print(f"「探索して選ぶ」ことの真の価値(unseen - unseen_mean): {true_edge_pt:+.2f}pt "
          f"(sd={true_edge_sd:.2f}, 未使用側で全順序平均を上回った割合={win_rate:.1%})")

    print("\n" + "=" * 72)
    print("結論")
    print("=" * 72)
    if true_edge_pt > 1.0 and win_rate > 0.6:
        print("探索して最良順序を選ぶ行為に、偶然では説明しづらい真の価値がある可能性がある。")
        print("ただし主検定(幾何平均vs線形平均)はすでに不採用のため、この結果だけでの本番反映はしない。")
    else:
        print("「最良順序」はtrain半分でしか良く見えておらず、test半分では平均的な順序と大差ない"
              "(true_edge_pt が小さい/win_rateが50%付近)。")
        print("これは2026-08-01のNAR300パターン探索撤回・JRA box4/3劣化の根本原因と同型の"
              "過学習パターンであり、in-sampleの最良順序をそのまま採用してはならないことを示す。")

    out = {
        "n_orders": n_orders, "n_races": n_races,
        "in_sample_mean_pt": float(in_sample_excess.mean()),
        "in_sample_std_pt": float(in_sample_excess.std()),
        "in_sample_min_pt": float(in_sample_excess.min()),
        "in_sample_max_pt": float(in_sample_excess.max()),
        "best_order": best_order, "best_order_in_sample_excess_pt": float(in_sample_excess[best_idx]),
        "best_order_vs_linear_ci_lo": diff_best["lo"], "best_order_vs_linear_ci_hi": diff_best["hi"],
        "order1_excess_pt": float(in_sample_excess[order1_idx]), "order1_percentile": pct,
        "selection_bias": {
            "selected_side_pct": float(sel.mean()), "unseen_side_pct": float(unseen.mean()),
            "unseen_all_mean_pct": float(unseen_mean.mean()),
            "optimism_pt": optimism_pt, "true_edge_pt": true_edge_pt,
            "true_edge_sd": true_edge_sd, "win_rate": win_rate,
        },
    }
    import json
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "jra_radar_area_order_search_2026_08_29_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nwrote jra_radar_area_order_search_2026_08_29_result.json")


if __name__ == "__main__":
    main()
