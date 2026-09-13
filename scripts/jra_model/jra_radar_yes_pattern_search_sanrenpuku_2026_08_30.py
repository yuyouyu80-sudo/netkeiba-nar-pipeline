# -*- coding: utf-8 -*-
"""3連複限定版の総当たり探索サイドカー(ユーザー明示依頼、2026-08-30)。

jra_radar_yes_pattern_search_2026_08_29.py と探索空間・手法を完全に同一に保ったまま、
目的券種(bets)だけを ["複勝", "ワイド"] から ["3連複"] に差し替えて再実行する。
元スクリプトは一切改造していない(本ファイルはサイドカーのコピー)。

**位置づけ(重要)**: これは採用候補を選ぶための探索ではない。
  (軸統合レベル 8/7/6/5) × (box_n 3/4/5) × (集約方法 幾何平均/最小値/上位3軸和/面積[全順序])
の全空間を総当たりし、名目上のPASS(lo>0)を隠さず全部列挙する。そのうえで、ブロック
半分割の分割検証(n_rep=200、seed固定)を必ず併走させ、見つかったPASSが「真に汎化する」のか
「探索アーティファクト」なのかを両論併記する。名目PASSをそのまま採用可能な発見として
報告してはならない。

3連複固有の注意:
  * BoxSettler が box_n に応じた全券種の的中判定を内部で持つため、決済ロジックは
    そのまま使う(自前で再実装しない)。点数は box3=C(3,3)=1、box4=C(4,3)=4、box5=C(5,3)=10。
  * box3 は「選んだ3頭がそのまま1-2-3着」でしか的中しないため標本が極端に薄くなる。
    的中レース数を各(レベル, box_n)でログに出して、標本の薄さを可視化する。
  * 3連複は複勝+ワイドより配当分散が桁違いに大きいので、ブロックブートストラップCIは
    広くなりやすい(=PASSしにくい)一方、極端な高配当1本がCIを歪めうる。診断値として
    「線形/最良候補の払戻上位1レースが総払戻に占める割合」もログに出す。

モデル計算側(jra_radar_categories.py / jra_signals.py / jra_eval.py / jra_backtest.py)は
一切改造しない。
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jra_backtest as JB  # noqa: E402
import jra_dataset  # noqa: E402
import jra_eval as JE  # noqa: E402
import jra_lap33_signals as L33  # noqa: E402
import jra_radar_categories as RC  # noqa: E402
import jra_signals as JS  # noqa: E402
from jra_radar_axis_merge_search_2026_08_29 import LEVELS  # noqa: E402

OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
STEM = "jra_radar_yes_pattern_search_sanrenpuku_2026_08_30"
OBJ_BETS = ["3連複"]     # ★ 今回の唯一の変更点(前回は JE.OBJ_BETS = 複勝+ワイド)
BOX_NS = [3, 4, 5]
N_BOOT = 2000
BOOT_SEED = 11          # jra_eval.block_bootstrap_diff の既定 seed と一致させる
SPLIT_N_REP = 200
SPLIT_SEED = 17         # jra_radar_area_order_search_2026_08_29.py と同じ
SCALAR_METHODS = ["geometric", "min", "top3sum"]


# --------------------------------------------------------------------- ヘルパ

def unique_cyclic_orders(categories: list) -> list:
    """巡回順序を回転・反転の重複を除いて列挙する((n-1)!/2 通り)。"""
    anchor = categories[0]
    rest = categories[1:]
    seen = {}
    for perm in itertools.permutations(rest):
        order = [anchor] + list(perm)
        key = frozenset(RC._cyclic_edges(order))
        if key not in seen:
            seen[key] = order
    return list(seen.values())


def _picks_from_vals(vals: np.ndarray, box_n: int) -> np.ndarray:
    """RC._picks_from_scores と同一のロジック(NaNは最下位、安定ソート)。"""
    v = np.where(np.isnan(vals), -1e18, vals)
    return np.argsort(-v, kind="stable")[: min(box_n, len(v))]


def area_scores(records: list, order: list) -> list:
    """RC.radar_area_picks と同じ面積スコアを box_n 非依存で1回だけ計算する。"""
    out = []
    for rec in records:
        axis = rec["axis"]
        present = [c for c in order if c in axis.columns]
        n = len(present)
        vals = axis[present].to_numpy(dtype=float)
        shifted = np.roll(vals, -1, axis=1)
        out.append(0.5 * np.sin(2 * np.pi / n) * (vals * shifted).sum(axis=1))
    return out


def scalar_scores(records: list, how: str) -> list:
    return [RC._score_from_axis(rec["axis"], how).to_numpy(dtype=float) for rec in records]


def block_codes_of(ev: JE.Evaluator) -> np.ndarray:
    pos = {b: j for j, b in enumerate(ev.block_ids)}
    return np.array([pos[b] for b in ev.blocks], dtype=np.int64)


def block_sums(codes: np.ndarray, n_blocks: int, st: np.ndarray, rt: np.ndarray,
               bet_cols: list) -> tuple:
    s = st[:, bet_cols].sum(axis=1).astype(np.float64)
    r = rt[:, bet_cols].sum(axis=1).astype(np.float64)
    return (np.bincount(codes, weights=s, minlength=n_blocks),
            np.bincount(codes, weights=r, minlength=n_blocks))


def bootstrap_count_matrix(n_blocks: int, n_boot: int, seed: int) -> np.ndarray:
    """jra_eval.block_bootstrap_diff と同一の乱数列でリサンプルし、各反復のブロック出現回数
    行列 (n_boot, n_blocks) を返す。"""
    rng = np.random.default_rng(seed)
    C = np.empty((n_boot, n_blocks), dtype=np.float64)
    for k in range(n_boot):
        chosen = rng.choice(n_blocks, size=n_blocks, replace=True)
        C[k] = np.bincount(chosen, minlength=n_blocks)
    return C


def ci_all_candidates(BS: np.ndarray, BR: np.ndarray, lin_s: np.ndarray, lin_r: np.ndarray,
                      C: np.ndarray) -> tuple:
    num = BR @ C.T                       # (n_cand, n_boot)
    den = BS @ C.T
    rate = np.where(den > 0, num / np.where(den == 0, 1, den) * 100.0, 0.0)
    lnum = lin_r @ C.T                   # (n_boot,)
    lden = lin_s @ C.T
    lrate = np.where(lden > 0, lnum / np.where(lden == 0, 1, lden) * 100.0, 0.0)
    diff = rate - lrate[None, :]
    return diff.mean(axis=1), np.percentile(diff, 2.5, axis=1), np.percentile(diff, 97.5, axis=1)


def pooled_rate(bs: np.ndarray, br: np.ndarray, mask: np.ndarray) -> float:
    s = bs[mask].sum()
    return float(br[mask].sum() / s * 100.0) if s else 0.0


def hit_stats(st: np.ndarray, rt: np.ndarray, bet_cols: list) -> dict:
    """3連複の的中レース数・的中率・払戻集中度(最大1レースの寄与率)を返す。"""
    r = rt[:, bet_cols].sum(axis=1).astype(float)
    n = len(r)
    tot = r.sum()
    return {
        "n_races": int(n),
        "hit_races": int((r > 0).sum()),
        "hit_rate_pct": float((r > 0).mean() * 100.0),
        "total_return": float(tot),
        "total_stake": float(st[:, bet_cols].sum()),
        "max_payout": float(r.max()) if n else 0.0,
        "top1_share_pct": float(r.max() / tot * 100.0) if tot > 0 else 0.0,
        "top3_share_pct": float(np.sort(r)[-3:].sum() / tot * 100.0) if tot > 0 and n >= 3 else 0.0,
    }


# --------------------------------------------------------------------- 本体

def main():
    t0 = time.time()
    print("データロード中...")
    data = jra_dataset.load(rebuild=False)
    races_all, actual = data["races"], data["actual"]
    priors_all = JS.make_priors([r["df"] for r in races_all])
    lap33_lookup = L33.load_lap33_lookup()
    race_meta = L33.load_race_surface_distance()
    history_index = L33.build_history_index()
    bet_cols = [JB.BET_TYPES.index(b) for b in OBJ_BETS]
    print(f"  レース数(除外前)={len(races_all)}  目的券種={OBJ_BETS}  "
          f"(前回は{JE.OBJ_BETS})  ({time.time()-t0:.1f}s)")
    for bn in BOX_NS:
        print(f"  box{bn}: 3連複点数=C({bn},3)={len(list(itertools.combinations(range(bn), 3)))} "
              f"理論ブレークイーブン={JE.breakeven_pct(bn, bets=OBJ_BETS):.2f}%")

    ev_cache = {}
    all_rows = []          # 全候補の一覧(名目CI込み)
    split_results = {}     # (level, box_n) -> 分割検証(候補プール全体スコープ)
    cand_split = {}        # 候補キー -> 候補固有の分割検証
    verify_notes = []
    payout_diag = {}       # (level, box_n) -> 3連複の的中・配当集中度診断

    for level_name, cat_map in LEVELS.items():
        expected = set().union(*cat_map.values()) - {"lap33_axis", "pace_fit"}
        expected |= {"mark_composite_score"}
        assert expected == set(JS.ALL_SIGNALS_V4), f"{level_name}: カテゴリ完全性チェック失敗"

        cats = list(cat_map.keys())
        orders = unique_cyclic_orders(cats)
        print(f"\n=== {level_name}  ({len(cats)}軸, 順序数={len(orders)}) ===")

        records_all = RC.build_axis_matrix(
            races_all, priors_all, history_index, lap33_lookup, race_meta, category_map=cat_map
        )
        races, records = RC.filter_min_categories(
            races_all, records_all, min_categories=min(4, len(cat_map))
        )
        print(f"  対象レース数={len(races)}  ({time.time()-t0:.1f}s)")

        sc = {m: scalar_scores(records, m) for m in ["linear"] + SCALAR_METHODS}
        area_sc = [area_scores(records, o) for o in orders]

        for box_n in BOX_NS:
            key_ev = (tuple(r["race_id"] for r in races), box_n)
            if key_ev not in ev_cache:
                ev_cache[key_ev] = JE.Evaluator(races, actual, box_n=box_n)
            ev = ev_cache[key_ev]
            n_blocks = len(ev.block_ids)
            codes = block_codes_of(ev)
            C = bootstrap_count_matrix(n_blocks, N_BOOT, BOOT_SEED)

            lin_picks = [_picks_from_vals(v, box_n) for v in sc["linear"]]
            lin_st, lin_rt = ev.settler.returns_for(lin_picks)
            lin_bs, lin_br = block_sums(codes, n_blocks, lin_st, lin_rt, bet_cols)
            lin_rate = float(lin_rt[:, bet_cols].sum() / lin_st[:, bet_cols].sum() * 100)
            mkt_rate = JE.cost_weighted_rate(ev.mkt_stake, ev.mkt_ret, bets=OBJ_BETS)

            labels, BS_rows, BR_rows = [], [], []
            picks_cache = {}
            for m in SCALAR_METHODS:
                picks = [_picks_from_vals(v, box_n) for v in sc[m]]
                st, rt = ev.settler.returns_for(picks)
                bs, br = block_sums(codes, n_blocks, st, rt, bet_cols)
                labels.append((m, None))
                BS_rows.append(bs)
                BR_rows.append(br)
                picks_cache[(m, None)] = picks
            for oi, o in enumerate(orders):
                picks = [_picks_from_vals(v, box_n) for v in area_sc[oi]]
                st, rt = ev.settler.returns_for(picks)
                bs, br = block_sums(codes, n_blocks, st, rt, bet_cols)
                labels.append(("area", oi))
                BS_rows.append(bs)
                BR_rows.append(br)
            BS = np.array(BS_rows)
            BR = np.array(BR_rows)

            mean_d, lo_d, hi_d = ci_all_candidates(BS, BR, lin_bs, lin_br, C)
            cand_rate = BR.sum(axis=1) / BS.sum(axis=1) * 100.0

            # --- ベクトル化ブートストラップが本家 block_bootstrap_diff と一致することを
            #     毎 (level, box_n) で「スカラー1件 + 面積1件」の2箇所検証する。
            check_positions = [0, int(np.where([lab[0] == "area" for lab in labels])[0][0])]
            for vi in check_positions:
                m_v, oi_v = labels[vi]
                pv = (picks_cache[(m_v, oi_v)] if (m_v, oi_v) in picks_cache
                      else [_picks_from_vals(v, box_n) for v in area_sc[oi_v]])
                ref = ev.block_bootstrap_diff(pv, lin_picks, bets=OBJ_BETS)
                ok = (abs(ref["lo"] - lo_d[vi]) < 1e-9 and abs(ref["hi"] - hi_d[vi]) < 1e-9
                      and abs(ref["mean"] - mean_d[vi]) < 1e-9)
                verify_notes.append({
                    "level": level_name, "box_n": box_n,
                    "checked": f"{m_v}" + (f"/order_idx={oi_v}" if oi_v is not None else ""),
                    "ref_mean": ref["mean"], "vec_mean": float(mean_d[vi]),
                    "ref_lo": ref["lo"], "vec_lo": float(lo_d[vi]),
                    "ref_hi": ref["hi"], "vec_hi": float(hi_d[vi]), "match": bool(ok),
                })
                assert ok, (f"ベクトル化ブートストラップが block_bootstrap_diff と不一致: "
                            f"{ref} vs mean={mean_d[vi]} lo={lo_d[vi]} hi={hi_d[vi]}")

            area_mask = np.array([lab[0] == "area" for lab in labels])
            area_idx = np.where(area_mask)[0]
            best_area_pos = int(area_idx[np.argmax(cand_rate[area_idx])])
            n_area_nominal_pass = int((lo_d[area_idx] > 0).sum())

            # --- 3連複固有の診断: 標本の薄さと配当集中度
            best_pos = int(np.argmax(cand_rate))
            m_b, oi_b = labels[best_pos]
            pb = (picks_cache[(m_b, oi_b)] if (m_b, oi_b) in picks_cache
                  else [_picks_from_vals(v, box_n) for v in area_sc[oi_b]])
            bst, brt = ev.settler.returns_for(pb)
            payout_diag[(level_name, box_n)] = {
                "linear": hit_stats(lin_st, lin_rt, bet_cols),
                "market": hit_stats(ev.mkt_stake, ev.mkt_ret, bet_cols),
                "best_in_sample": {"method": m_b, "order_idx": oi_b,
                                   **hit_stats(bst, brt, bet_cols)},
            }
            d = payout_diag[(level_name, box_n)]
            print(f"  box{box_n}: 線形={lin_rate:.2f}%(市場={mkt_rate:.2f}%) "
                  f"候補{len(labels)}件 名目PASS(全候補)={int((lo_d>0).sum())} "
                  f"[うち面積={n_area_nominal_pass}] "
                  f"({time.time()-t0:.1f}s)")
            print(f"        3連複的中: 線形={d['linear']['hit_races']}/{d['linear']['n_races']}"
                  f"({d['linear']['hit_rate_pct']:.1f}%) "
                  f"市場={d['market']['hit_races']}({d['market']['hit_rate_pct']:.1f}%) "
                  f"最良={d['best_in_sample']['hit_races']}"
                  f"({d['best_in_sample']['hit_rate_pct']:.1f}%)")
            print(f"        配当集中度: 線形 top1={d['linear']['top1_share_pct']:.1f}% "
                  f"top3={d['linear']['top3_share_pct']:.1f}% (最大払戻={d['linear']['max_payout']:.0f}円) "
                  f"/ 最良 top1={d['best_in_sample']['top1_share_pct']:.1f}% "
                  f"top3={d['best_in_sample']['top3_share_pct']:.1f}%")

            for i, (m, oi) in enumerate(labels):
                is_primary = (m != "area") or (i == best_area_pos)
                all_rows.append({
                    "level": level_name, "n_categories": len(cats), "box_n": box_n,
                    "method": m, "order_idx": oi,
                    "order": orders[oi] if oi is not None else None,
                    "n_races": len(races), "n_blocks": n_blocks,
                    "market_rate_pct": mkt_rate,
                    "linear_rate_pct": lin_rate, "candidate_rate_pct": float(cand_rate[i]),
                    "diff_pt": float(cand_rate[i] - lin_rate),
                    "boot_mean_pt": float(mean_d[i]), "ci_lo": float(lo_d[i]),
                    "ci_hi": float(hi_d[i]),
                    "ci_width_pt": float(hi_d[i] - lo_d[i]),
                    "gate_pass": bool(lo_d[i] > 0),
                    "is_primary_protocol": bool(is_primary),
                })

            n_pass_primary = sum(
                1 for i, (m, oi) in enumerate(labels)
                if lo_d[i] > 0 and ((m != "area") or i == best_area_pos)
            )
            print(f"        主プロトコルPASS={n_pass_primary}")

            # --- 分割検証(ブロック半分割)。この (level, box_n) の候補プール全体で実施。
            need_split = bool((lo_d > 0).any())
            if need_split:
                rng = np.random.default_rng(SPLIT_SEED)
                sel, unseen, unseen_mean = [], [], []
                cand_train = np.zeros((SPLIT_N_REP, len(labels)))
                cand_test = np.zeros((SPLIT_N_REP, len(labels)))
                chosen_hist = np.zeros(len(labels), dtype=int)
                for rep in range(SPLIT_N_REP):
                    perm = rng.permutation(n_blocks)
                    ma = np.zeros(n_blocks, dtype=bool)
                    ma[perm[: n_blocks // 2]] = True
                    mb = ~ma
                    la = pooled_rate(lin_bs, lin_br, ma)
                    lb = pooled_rate(lin_bs, lin_br, mb)
                    va = BR[:, ma].sum(axis=1) / np.maximum(BS[:, ma].sum(axis=1), 1) * 100.0 - la
                    vb = BR[:, mb].sum(axis=1) / np.maximum(BS[:, mb].sum(axis=1), 1) * 100.0 - lb
                    cand_train[rep] = va
                    cand_test[rep] = vb
                    best = int(np.argmax(va))
                    chosen_hist[best] += 1
                    sel.append(va[best])
                    unseen.append(vb[best])
                    unseen_mean.append(vb.mean())
                sel, unseen, unseen_mean = map(np.array, (sel, unseen, unseen_mean))
                split_results[(level_name, box_n)] = {
                    "pool_size": len(labels),
                    "selected_side_pt": float(sel.mean()),
                    "unseen_side_pt": float(unseen.mean()),
                    "unseen_all_mean_pt": float(unseen_mean.mean()),
                    "optimism_pt": float(sel.mean() - unseen.mean()),
                    "true_edge_pt": float((unseen - unseen_mean).mean()),
                    "true_edge_sd": float((unseen - unseen_mean).std()),
                    "win_rate": float((unseen > unseen_mean).mean()),
                }
                for i, (m, oi) in enumerate(labels):
                    if lo_d[i] <= 0:
                        continue
                    ck = (level_name, box_n, m, oi)
                    cand_split[ck] = {
                        "own_train_pt": float(cand_train[:, i].mean()),
                        "own_test_pt": float(cand_test[:, i].mean()),
                        "own_test_sd": float(cand_test[:, i].std()),
                        "own_optimism_pt": float((cand_train[:, i] - cand_test[:, i]).mean()),
                        "own_test_win_rate_vs_linear": float((cand_test[:, i] > 0).mean()),
                        "own_test_win_rate_vs_pool_mean":
                            float((cand_test[:, i] > cand_test.mean(axis=1)).mean()),
                        "selected_as_train_best_rate": float(chosen_hist[i] / SPLIT_N_REP),
                    }

    df = pd.DataFrame(all_rows)
    n_total = len(df)
    n_primary = int(df["is_primary_protocol"].sum())

    print("\n" + "=" * 90)
    print("総当たり結果サマリ(目的券種=3連複)")
    print("=" * 90)
    print(f"評価した候補の総数(面積の全順序を含む): {n_total}")
    print(f"うち主プロトコル対象(面積は各(レベル,box)のin-sample最良順序のみ): {n_primary}")
    print(f"名目PASS(全候補ベース): {int(df['gate_pass'].sum())}")
    print(f"名目PASS(主プロトコルベース): {int((df['gate_pass'] & df['is_primary_protocol']).sum())}")
    print(f"CI幅の中央値: {df['ci_width_pt'].median():.2f}pt "
          f"(最小={df['ci_width_pt'].min():.2f} 最大={df['ci_width_pt'].max():.2f})")

    print("\n--- 主プロトコル一覧(レベル×box_n×集約方法) ---")
    prim = df[df["is_primary_protocol"]].copy()
    for _, r in prim.iterrows():
        tag = "PASS" if r["gate_pass"] else "NO"
        print(f"{r['level']:32s} box{r['box_n']} {r['method']:>9s} "
              f"lin={r['linear_rate_pct']:7.2f}% cand={r['candidate_rate_pct']:7.2f}% "
              f"diff={r['diff_pt']:+7.2f}pt CI=[{r['ci_lo']:+7.2f},{r['ci_hi']:+7.2f}] {tag}")

    passes = df[df["gate_pass"]].sort_values("ci_lo", ascending=False)
    print("\n" + "=" * 90)
    print(f"gate_pass=True の候補一覧(全候補ベース、{len(passes)}件)")
    print("=" * 90)
    if passes.empty:
        near = df.sort_values("ci_lo", ascending=False).head(10)
        print("0件。最も惜しかった候補(CI下限が0に最も近い順、上位10件):")
        for _, r in near.iterrows():
            print(f"  {r['level']:32s} box{r['box_n']} {r['method']:>9s} "
                  f"order_idx={r['order_idx']} lin={r['linear_rate_pct']:7.2f}% "
                  f"cand={r['candidate_rate_pct']:7.2f}% diff={r['diff_pt']:+7.2f}pt "
                  f"CI=[{r['ci_lo']:+7.2f},{r['ci_hi']:+7.2f}]")
            if r["order"] is not None:
                print(f"      order: {' → '.join(r['order'])}")
    else:
        for _, r in passes.iterrows():
            print(f"  {r['level']:32s} box{r['box_n']} {r['method']:>9s} order_idx={r['order_idx']} "
                  f"lin={r['linear_rate_pct']:7.2f}% cand={r['candidate_rate_pct']:7.2f}% "
                  f"diff={r['diff_pt']:+7.2f}pt CI=[{r['ci_lo']:+7.2f},{r['ci_hi']:+7.2f}] "
                  f"primary={bool(r['is_primary_protocol'])}")
            if r["order"] is not None:
                print(f"      order: {' → '.join(r['order'])}")

    print("\n" + "=" * 90)
    print(f"分割検証(ブロック半分割、n_rep={SPLIT_N_REP}、seed={SPLIT_SEED})")
    print("=" * 90)
    for (lv, bn), s in split_results.items():
        print(f"[{lv} box{bn}] 候補プール={s['pool_size']}")
        print(f"  選抜側(train半分での超過pt)      : {s['selected_side_pt']:+.2f}pt")
        print(f"  未使用側(test半分での同候補)      : {s['unseen_side_pt']:+.2f}pt")
        print(f"  未使用側の全候補平均              : {s['unseen_all_mean_pt']:+.2f}pt")
        print(f"  optimism_pt (selected - unseen)  : {s['optimism_pt']:+.2f}pt")
        print(f"  true_edge_pt (unseen - poolmean) : {s['true_edge_pt']:+.2f}pt "
              f"(sd={s['true_edge_sd']:.2f}, win_rate={s['win_rate']:.1%})")
    if not split_results:
        print("(名目PASSが0件だったため分割検証は実施していない)")

    print("\n--- PASS候補ごとの固有分割検証 ---")
    for (lv, bn, m, oi), s in cand_split.items():
        print(f"[{lv} box{bn} {m} order_idx={oi}]")
        print(f"  own_train_pt={s['own_train_pt']:+.2f}  own_test_pt={s['own_test_pt']:+.2f}"
              f"(sd={s['own_test_sd']:.2f})  own_optimism_pt={s['own_optimism_pt']:+.2f}")
        print(f"  test半分で線形を上回った割合={s['own_test_win_rate_vs_linear']:.1%}  "
              f"候補プール平均を上回った割合={s['own_test_win_rate_vs_pool_mean']:.1%}  "
              f"train半分で最良に選ばれた割合={s['selected_as_train_best_rate']:.1%}")
    if not cand_split:
        print("(なし)")

    print("\n" + "=" * 90)
    print("3連複固有の診断(標本の薄さ・配当集中度)")
    print("=" * 90)
    for (lv, bn), d in payout_diag.items():
        print(f"[{lv} box{bn}] 点数/レース={len(list(itertools.combinations(range(bn),3)))}")
        for tag in ["linear", "market", "best_in_sample"]:
            h = d[tag]
            print(f"  {tag:14s} hit={h['hit_races']:3d}/{h['n_races']} ({h['hit_rate_pct']:5.1f}%) "
                  f"return={h['total_return']:.0f} stake={h['total_stake']:.0f} "
                  f"max_payout={h['max_payout']:.0f} top1_share={h['top1_share_pct']:5.1f}% "
                  f"top3_share={h['top3_share_pct']:5.1f}%")

    print("\n--- ベクトル化ブートストラップの数値検証 ---")
    for v in verify_notes:
        print(f"  [{v['level']} box{v['box_n']} {v['checked']}] match={v['match']} "
              f"ref=({v['ref_mean']:+.6f},{v['ref_lo']:+.6f},{v['ref_hi']:+.6f}) "
              f"vec=({v['vec_mean']:+.6f},{v['vec_lo']:+.6f},{v['vec_hi']:+.6f})")
    print(f"  検証箇所数={len(verify_notes)} 全一致={all(v['match'] for v in verify_notes)}")

    out = {
        "config": {
            "levels": list(LEVELS.keys()), "box_ns": BOX_NS,
            "scalar_methods": SCALAR_METHODS, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "split_n_rep": SPLIT_N_REP, "split_seed": SPLIT_SEED,
            "obj_bets": OBJ_BETS,
            "obj_bets_previous_run": JE.OBJ_BETS,
            "breakeven_pct": {f"box{bn}": JE.breakeven_pct(bn, bets=OBJ_BETS) for bn in BOX_NS},
        },
        "vectorized_bootstrap_verification": verify_notes,
        "n_candidates_total": n_total,
        "n_candidates_primary": n_primary,
        "n_nominal_pass_all": int(df["gate_pass"].sum()),
        "n_nominal_pass_primary": int((df["gate_pass"] & df["is_primary_protocol"]).sum()),
        "ci_width_median_pt": float(df["ci_width_pt"].median()),
        "ci_width_min_pt": float(df["ci_width_pt"].min()),
        "ci_width_max_pt": float(df["ci_width_pt"].max()),
        "primary_table": prim.to_dict(orient="records"),
        "pass_candidates": passes.to_dict(orient="records"),
        "near_miss_top10": df.sort_values("ci_lo", ascending=False).head(10).to_dict(orient="records"),
        "split_validation_pool": {f"{lv}|box{bn}": s for (lv, bn), s in split_results.items()},
        "split_validation_candidate": {
            f"{lv}|box{bn}|{m}|order_idx={oi}": s for (lv, bn, m, oi), s in cand_split.items()
        },
        "sanrenpuku_payout_diagnostics": {
            f"{lv}|box{bn}": d for (lv, bn), d in payout_diag.items()
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{STEM}_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8", newline="\n"
    )
    df.to_csv(OUT_DIR / f"{STEM}_all_candidates.csv", index=False, encoding="utf-8")
    print(f"\nwrote {STEM}_result.json / _all_candidates.csv ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
