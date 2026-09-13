# -*- coding: utf-8 -*-
"""「gate_pass=True(95%CI下限>0)になる組み合わせを見つける」ための総当たり探索サイドカー
(ユーザー明示依頼、2026-08-29)。

**位置づけ(重要)**: これは採用候補を選ぶための探索ではない。ユーザー依頼は
「判定がYESになるパターンを見つけてほしい」であり、本スクリプトは
  (軸統合レベル 8/7/6/5) × (box_n 3/4/5) × (集約方法 幾何平均/最小値/上位3軸和/面積[全順序])
の全空間を総当たりし、名目上のPASS(lo>0)を隠さず全部列挙する。そのうえで、
jra_radar_area_order_search_2026_08_29.py と同じ発想の分割検証(ブロック半分割 n_rep=200、
seed固定)を必ず併走させ、見つかったPASSが「真に汎化する」のか「探索アーティファクト」なのかを
両論併記する。名目PASSをそのまま採用可能な発見として報告してはならない。

モデル計算側(jra_radar_categories.py / jra_signals.py / jra_eval.py / jra_backtest.py)は
一切改造しない。

計算上の工夫:
  * ブートストラップはブロック単位・比推定量なので、事前に「候補×ブロック」の賭金和・払戻和を
    作れば、jra_eval.Evaluator.block_bootstrap_diff と数値的に完全一致する結果を行列積で
    一括計算できる(同じ seed=11 / rng.choice(len(ids), size=len(ids), replace=True) を使い、
    リサンプルをブロックの出現回数ベクトルに変換して掛けるだけ)。これにより2520順序すべてに
    ついて正式なCIを出せる(等価性は run 時に ev.block_bootstrap_diff と突き合わせて検証する)。
  * 面積スコアは box_n に依存しないので、(レベル, 順序)ごとに1回だけ計算して3つのbox_nで使い回す。
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
BOX_NS = [3, 4, 5]
N_BOOT = 2000
BOOT_SEED = 11          # jra_eval.block_bootstrap_diff の既定 seed と一致させる
SPLIT_N_REP = 200
SPLIT_SEED = 17         # jra_radar_area_order_search_2026_08_29.py と同じ
SCALAR_METHODS = ["geometric", "min", "top3sum"]


# --------------------------------------------------------------------- ヘルパ

def unique_cyclic_orders(categories: list) -> list:
    """巡回順序を回転・反転の重複を除いて列挙する((n-1)!/2 通り)。
    jra_radar_area_order_search_2026_08_29.py の同名関数と同一実装(汎用)。"""
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
    """各レースが ev.block_ids の何番目のブロックに属するかのコード配列を返す。"""
    pos = {b: j for j, b in enumerate(ev.block_ids)}
    return np.array([pos[b] for b in ev.blocks], dtype=np.int64)


def block_sums(codes: np.ndarray, n_blocks: int, st: np.ndarray, rt: np.ndarray,
               bet_cols: list) -> tuple:
    """(n_races, n_bets) の賭金・払戻を、目的券種だけ合算してブロック単位に畳む。
    戻り値: (block_stake[n_blocks], block_return[n_blocks]) — ev.block_ids の順。"""
    s = st[:, bet_cols].sum(axis=1).astype(np.float64)
    r = rt[:, bet_cols].sum(axis=1).astype(np.float64)
    return (np.bincount(codes, weights=s, minlength=n_blocks),
            np.bincount(codes, weights=r, minlength=n_blocks))


def bootstrap_count_matrix(n_blocks: int, n_boot: int, seed: int) -> np.ndarray:
    """jra_eval.block_bootstrap_diff と同一の乱数列でリサンプルし、各反復のブロック出現回数
    行列 (n_boot, n_blocks) を返す。ブロック内のレース集合はまとめて再標本化されるため、
    「回数×ブロック和」は元実装の np.concatenate(idx) 後の合計と厳密に一致する。"""
    rng = np.random.default_rng(seed)
    C = np.empty((n_boot, n_blocks), dtype=np.float64)
    for k in range(n_boot):
        chosen = rng.choice(n_blocks, size=n_blocks, replace=True)
        C[k] = np.bincount(chosen, minlength=n_blocks)
    return C


def ci_all_candidates(BS: np.ndarray, BR: np.ndarray, lin_s: np.ndarray, lin_r: np.ndarray,
                      C: np.ndarray) -> tuple:
    """全候補について「候補 - 線形平均」のブロックブートストラップ差分の (mean, lo, hi) を返す。
    BS/BR: (n_cand, n_blocks)、lin_s/lin_r: (n_blocks,)、C: (n_boot, n_blocks)。"""
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
    bet_cols = [JB.BET_TYPES.index(b) for b in JE.OBJ_BETS]
    print(f"  レース数(除外前)={len(races_all)}  目的券種={JE.OBJ_BETS}  ({time.time()-t0:.1f}s)")

    ev_cache = {}
    all_rows = []          # 全候補の一覧(名目CI込み)
    split_results = {}     # (level, box_n) -> 分割検証(候補プール全体スコープ)
    cand_split = {}        # 候補キー -> 候補固有の分割検証
    verify_notes = []

    for level_name, cat_map in LEVELS.items():
        # --- カテゴリ完全性チェック(merge searchと同じ)
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

        # --- スコアは box_n 非依存なので先に全部作る
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
            mkt_rate = JE.cost_weighted_rate(ev.mkt_stake, ev.mkt_ret, bets=JE.OBJ_BETS)

            labels, BS_rows, BR_rows = [], [], []
            for m in SCALAR_METHODS:
                picks = [_picks_from_vals(v, box_n) for v in sc[m]]
                st, rt = ev.settler.returns_for(picks)
                bs, br = block_sums(codes, n_blocks, st, rt, bet_cols)
                labels.append((m, None))
                BS_rows.append(bs)
                BR_rows.append(br)
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

            # --- ベクトル化ブートストラップが本家と一致することを毎回1件だけ検証する
            vi = 0
            ref = ev.block_bootstrap_diff(
                [_picks_from_vals(v, box_n) for v in sc[SCALAR_METHODS[vi]]], lin_picks
            )
            ok = (abs(ref["lo"] - lo_d[vi]) < 1e-9 and abs(ref["hi"] - hi_d[vi]) < 1e-9)
            verify_notes.append({
                "level": level_name, "box_n": box_n, "checked": SCALAR_METHODS[vi],
                "ref_lo": ref["lo"], "vec_lo": float(lo_d[vi]),
                "ref_hi": ref["hi"], "vec_hi": float(hi_d[vi]), "match": bool(ok),
            })
            assert ok, f"ベクトル化ブートストラップが block_bootstrap_diff と不一致: {ref} vs {lo_d[vi]},{hi_d[vi]}"

            # --- 面積は「in-sample最良の順序」を主プロトコルの代表とする
            area_mask = np.array([lab[0] == "area" for lab in labels])
            area_idx = np.where(area_mask)[0]
            best_area_pos = int(area_idx[np.argmax(cand_rate[area_idx])])
            n_area_nominal_pass = int((lo_d[area_idx] > 0).sum())

            for i, (m, oi) in enumerate(labels):
                is_primary = (m != "area") or (i == best_area_pos)
                row = {
                    "level": level_name, "n_categories": len(cats), "box_n": box_n,
                    "method": m, "order_idx": oi,
                    "order": orders[oi] if oi is not None else None,
                    "n_races": len(races), "n_blocks": n_blocks,
                    "market_rate_pct": mkt_rate,
                    "linear_rate_pct": lin_rate, "candidate_rate_pct": float(cand_rate[i]),
                    "diff_pt": float(cand_rate[i] - lin_rate),
                    "boot_mean_pt": float(mean_d[i]), "ci_lo": float(lo_d[i]), "ci_hi": float(hi_d[i]),
                    "gate_pass": bool(lo_d[i] > 0),
                    "is_primary_protocol": bool(is_primary),
                }
                all_rows.append(row)

            n_pass_primary = sum(
                1 for i, (m, oi) in enumerate(labels)
                if lo_d[i] > 0 and ((m != "area") or i == best_area_pos)
            )
            print(f"  box{box_n}: 線形={lin_rate:.2f}%(市場={mkt_rate:.2f}%) "
                  f"候補{len(labels)}件 名目PASS(全候補)={int((lo_d>0).sum())} "
                  f"[うち面積={n_area_nominal_pass}] 主プロトコルPASS={n_pass_primary} "
                  f"({time.time()-t0:.1f}s)")

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
    print("総当たり結果サマリ")
    print("=" * 90)
    print(f"評価した候補の総数(面積の全順序を含む): {n_total}")
    print(f"うち主プロトコル対象(面積は各(レベル,box)のin-sample最良順序のみ): {n_primary}")
    print(f"名目PASS(全候補ベース): {int(df['gate_pass'].sum())}")
    print(f"名目PASS(主プロトコルベース): {int((df['gate_pass'] & df['is_primary_protocol']).sum())}")

    print("\n--- 主プロトコル一覧(レベル×box_n×集約方法) ---")
    prim = df[df["is_primary_protocol"]].copy()
    for _, r in prim.iterrows():
        tag = "PASS" if r["gate_pass"] else "NO"
        print(f"{r['level']:32s} box{r['box_n']} {r['method']:>9s} "
              f"lin={r['linear_rate_pct']:6.2f}% cand={r['candidate_rate_pct']:6.2f}% "
              f"diff={r['diff_pt']:+6.2f}pt CI=[{r['ci_lo']:+6.2f},{r['ci_hi']:+6.2f}] {tag}")

    passes = df[df["gate_pass"]].sort_values("ci_lo", ascending=False)
    print("\n" + "=" * 90)
    print(f"gate_pass=True の候補一覧(全候補ベース、{len(passes)}件)")
    print("=" * 90)
    if passes.empty:
        near = df.sort_values("ci_lo", ascending=False).head(5)
        print("0件。最も惜しかった候補(CI下限が0に最も近い順、上位5件):")
        for _, r in near.iterrows():
            print(f"  {r['level']:32s} box{r['box_n']} {r['method']:>9s} "
                  f"order_idx={r['order_idx']} diff={r['diff_pt']:+6.2f}pt "
                  f"CI=[{r['ci_lo']:+6.2f},{r['ci_hi']:+6.2f}]")
    else:
        for _, r in passes.iterrows():
            print(f"  {r['level']:32s} box{r['box_n']} {r['method']:>9s} order_idx={r['order_idx']} "
                  f"lin={r['linear_rate_pct']:6.2f}% cand={r['candidate_rate_pct']:6.2f}% "
                  f"diff={r['diff_pt']:+6.2f}pt CI=[{r['ci_lo']:+6.2f},{r['ci_hi']:+6.2f}] "
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

    out = {
        "config": {
            "levels": list(LEVELS.keys()), "box_ns": BOX_NS,
            "scalar_methods": SCALAR_METHODS, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "split_n_rep": SPLIT_N_REP, "split_seed": SPLIT_SEED,
            "obj_bets": JE.OBJ_BETS,
        },
        "vectorized_bootstrap_verification": verify_notes,
        "n_candidates_total": n_total,
        "n_candidates_primary": n_primary,
        "n_nominal_pass_all": int(df["gate_pass"].sum()),
        "n_nominal_pass_primary": int((df["gate_pass"] & df["is_primary_protocol"]).sum()),
        "primary_table": prim.to_dict(orient="records"),
        "pass_candidates": passes.to_dict(orient="records"),
        "near_miss_top5": df.sort_values("ci_lo", ascending=False).head(5).to_dict(orient="records"),
        "split_validation_pool": {f"{lv}|box{bn}": s for (lv, bn), s in split_results.items()},
        "split_validation_candidate": {
            f"{lv}|box{bn}|{m}|order_idx={oi}": s for (lv, bn, m, oi), s in cand_split.items()
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "jra_radar_yes_pattern_search_2026_08_29_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8", newline="\n"
    )
    df.to_csv(OUT_DIR / "jra_radar_yes_pattern_search_2026_08_29_all_candidates.csv",
              index=False, encoding="utf-8")
    print(f"\nwrote jra_radar_yes_pattern_search_2026_08_29_result.json / _all_candidates.csv "
          f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
