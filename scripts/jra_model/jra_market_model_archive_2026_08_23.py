# -*- coding: utf-8 -*-
"""Step1再挑戦(選択肢B、2026-08-23)の本体スクリプト。7ゲート評価。

事前登録(実行前に確定):
  母集団      : jra_archive_dataset.build(model_start="2025-04-01", model_end="2026-02-28")
                除外 = 新馬|未勝利|障害
  履歴ソース  : 2024-08-24〜(jra_history.py、Phase0で欠落11日を補完済み)
  モデル      : u = beta0・log(q) + beta1・z_近走(2パラメータ、MM.fit_2param)
                q は odds_final 由来。z_近走 は timediff/form/margin/agari の等重み平均を
                レース内zscore(min_valid=4)
  賭け条件    : p/q >= 1.10 固定、レース内p/q最大の1頭のみ、単勝・複勝を各100円(1点固定)
                (EV閾値は使わない。閾値グリッドは記述的診断=G6のみに使う)
  主指標      : 複勝コスト加重回収率(副指標: 単勝)
  CV          : walk_forward_oof(月次リフィット、burn_in=6ヶ月)が主判定
  ロックホールドアウト: 2026-03-01〜2026-06-30(開発中は一切参照しない、G7で最後に1回だけ使用)
  汚染済み参照: 2026-07-01〜2026-08-16(211レース含む、既存jra_market_model_search_2026_08_22.py
                の結果を参考として併記するのみ、採否判定には使わない)

実行方法: python scripts/jra_model/jra_market_model_archive_2026_08_23.py
出力: data/jra_pipeline/jra_market_model_archive_2026_08_23_result.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_archive_dataset as JAD  # noqa: E402
import jra_market_model as MM  # noqa: E402
import jra_signals as JS  # noqa: E402
import jra_singles_eval as SE  # noqa: E402

OUT_PATH = PROJECT_ROOT / "data" / "jra_pipeline" / "jra_market_model_archive_2026_08_23_result.json"

MODEL_START = "2025-04-01"
MODEL_END = "2026-02-28"
HOLDOUT_START = "2026-03-01"
HOLDOUT_END = "2026-06-30"
ODDS_COL = "odds_final"
NINKI_COL = "popularity"
MIN_VALID_Z = MM.DEFAULT_MIN_VALID_Z  # 4
PQ_THRESHOLD = MM.DEFAULT_PQ_THRESHOLD  # 1.10
CENTRAL_4 = {"東京", "中山", "京都", "阪神"}


def build_feats(races, actual):
    priors_all = JS.make_priors([r["df"] for r in races])
    return MM.build_composite_features(races, actual, priors_all, JS.CLASS_ORDINAL,
                                       odds_col=ODDS_COL, min_valid_z=MIN_VALID_Z)


def fit_fn_factory(feats):
    def fit_fn(train_idx):
        return MM.fit_2param(feats, idx=train_idx)
    return fit_fn


def fit_fn_market_only_factory(feats):
    def fit_fn(train_idx):
        return MM.fit_beta0_only(feats, idx=train_idx)
    return fit_fn


def main():
    result = {"model_start": MODEL_START, "model_end": MODEL_END,
              "holdout_start": HOLDOUT_START, "holdout_end": HOLDOUT_END}

    print(f"=== 母集団構築: {MODEL_START} 〜 {MODEL_END} ===")
    data = JAD.build(model_start=MODEL_START, model_end=MODEL_END, verbose=True)
    races, actual = data["races"], data["actual"]
    result["n_races"] = len(races)
    result["n_dates"] = len(data["dates"])

    feats = build_feats(races, actual)
    ev = SE.Evaluator(races, actual, ninki_col=NINKI_COL)
    fit_fn = fit_fn_factory(feats)
    fit_fn_mkt = fit_fn_market_only_factory(feats)

    beta_full = MM.fit_2param(feats)
    result["beta_full_train_insample"] = beta_full.tolist()
    print(f"beta_full(in-sample, 参考値・採否には使わない)={beta_full}")

    # ============================================================ Gate1: NLL改善
    print("\n=== Gate1: walk_forward NLL改善 ===")
    wf_nll_model = ev.walk_forward_oof_nll(fit_fn, feats, burn_in_months=6)
    wf_nll_mkt = ev.walk_forward_oof_nll(fit_fn_mkt, feats, burn_in_months=6)
    diff_nll = ev.block_bootstrap_diff_nll(wf_nll_model["nll_per_race"], wf_nll_mkt["nll_per_race"])
    result["gate1"] = {
        "mean_nll_model": wf_nll_model["mean_nll"], "mean_nll_market_only": wf_nll_mkt["mean_nll"],
        "diff_ci": diff_nll, "n_folds": wf_nll_model["chosen_params"] and len(wf_nll_model["chosen_params"]),
        "pass": diff_nll["lo"] > 0,
    }
    print(f"  model_nll={wf_nll_model['mean_nll']:.5f} market_only_nll={wf_nll_mkt['mean_nll']:.5f} "
          f"diff_ci=[{diff_nll['lo']:.5f},{diff_nll['hi']:.5f}] -> "
          f"{'PASS' if result['gate1']['pass'] else 'FAIL'}")

    # ============================================================ Gate2: beta1の層別安定性
    print("\n=== Gate2: beta1の層別安定性 ===")
    dates_sorted = sorted({r["kaisai_date"] for r in races})
    n_chunks = 3
    chunk_bounds = np.array_split(dates_sorted, n_chunks)
    strata = {}
    for ci, chunk_dates in enumerate(chunk_bounds):
        chunk_dates = set(chunk_dates.tolist())
        idx = np.array([i for i, r in enumerate(races) if r["kaisai_date"] in chunk_dates])
        if len(idx) > 20:
            b = MM.fit_2param(feats, idx=idx)
            strata[f"temporal_chunk{ci+1}"] = {"n": int(len(idx)), "beta1": float(b[1])}

    central_idx = np.array([i for i, r in enumerate(races) if r["racecourse"] in CENTRAL_4])
    local_idx = np.array([i for i, r in enumerate(races) if r["racecourse"] not in CENTRAL_4])
    for name, idx in (("central4", central_idx), ("local6", local_idx)):
        if len(idx) > 20:
            b = MM.fit_2param(feats, idx=idx)
            strata[name] = {"n": int(len(idx)), "beta1": float(b[1])}

    beta1_vals = [v["beta1"] for v in strata.values()]
    signs = set(np.sign(v) for v in beta1_vals if v != 0)
    abs_vals = [abs(v) for v in beta1_vals if abs(v) > 1e-9]
    ratio = (max(abs_vals) / min(abs_vals)) if len(abs_vals) >= 2 else None
    result["gate2"] = {"strata": strata, "same_sign": len(signs) <= 1,
                       "max_min_abs_ratio": ratio,
                       "pass": (len(signs) <= 1) and (ratio is not None and ratio <= 3.0)}
    print(f"  strata={strata}")
    print(f"  same_sign={len(signs)<=1} ratio={ratio} -> {'PASS' if result['gate2']['pass'] else 'FAIL'}")

    # ============================================================ Gate3/4: walk_forward回収率
    print("\n=== Gate3/4: walk_forward OOF 複勝/単勝回収率 ===")
    wf = ev.walk_forward_oof(fit_fn, feats, burn_in_months=6, pq_threshold=PQ_THRESHOLD,
                             odds_cap=float("inf"))
    tested_idx = wf["tested_race_idx"]
    place_boot = ev.block_bootstrap(wf["picks"], bet="複勝", block_subset=None)
    win_boot = ev.block_bootstrap(wf["picks"], bet="単勝", block_subset=None)
    place_rate = SE.cost_weighted_rate(*ev.settler.returns_for(wf["picks"]), bet="複勝", idx=tested_idx)
    win_rate_pt = SE.cost_weighted_rate(*ev.settler.returns_for(wf["picks"]), bet="単勝", idx=tested_idx)
    result["gate3"] = {"place_return_pct": place_rate, "place_boot": place_boot,
                       "n_bet_races": wf["n_bet_races"],
                       "pass": place_boot["lo"] > 100.0 and wf["n_bet_races"] >= 50}
    result["gate4"] = {"win_return_pct": win_rate_pt, "win_boot": win_boot,
                       "pass": win_rate_pt > 100.0}
    print(f"  複勝回収率={place_rate:.2f}% CI=[{place_boot['lo']:.2f},{place_boot['hi']:.2f}] "
          f"n_bet_races={wf['n_bet_races']} -> {'PASS' if result['gate3']['pass'] else 'FAIL'}")
    print(f"  単勝回収率={win_rate_pt:.2f}% CI=[{win_boot['lo']:.2f},{win_boot['hi']:.2f}] "
          f"-> {'PASS' if result['gate4']['pass'] else 'FAIL'}")

    # ============================================================ Gate5: オッズマッチ順列検定
    print("\n=== Gate5: オッズマッチ順列検定(複勝) ===")
    perm = SE.odds_matched_permutation_test(ev, races, wf["picks"], bet="複勝", n_perm=2000,
                                            seed=77, odds_col=ODDS_COL, tol_log=0.15)
    result["gate5"] = {**perm, "pass": perm["p_value_ge_real"] < 0.05}
    print(f"  real_rate={perm['real_rate']:.2f}% sim_mean={perm['sim_mean']:.2f}% "
          f"p={perm['p_value_ge_real']:.4f} -> {'PASS' if result['gate5']['pass'] else 'FAIL'}")

    # ============================================================ Gate6: 閾値単調性(記述的)
    print("\n=== Gate6: p/q閾値グリッドの単調性(記述的診断) ===")
    grid_rates = {}
    for t in (1.00, 1.05, 1.10, 1.15, 1.20):
        wf_t = ev.walk_forward_oof(fit_fn, feats, burn_in_months=6, pq_threshold=t, odds_cap=float("inf"))
        r = SE.cost_weighted_rate(*ev.settler.returns_for(wf_t["picks"]), bet="複勝",
                                  idx=wf_t["tested_race_idx"])
        grid_rates[t] = {"place_return_pct": r, "n_bet_races": wf_t["n_bet_races"]}
        print(f"  pq>={t}: 複勝回収率={r:.2f}%  n_bet_races={wf_t['n_bet_races']}")
    vals = [grid_rates[t]["place_return_pct"] for t in sorted(grid_rates)]
    non_decreasing = all(vals[i] <= vals[i+1] + 5.0 for i in range(len(vals) - 1))  # 5pt許容
    result["gate6"] = {"grid": grid_rates, "monotonic_nondecreasing_approx": non_decreasing,
                       "pass": non_decreasing}
    print(f"  単調非減少(5pt許容)={non_decreasing} -> {'PASS' if result['gate6']['pass'] else 'FAIL(記述的)'}")

    # ============================================================ Gate7: ロックホールドアウト
    print(f"\n=== Gate7: ロックホールドアウト {HOLDOUT_START}〜{HOLDOUT_END}(最終確認・1回のみ) ===")
    hdata = JAD.build(model_start=HOLDOUT_START, model_end=HOLDOUT_END, verbose=True)
    hraces, hactual = hdata["races"], hdata["actual"]
    result["n_holdout_races"] = len(hraces)
    if hraces:
        hfeats = build_feats(hraces, hactual)
        hev = SE.Evaluator(hraces, hactual, ninki_col=NINKI_COL)
        beta_final = MM.fit_2param(feats)  # 学習母集団(2025-04〜2026-02)全体でfit、ホールドアウトは一切参照しない
        h_picks = MM.pq_picks(beta_final, hfeats, pq_threshold=PQ_THRESHOLD, odds_cap=float("inf"))
        h_place_rate = SE.cost_weighted_rate(*hev.settler.returns_for(h_picks), bet="複勝")
        h_boot = hev.block_bootstrap(h_picks, bet="複勝")
        n_bet_h = int(sum(1 for p in h_picks if p is not None and len(p) > 0))
        result["gate7"] = {"beta_used": beta_final.tolist(), "place_return_pct": h_place_rate,
                           "place_boot": h_boot, "n_bet_races": n_bet_h,
                           "pass": h_place_rate > 100.0 and h_boot["lo"] > 90.0}
        print(f"  複勝回収率={h_place_rate:.2f}% CI=[{h_boot['lo']:.2f},{h_boot['hi']:.2f}] "
              f"n_bet_races={n_bet_h} -> {'PASS' if result['gate7']['pass'] else 'FAIL'}")
    else:
        result["gate7"] = {"pass": False, "note": "ホールドアウト母集団が空"}
        print("  ホールドアウト母集団が空のためFAIL扱い")

    # ============================================================ 総合判定
    gate_keys = [f"gate{i}" for i in range(1, 8)]
    n_pass = sum(1 for k in gate_keys if result[k]["pass"])
    result["n_gates_passed"] = n_pass
    result["n_gates_total"] = len(gate_keys)
    result["decision"] = ("採用候補(7ゲート全通過)" if n_pass == len(gate_keys)
                          else f"不採用({len(gate_keys)}ゲート中{n_pass}個のみ通過)")
    print(f"\n=== 総合判定: {result['decision']} ===")
    for k in gate_keys:
        print(f"  {k}: {'PASS' if result[k]['pass'] else 'FAIL'}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
