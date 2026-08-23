# -*- coding: utf-8 -*-
"""馬連特化モデリング(2026-08-22、Step1アーカイブ拡張の馬連版)の本体スクリプト。7ゲート評価。

既存`jra_market_model_archive_2026_08_23.py`(単勝/複勝版)と同じ母集団・同じ勝率モデル
(u=beta0・log(q)+beta1・z_近走、2パラメータ)・同じロックホールドアウトを使う。
**勝率モデル自体は無改造**、変わるのは賭け目を「単勝/複勝1点」から「馬連(Harville式ペア
確率のp/q比、top1固定)」に置き換えた点のみ。

事前登録(実行前に確定):
  母集団      : jra_archive_dataset.build(model_start="2025-04-01", model_end="2026-02-28")
                除外 = 新馬|未勝利|障害(単勝/複勝版と完全同一)
  勝率モデル  : u = beta0・log(q) + beta1・z_近走(2パラメータ、MM.fit_2param、単勝/複勝版と
                完全同一の推定手続き)
  賭け目      : 馬連1点(Harville式p_pair/q_pair >= 1.10、レース内最良1組、100円固定)
  払戻cap     : 想定払戻(市場インプライド確率から導出、ex-ante)が20,000円以上のペアは
                そもそも賭けない(ユーザー指定)。選抜の時点で一貫して適用する。
  CV          : walk_forward_oof(月次リフィット、burn_in=6ヶ月)が主判定
  ロックホールドアウト: 2026-03-01〜2026-06-30(単勝/複勝版と同一期間)

Gate1(NLL改善)・Gate2(beta1層別安定性)は勝率モデル自体の性質であり馬連/単勝で変わらない
ため独立に再計算する(既存jra_market_model_archive_2026_08_23_result.jsonと数値一致するはず
=モデル無改造の裏付け・セルフチェック)。

2026-08-23 Opus 5サブエージェントのレビューを受けた修正: 初版は払戻capを実現後(realized)
基準で実装しており、(a)実運用不能、(b)Gate5の順列検定で帰無分布だけが非対称に切り詰められ
「p=0.000でPASS」という偽陽性を生む、という不具合があった(cap無しで同一picks・同一seedで
再実行するとp=0.455でFAILに反転することを確認)。本版はcapをex-ante基準
(MM.estimated_umaren_payout、市場インプライド確率由来の想定払戻)に修正し、Gate5では
cap適用版・cap無し版の両方を出力する。また市場ベンチマーク(人気1-2位の馬連)との差
(excess)を全ゲートで出力する(初版は計算済みなのにレポートしていなかった)。

実行方法: python scripts/jra_model/jra_umaren_archive_2026_08_23.py
出力: data/jra_pipeline/jra_umaren_archive_2026_08_23_result.json
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_archive_dataset as JAD  # noqa: E402
import jra_market_model as MM  # noqa: E402
import jra_signals as JS  # noqa: E402
import jra_singles_eval as SE  # noqa: E402
import jra_umaren_eval as UE  # noqa: E402

OUT_PATH = PROJECT_ROOT / "data" / "jra_pipeline" / "jra_umaren_archive_2026_08_23_result.json"

MODEL_START = "2025-04-01"
MODEL_END = "2026-02-28"
HOLDOUT_START = "2026-03-01"
HOLDOUT_END = "2026-06-30"
ODDS_COL = "odds_final"
NINKI_COL = "popularity"
MIN_VALID_Z = MM.DEFAULT_MIN_VALID_Z  # 4
PQ_THRESHOLD = MM.DEFAULT_PQ_THRESHOLD  # 1.10
MAX_PAYOUT = UE.DEFAULT_MAX_PAYOUT  # 20000円(ユーザー指定)
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
              "holdout_start": HOLDOUT_START, "holdout_end": HOLDOUT_END,
              "max_payout_cap": MAX_PAYOUT}

    print(f"=== 母集団構築: {MODEL_START} 〜 {MODEL_END} ===")
    data = JAD.build(model_start=MODEL_START, model_end=MODEL_END, verbose=True)
    races, actual = data["races"], data["actual"]
    result["n_races"] = len(races)
    result["n_dates"] = len(data["dates"])

    feats = build_feats(races, actual)
    ev_single = SE.Evaluator(races, actual, ninki_col=NINKI_COL)  # Gate1/2用(勝率モデルの性質)
    ev_umaren = UE.UmarenEvaluator(races, actual, ninki_col=NINKI_COL)  # Gate3-7用
    fit_fn = fit_fn_factory(feats)
    fit_fn_mkt = fit_fn_market_only_factory(feats)

    beta_full = MM.fit_2param(feats)
    result["beta_full_train_insample"] = beta_full.tolist()
    print(f"beta_full(in-sample, 参考値・採否には使わない)={beta_full}")

    # ============================================================ Gate1: NLL改善(勝率モデルの性質、単勝/複勝版と同一)
    print("\n=== Gate1: walk_forward NLL改善(勝率モデル、単勝/複勝版と同一のはず) ===")
    wf_nll_model = ev_single.walk_forward_oof_nll(fit_fn, feats, burn_in_months=6)
    wf_nll_mkt = ev_single.walk_forward_oof_nll(fit_fn_mkt, feats, burn_in_months=6)
    diff_nll = ev_single.block_bootstrap_diff_nll(wf_nll_model["nll_per_race"], wf_nll_mkt["nll_per_race"])
    # NLL改善を対数成長率(%)に変換し、馬連の控除率(22.5%)と桁で比較できるようにする
    # (Opus 5レビュー指摘: 0.0057 nats/レースという絶対値だけでは「控除を超えられるか」の
    # 判断材料にならない。exp(diff)-1が「市場のみモデルに対して何%良く賭けられるか」の目安)。
    nll_improvement_pct = (float(np.exp(diff_nll["mean"])) - 1) * 100
    result["gate1"] = {
        "mean_nll_model": wf_nll_model["mean_nll"], "mean_nll_market_only": wf_nll_mkt["mean_nll"],
        "diff_ci": diff_nll, "n_folds": len(wf_nll_model["chosen_params"]),
        "nll_improvement_pct_vs_takeout": {"value": nll_improvement_pct, "umaren_takeout_pct": 22.5},
        "pass": diff_nll["lo"] > 0,
    }
    print(f"  model_nll={wf_nll_model['mean_nll']:.5f} market_only_nll={wf_nll_mkt['mean_nll']:.5f} "
          f"diff_ci=[{diff_nll['lo']:.5f},{diff_nll['hi']:.5f}] -> "
          f"{'PASS' if result['gate1']['pass'] else 'FAIL'}")
    print(f"  NLL改善を成長率換算: 約{nll_improvement_pct:.2f}%(馬連の控除率22.5%と比較すると"
          f"2桁小さい水準)")

    # ============================================================ Gate2: beta1の層別安定性(単勝/複勝版と同一)
    print("\n=== Gate2: beta1の層別安定性(勝率モデル、単勝/複勝版と同一のはず) ===")
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

    # ============================================================ Gate3/4: walk_forward 馬連回収率(ex-ante cap有/無)
    # 2026-08-23修正(Opus 5レビュー): 払戻capを実現後(realized)から事前(ex-ante)基準に
    # 変更。MM.umaren_pq_picksのmax_payout引数で「想定払戻が上限を超えるペアはそもそも
    # 賭けない」を選抜の時点で適用する。picks自体が既にcap済みなので、決済・ブートストラップ
    # は事後フィルタ不要(旧UE.apply_payout_capは廃止)。
    print("\n=== Gate3/4: walk_forward OOF 馬連回収率(ex-ante cap適用/参考=cap無し) ===")
    wf = ev_umaren.walk_forward_oof(fit_fn, feats, burn_in_months=6, pq_threshold=PQ_THRESHOLD,
                                    max_payout=MAX_PAYOUT)
    wf_nocap = ev_umaren.walk_forward_oof(fit_fn, feats, burn_in_months=6, pq_threshold=PQ_THRESHOLD,
                                          max_payout=None)
    tested_idx = wf["tested_race_idx"]
    picks = wf["picks"]

    boot_capped = ev_umaren.block_bootstrap(picks)
    boot_uncapped = ev_umaren.block_bootstrap(wf_nocap["picks"])

    result["gate3"] = {"umaren_return_pct": wf["model"], "market_return_pct": wf["market"],
                       "excess_pt": wf["excess"], "umaren_boot": boot_capped,
                       "n_bet_races": wf["n_bet_races"],
                       "pass": boot_capped["lo"] > 100.0 and wf["n_bet_races"] >= 50}
    result["gate4"] = {"umaren_return_pct_nocap": wf_nocap["model"],
                       "market_return_pct_nocap": wf_nocap["market"],
                       "excess_pt_nocap": wf_nocap["excess"],
                       "umaren_boot_nocap": boot_uncapped, "n_bet_races_nocap": wf_nocap["n_bet_races"],
                       "pass": wf_nocap["model"] > 100.0}
    print(f"  馬連回収率(ex-ante cap適用)={wf['model']:.2f}% 市場(人気1-2)={wf['market']:.2f}% "
          f"市場差={wf['excess']:+.2f}pt CI=[{boot_capped['lo']:.2f},{boot_capped['hi']:.2f}] "
          f"n_bet_races={wf['n_bet_races']} -> {'PASS' if result['gate3']['pass'] else 'FAIL'}")
    print(f"  馬連回収率(cap無し・参考)={wf_nocap['model']:.2f}% 市場={wf_nocap['market']:.2f}% "
          f"市場差={wf_nocap['excess']:+.2f}pt CI=[{boot_uncapped['lo']:.2f},{boot_uncapped['hi']:.2f}] -> "
          f"{'PASS' if result['gate4']['pass'] else 'FAIL'}")

    # ============================================================ Gate5: オッズマッチ順列検定(ペア版、ex-ante cap対称適用)
    print("\n=== Gate5: オッズマッチ順列検定(馬連ペア版) ===")
    perm = UE.odds_matched_permutation_test(ev_umaren, races, picks, n_perm=2000, seed=77,
                                            odds_col=ODDS_COL, tol_log=0.15, max_payout=MAX_PAYOUT)
    perm_nocap = UE.odds_matched_permutation_test(ev_umaren, races, wf_nocap["picks"], n_perm=2000,
                                                  seed=77, odds_col=ODDS_COL, tol_log=0.15,
                                                  max_payout=None)
    result["gate5"] = {**perm, "pass": perm["p_value_ge_real"] < 0.05,
                       "nocap_reference": perm_nocap}
    print(f"  [cap適用/主判定] real_rate={perm['real_rate']:.2f}% sim_mean={perm['sim_mean']:.2f}% "
          f"p={perm['p_value_ge_real']:.4f} -> {'PASS' if result['gate5']['pass'] else 'FAIL'}")
    print(f"  [cap無し/参考]   real_rate={perm_nocap['real_rate']:.2f}% "
          f"sim_mean={perm_nocap['sim_mean']:.2f}% p={perm_nocap['p_value_ge_real']:.4f}")

    # ============================================================ Gate6: 閾値単調性(記述的、ex-ante cap適用)
    print("\n=== Gate6: p/q閾値グリッドの単調性(記述的診断、ex-ante cap適用) ===")
    grid_rates = {}
    for t in (1.00, 1.05, 1.10, 1.15, 1.20):
        wf_t = ev_umaren.walk_forward_oof(fit_fn, feats, burn_in_months=6, pq_threshold=t,
                                          max_payout=MAX_PAYOUT)
        grid_rates[t] = {"umaren_return_pct": wf_t["model"], "market_return_pct": wf_t["market"],
                         "n_bet_races": wf_t["n_bet_races"]}
        print(f"  pq>={t}: 馬連回収率={wf_t['model']:.2f}%  市場={wf_t['market']:.2f}%  "
              f"n_bet_races={wf_t['n_bet_races']}")
    vals = [grid_rates[t]["umaren_return_pct"] for t in sorted(grid_rates)]
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
        hev = UE.UmarenEvaluator(hraces, hactual, ninki_col=NINKI_COL)
        beta_final = MM.fit_2param(feats)  # 学習母集団全体でfit、ホールドアウトは一切参照しない
        h_picks = MM.umaren_pq_picks(beta_final, hfeats, pq_threshold=PQ_THRESHOLD,
                                     max_payout=MAX_PAYOUT)
        h_eval = hev.evaluate(h_picks)
        h_boot = hev.block_bootstrap(h_picks)
        result["gate7"] = {"beta_used": beta_final.tolist(), "umaren_return_pct": h_eval["model"],
                           "market_return_pct": h_eval["market"], "excess_pt": h_eval["excess"],
                           "umaren_boot": h_boot, "n_bet_races": h_eval["n_bet_races"],
                           "pass": h_eval["model"] > 100.0 and h_boot["lo"] > 90.0}
        print(f"  馬連回収率(ex-ante cap適用)={h_eval['model']:.2f}% "
              f"市場(人気1-2)={h_eval['market']:.2f}% 市場差={h_eval['excess']:+.2f}pt "
              f"CI=[{h_boot['lo']:.2f},{h_boot['hi']:.2f}] n_bet_races={h_eval['n_bet_races']} -> "
              f"{'PASS' if result['gate7']['pass'] else 'FAIL'}")
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
