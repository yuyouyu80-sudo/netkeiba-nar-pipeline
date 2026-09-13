# -*- coding: utf-8 -*-
"""凡走予測(flop) FLOP_SIGNALS_V2ショートリストのDirichlet多パターン探索(2026-09-11)。

計画: `C:\\Users\\yuyou\\.claude\\plans\\valiant-cuddling-aho.md`
(「JRA凡走シグナル(flop) 全候補(候補200由来 約40本)の網羅探索」)

`jra_flop_phase0_v2_2026_09_11.py`が出力したショートリスト(`jra_flop_v2_shortlist_2026_09_11.json`)
を対象に、alphaごとに独立したDirichlet探索(box3テンプレートと同じ2000パターン) + Nested OOF
(`group_kfold_oof_generic`, n_folds=8) + 選択バイアス補正(`flop_selection_optimism`)を行う。

判定の主軸は「Nested OOF + true_edge_pt/true_edge_sd>=2.0」(held-outではない、検出力上の理由。
計画「検出力(レビュー指摘C3への対応)」参照)。held-outは探索前に予約した直近日付15-20%を使う
参考確認として、主ゲートをPASSしたalphaがあった場合にのみ算出する。

`jra_eval.selection_optimism`/box3テンプレートの`fit_fn`はadjusted_score(alpha調整済み)には
そのまま使えないため、`jra_flop_eval.flop_selection_optimism`/`make_flop_predict_fn`
(いずれも本計画で新規実装)を使う(シニアエンジニアレビュー指摘C1/C2)。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_candidate_factors_v1 as CF1
import jra_candidate_factors_v2 as CF2
import jra_candidate_factors_v3 as CF3
import jra_dataset
import jra_eval as JE
import jra_flop_eval as JFE
import jra_flop_signals as JFS
import jra_history as JH
import jra_lap33_signals as L33
import jra_model_scoring as MS
import jra_signals as JS

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "jra_pipeline"
OUT_JSON = DATA_DIR / "jra_flop_search_v2_2026_09_11_result.json"
OUT_TXT = DATA_DIR / "jra_flop_search_v2_2026_09_11_report.txt"
SHORTLIST_JSON = DATA_DIR / "jra_flop_v2_shortlist_2026_09_11.json"

ALPHAS = [0.02, 0.05, 0.1, 0.2, 0.3]
N_PATTERNS = 2000
BOX_NS = [4, 5, 3]  # box_n=4を主指標とする(事前登録、計画のtie-break規則)
BOX_TO_WEIGHT_KEY = {5: "normal", 4: "normal_box4", 3: "normal_box3"}
HOLDOUT_FRACTION = 0.18  # 15-20%の範囲内、探索開始前に固定する
DECISION_GATE_RATIO = 2.0  # true_edge_pt/true_edge_sd の主ゲート閾値

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


# ============================================================ ショートリスト読み込み
shortlist = json.loads(SHORTLIST_JSON.read_text(encoding="utf-8"))["shortlist"]
log(f"V2ショートリスト: {len(shortlist)}候補")
log(f"  {shortlist}")

# ============================================================ データロード・holdout予約
data = jra_dataset.load(rebuild=True)
races_all, actual = data["races"], data["actual"]
dates_sorted = sorted(data["dates"])
n_holdout_dates = max(1, round(len(dates_sorted) * HOLDOUT_FRACTION))
holdout_dates = set(dates_sorted[-n_holdout_dates:])
search_dates = set(dates_sorted) - holdout_dates
log(f"\n全{len(dates_sorted)}日中、直近{len(holdout_dates)}日をheld-outとして予約(探索には使わない): "
    f"{sorted(holdout_dates)}")
log(f"探索(search)に使う日付: {len(search_dates)}日")

idx_search = [i for i, r in enumerate(races_all) if r["kaisai_date"] in search_dates]
races_search = [races_all[i] for i in idx_search]
log(f"探索母集団: {len(races_search)}レース / 全{len(races_all)}レース")

priors_search = JS.make_priors([r["df"] for r in races_search])
priors_all = JS.make_priors([r["df"] for r in races_all])

log("\ncf1/cf2/cf3計算用の参照データをロード中...")
results_all = JH.load_results()
race_meta = L33.load_race_surface_distance()
hist_idx = CF1.build_history_index(results_all)
cf3_ctx = CF3.Context(results_all)


def _compute_cf(races):
    cf1_l, cf2_l, cf3_l = [], [], []
    for r in races:
        df = r["df"]
        meta = race_meta.get(r["race_id"])
        racecourse, surface, distance_m = meta if meta else (r["racecourse"], None, None)
        cf1_l.append(CF1.compute_for_race(df, racecourse, distance_m, surface, r["kaisai_date"], hist_idx))
        cf2_l.append(CF2.compute_for_race(df, r["race_name"], r["kaisai_date"], hist_idx))
        cf3_l.append(CF3.compute_for_race(df, r["race_id"], r["kaisai_date"], racecourse, surface,
                                          cf3_ctx, distance_m=distance_m))
    return cf1_l, cf2_l, cf3_l


cf1_all, cf2_all, cf3_all = _compute_cf(races_all)
mats_flop_all = JFS.flop_signal_matrices_v2(cf1_all, cf2_all, cf3_all, shortlist)
mats_flop_search = [mats_flop_all[i] for i in idx_search]

NAMES_SCORE = JS.LEGACY_SIGNALS
mats_score_all = JS.signal_matrices(races_all, priors_all, NAMES_SCORE, JS.CLASS_ORDINAL)
mats_score_search = [mats_score_all[i] for i in idx_search]
w_score_prod = {}
for rt in ("normal", "normal_box4", "normal_box3"):
    w = MS.load_weights(rt)
    w_score_prod[rt] = np.array([float(w.get(n, 0.0)) for n in NAMES_SCORE])

# ============================================================ Phase1: alphaごとのDirichlet探索
log("\n" + "=" * 72)
log("Phase1: alphaごとのDirichlet探索 + Nested OOF(8-fold) + flop_selection_optimism")
log(f"主ゲート: true_edge_pt/true_edge_sd >= {DECISION_GATE_RATIO}(box_n=4主指標)")
log("=" * 72)

gate_results = {}
for box_n in BOX_NS:
    log(f"\n--- box_n={box_n}(重み: {BOX_TO_WEIGHT_KEY[box_n]}) ---")
    ev_search = JE.Evaluator(races_search, actual, box_n=box_n)
    w_score = w_score_prod[BOX_TO_WEIGHT_KEY[box_n]]
    picks_base_search = JE.score_picks(mats_score_search, w_score, box_n)
    eval_base = ev_search.evaluate(picks_base_search)
    log(f"  baseline(alpha=0、現行と同一、探索母集団のみ): model={eval_base['model']:.2f}% "
        f"market={eval_base['market']:.2f}% excess={eval_base['excess']:+.2f}pt")

    for alpha in ALPHAS:
        rng = np.random.default_rng(hash((box_n, alpha)) % (2**31))
        W_POOL = np.column_stack([rng.dirichlet([1.0] * len(shortlist)) for _ in range(N_PATTERNS)])

        predict_fn = JFE.make_flop_predict_fn(ev_search, mats_score_search, w_score,
                                               mats_flop_search, W_POOL, alpha)
        nested_oof = ev_search.group_kfold_oof_generic(predict_fn, n_folds=8, seed=13)
        opt = JFE.flop_selection_optimism(ev_search, mats_score_search, w_score, mats_flop_search,
                                          W_POOL, alpha, n_rep=200, seed=99)
        edge_ratio = (opt["true_edge_pt"] / opt["true_edge_sd"]) if opt["true_edge_sd"] > 0 else 0.0
        gate_pass = edge_ratio >= DECISION_GATE_RATIO

        log(f"  [alpha={alpha}] Nested OOF: excess={nested_oof['excess']:+.2f}pt "
            f"(fold argmax unique: {nested_oof['fold_argmax_unique']}/{nested_oof['n_folds']})"
            f"  true_edge={opt['true_edge_pt']:+.2f}pt(sd={opt['true_edge_sd']:.2f}, "
            f"ratio={edge_ratio:+.2f})  主ゲート={'PASS' if gate_pass else 'NO'}")

        gate_results.setdefault(f"alpha_{alpha}", {})[f"box{box_n}"] = {
            "baseline_excess_pt": eval_base["excess"],
            "nested_oof_excess_pt": nested_oof["excess"],
            "nested_oof_fold_argmax_unique": nested_oof["fold_argmax_unique"],
            "nested_oof_n_folds": nested_oof["n_folds"],
            "selection_optimism": opt,
            "edge_ratio": edge_ratio,
            "gate_pass": bool(gate_pass),
        }

# ============================================================ tie-break(事前登録済み規則)
log("\n" + "=" * 72)
log("Phase1 まとめ・tie-break(box_n=4主指標、PASSしたalphaのうち最小値を採用)")
log("=" * 72)
passing_alphas_box4 = [a for a in ALPHAS if gate_results[f"alpha_{a}"]["box4"]["gate_pass"]]
for alpha in ALPHAS:
    g4 = gate_results[f"alpha_{alpha}"]["box4"]
    consistent = all(gate_results[f"alpha_{alpha}"][f"box{b}"]["gate_pass"] == g4["gate_pass"]
                     for b in BOX_NS)
    log(f"  alpha={alpha}: box4 edge_ratio={g4['edge_ratio']:+.2f} "
        f"主ゲート={'PASS' if g4['gate_pass'] else 'NO'}(box5/3一貫: {consistent})")

chosen_alpha = min(passing_alphas_box4) if passing_alphas_box4 else None
log(f"\n総合判定: {'box4でPASSしたalpha=' + str(chosen_alpha) + 'を採用' if chosen_alpha else '全alphaでREJECT(主ゲート不通過)'}")

# ============================================================ Phase2: held-out参考確認(PASS時のみ)
holdout_result = None
if chosen_alpha is not None:
    log("\n" + "=" * 72)
    log(f"Phase2: held-out参考確認(alpha={chosen_alpha}、box_n=4、"
        f"直近{len(holdout_dates)}日は探索に使っていない未見データ)")
    log("=" * 72)
    box_n = 4
    w_score = w_score_prod[BOX_TO_WEIGHT_KEY[box_n]]
    rng = np.random.default_rng(hash((box_n, chosen_alpha)) % (2**31))
    W_POOL = np.column_stack([rng.dirichlet([1.0] * len(shortlist)) for _ in range(N_PATTERNS)])
    # 探索母集団(train)でargmaxしたパターンを、全レース(search+holdout)に適用する
    all_picks_search = [JFE.adjusted_picks(mats_score_search, w_score, mats_flop_search,
                                           W_POOL[:, j], chosen_alpha, box_n)
                        for j in range(N_PATTERNS)]
    ev_search = JE.Evaluator(races_search, actual, box_n=box_n)
    all_st, all_rt = [], []
    for p in all_picks_search:
        s, r = ev_search.settler.returns_for(p)
        all_st.append(s)
        all_rt.append(r)
    vals = np.array([JE.cost_weighted_rate(all_st[j], all_rt[j]) for j in range(N_PATTERNS)])
    best = int(np.argmax(vals))
    w_flop_best = W_POOL[:, best]
    top_w = {n: float(w) for n, w in zip(shortlist, w_flop_best) if w > 0.01}
    log(f"  探索母集団で選ばれた最良パターン(重み1%以上): {json.dumps(top_w, ensure_ascii=False)}")

    ev_full = JE.Evaluator(races_all, actual, box_n=box_n)
    picks_final_full = JFE.adjusted_picks(mats_score_all, w_score, mats_flop_all, w_flop_best,
                                          chosen_alpha, box_n)
    picks_base_full = JE.score_picks(mats_score_all, w_score, box_n)
    holdout_blocks = sorted({f"{r['kaisai_date']}_{r['racecourse']}" for r in races_all
                             if r["kaisai_date"] in holdout_dates})
    diff_ho = ev_full.block_bootstrap_diff(picks_final_full, picks_base_full, seed=41,
                                           block_subset=holdout_blocks)
    log(f"  held-out({len(holdout_blocks)}ブロック)でのペア差分(alpha={chosen_alpha} vs alpha=0): "
        f"95%CI=[{diff_ho['lo']:+.2f},{diff_ho['hi']:+.2f}]  ※参考値、n_blocksが小さく"
        f"広いCIになりやすい(計画の検出力に関する注記参照、必須要件にはしない)")

    for box_n2 in (5, 3):
        w_score2 = w_score_prod[BOX_TO_WEIGHT_KEY[box_n2]]
        ev_full2 = JE.Evaluator(races_all, actual, box_n=box_n2)
        picks_final2 = JFE.adjusted_picks(mats_score_all, w_score2, mats_flop_all, w_flop_best,
                                          chosen_alpha, box_n2)
        picks_base2 = JE.score_picks(mats_score_all, w_score2, box_n2)
        diff2 = ev_full2.block_bootstrap_diff(picks_final2, picks_base2, seed=41,
                                              block_subset=holdout_blocks)
        log(f"  [参考、box_n={box_n2}一貫性確認] held-outペア差分95%CI="
            f"[{diff2['lo']:+.2f},{diff2['hi']:+.2f}]")

    holdout_result = {"chosen_alpha": chosen_alpha, "best_pattern_weights": top_w,
                      "holdout_blocks": holdout_blocks, "diff_box4": diff_ho}

# ============================================================ 保存
DATA_DIR.mkdir(parents=True, exist_ok=True)
result = {
    "shortlist": shortlist,
    "holdout_dates": sorted(holdout_dates), "search_dates": sorted(search_dates),
    "alphas_tested": ALPHAS, "decision_gate_ratio": DECISION_GATE_RATIO,
    "gate_results": gate_results,
    "chosen_alpha": chosen_alpha,
    "holdout_result": holdout_result,
}
OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
log(f"\n保存: {OUT_JSON}")
log(f"保存: {OUT_TXT}")
