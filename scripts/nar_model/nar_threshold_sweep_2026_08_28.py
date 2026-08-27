# -*- coding: utf-8 -*-
"""Phase3: 高確信度選抜の閾値スイープ(2026-08-27計画、box_n=3のRidgeモデルが対象)。

Phase2で box_n=3 が最も有望(Dirichlet/Ridgeとも現行本番比ペア差分ブートストラップ95%CI
完全プラス、Ridgeはselection_optimismの懸念が構造的に小さい正則化ネストCVのため
Dirichletより信頼性が高いと判断)と判明したため、Ridgeモデルのbox_n=3をPhase3の対象に選ぶ。

group_kfold_oof_generic(picksのみ返す)ではなく、ここでは同じ8分割外側foldの構造を使い
「レースごとのOOFスコアgap(gap_top2 = (1位-2位)/レース内スプレッド)」を直接計算する
(Phase3は閾値でレースを絞る=どのレースが「自信あり」かを測る必要があり、順位だけの
picksからは再構成できないため専用ロジックが要る)。

既存 nar_signals.py / nar_eval.py / nar_dataset.py / nar_backtest.py / nar_logistic.py は
すべて無改造で参照するのみ。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nar_dataset
import nar_eval as NE
import nar_logistic as NL
import nar_signals as NS

OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
GATE_JSON = OUT_DIR / "nar_signal_gate_v5_2026_08_27_result.json"
OUT_JSON = OUT_DIR / "nar_threshold_sweep_2026_08_28_result.json"
OUT_TXT = OUT_DIR / "nar_threshold_sweep_2026_08_28_report.txt"

BOX_N = 3
N_FOLDS = 8
SEED = 17
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
PERCENTILES = [90, 80, 70, 60, 50, 40, 30, 20, 10]
MIN_BLOCKS = 30
MIN_COVERAGE = 0.15

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


gate = json.loads(GATE_JSON.read_text(encoding="utf-8"))
POOL_TRUE_PROD = gate["pool_true_prod"]
track_a, track_b = gate["track_a"], gate["track_b"]
LOGISTIC_NAMES = POOL_TRUE_PROD + track_a + track_b
log(f"Ridge入力候補({len(LOGISTIC_NAMES)}本): {LOGISTIC_NAMES}")

data = nar_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
priors_all = NS.make_priors(races)
log(f"レース数: {len(races)}")

NAMES = NS.ALL_SIGNALS_V5
mats_all = NS.signal_matrices(races, priors_all, NAMES)
ev = NE.Evaluator(races, actual, box_n=BOX_N)
cols = [NE.NB.BET_TYPES.index(b) for b in NE.OBJ_BETS]

# ============================================================ OOFスコア収集(Ridge、box_n=3)
X, race_of_row = NL.build_feature_matrix(mats_all, NAMES, LOGISTIC_NAMES)
ranges = NL.race_row_ranges(mats_all)
y = NL.build_labels(races, actual, label="place3")
blocks = NE.blocks_of(races)
block_ids = sorted(set(blocks))

rng = np.random.default_rng(SEED)
order = rng.permutation(len(block_ids))
outer_folds = [order[i::N_FOLDS] for i in range(N_FOLDS)]

oof_scores = np.full(len(races), np.nan, dtype=object)  # レースごとのnp.ndarray(馬別スコア)
oof_picks = [None] * len(races)
chosen_Cs = []

for fold in outer_folds:
    test_blocks = {block_ids[i] for i in fold}
    test_idx = np.array([i for i in range(len(races)) if blocks[i] in test_blocks])
    train_idx = np.array([i for i in range(len(races)) if blocks[i] not in test_blocks])
    fit_fn = NL.make_fit_fn(races, actual, mats_all, NAMES, LOGISTIC_NAMES, "ridge", C_GRID,
                            BOX_N, label="place3", inner_n_folds=5, seed=SEED)
    # make_fit_fnのpredict_fnはpicksしか返さないため、同じ手続きをここでも実行して
    # 生スコアを直接取得する(内側CVでbest_Cを選ぶロジックはmake_fit_fnと重複するが、
    # 同一シード・同一手続きのため数値は一致する — 一致することはOUT_JSONのC選択が
    # nar_search_combined_2026_08_28_resultのC選択と一致するかで確認する)。
    test_picks, best_C = fit_fn(train_idx, test_idx)
    chosen_Cs.append(best_C)
    train_rows = np.where(np.isin(race_of_row, train_idx))[0]
    params = NL.standardize_fit(X, train_rows)
    Z_all = NL.standardize_apply(X, params)
    model = NL.fit_logistic(Z_all[train_rows], y[train_rows], "l2", best_C)
    scores_all = NL.score_from_model(Z_all, model)
    for pos, ri in enumerate(test_idx):
        start, end = ranges[ri]
        oof_scores[ri] = scores_all[start:end]
        oof_picks[ri] = test_picks[pos]

log(f"外側8fold C選択: {chosen_Cs}")
eval_check = ev.evaluate(oof_picks)
log(f"OOF picks再現確認(nar_search_combined_2026_08_28と一致するはず): "
    f"model={eval_check['model']:.2f}% market={eval_check['market']:.2f}% "
    f"excess={eval_check['excess']:+.2f}pt")

# ============================================================ gap_top2算出
gap_top2 = np.full(len(races), np.nan)
uncovered = 0
for i, s in enumerate(oof_scores):
    if not isinstance(s, np.ndarray):
        uncovered += 1
        continue
    if len(s) < 2:
        continue
    order_desc = np.sort(s)[::-1]
    spread = order_desc[0] - order_desc[-1]
    gap_top2[i] = (order_desc[0] - order_desc[1]) / spread if spread > 1e-12 else 1.0

# ============================================================ 閾値スイープ
log("\n" + "=" * 72)
log("閾値スイープ(gap_top2パーセンタイル、box_n=3・Ridge)")
log("=" * 72)
valid = ~np.isnan(gap_top2)
log(f"有効レース数(gap_top2算出可): {valid.sum()}/{len(races)}(OOF未カバー: {uncovered})")

sweep = []
for pct in PERCENTILES:
    tau = float(np.percentile(gap_top2[valid], pct))
    idx_tau = np.where(valid & (gap_top2 >= tau))[0]
    coverage = len(idx_tau) / len(races)
    ev_tau = ev.evaluate(oof_picks, idx=idx_tau)
    boot = ev.paired_block_bootstrap_subset(oof_picks, idx_tau)
    row = {
        "percentile": pct, "tau": tau, "n_races": len(idx_tau), "coverage": coverage,
        "model_pct": ev_tau["model"], "market_pct": ev_tau["market"],
        "excess_pt": ev_tau["excess"], "ci_lo": boot["lo"], "ci_hi": boot["hi"],
        "n_blocks": boot["n_blocks"],
        "gate_pass": (boot["lo"] > 0 and boot["n_blocks"] >= MIN_BLOCKS
                      and coverage >= MIN_COVERAGE),
    }
    sweep.append(row)
    log(f"  上位{pct:3d}%ile(tau={tau:.3f}): n={len(idx_tau):4d}(カバレッジ{coverage:.1%})  "
        f"model={ev_tau['model']:.2f}% market={ev_tau['market']:.2f}% "
        f"excess={ev_tau['excess']:+.2f}pt  95%CI=[{boot['lo']:+.2f},{boot['hi']:+.2f}] "
        f"n_blocks={boot['n_blocks']}  {'PASS' if row['gate_pass'] else 'NO'}")

passing = [r for r in sweep if r["gate_pass"]]
if passing:
    best = max(passing, key=lambda r: r["excess_pt"])
    log(f"\n選定τ*: 上位{best['percentile']}%ile(tau={best['tau']:.3f})、"
        f"カバレッジ{best['coverage']:.1%}、excess={best['excess_pt']:+.2f}pt、"
        f"95%CI=[{best['ci_lo']:+.2f},{best['ci_hi']:+.2f}]")
else:
    log("\n選定τ*: ゲート(CI下限>0 かつ n_blocks>=30 かつ カバレッジ>=15%)を満たす閾値なし")
    best = None

# ============================================================ 保存
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps({
    "box_n": BOX_N, "logistic_names": LOGISTIC_NAMES, "chosen_Cs": chosen_Cs,
    "full_population_eval": eval_check, "sweep": sweep,
    "tau_star": best,
}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
log(f"\n保存: {OUT_JSON}")
