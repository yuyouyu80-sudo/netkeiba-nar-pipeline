# -*- coding: utf-8 -*-
"""JRA Stage2 Phase J2(2026-08-28): 正則化ロジスティック回帰(Ridge/Lasso、固定C)を
box5/4/3に適用し、既存Dirichlet探索(jra_search500_2026_08_28_groupkfold_recheck.py)・
現行本番と同一のgroup_kfold_oof_generic/現行本番評価で比較する。

POOL: ALL_SIGNALS_V4(41本、LEGACY10+CANDIDATE9+V2の6+V3の5+V4の11)。NAR Stage1と同じ
方針(個別ゲート不合格分も含む全候補を正則化モデルへ入力し、組み合わせで効くかを確認する)。

Cは事前に固定(jra_logistic.select_fixed_cで全データ1回の5分割CV参考評価→事前登録した
グリッドから1点選定、レビュー2のJRA移植提言によりネストCVは行わない)。

既存 jra_signals.py / jra_eval.py / jra_logistic.py / jra_dataset.py / jra_backtest.py は
すべて無改造で参照するのみ。
"""
import json
import sys
from pathlib import Path

import numpy as np

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
sys.path.insert(0, str(LIB_DIR))
import jra_dataset  # noqa: E402
import jra_eval as JE  # noqa: E402
import jra_logistic as JL  # noqa: E402
import jra_signals as JS  # noqa: E402

BOX_NS = (4, 5, 3)
WINNER_FILES = {5: "winner_v3.json", 4: "winner_box4.json", 3: "winner_box3.json"}
N_FOLDS = 8
SEED = 2862  # 既存Phase2探索群と同一系列
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]  # 事前登録グリッド(小さめ、レビュー2のJRA提言)
RECHECK_JSON = DATA_DIR / "jra_search500_2026_08_28_groupkfold_recheck_result.json"
OUT_JSON = DATA_DIR / "jra_search_combined_2026_08_28_result.json"
OUT_TXT = DATA_DIR / "jra_search_combined_2026_08_28_report.txt"

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


data = jra_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
priors_all = JS.make_priors([r["df"] for r in races])
log(f"レース数: {len(races)}  日付: {data['dates'][0]}〜{data['dates'][-1]}({len(data['dates'])}日)")

NAMES = JS.ALL_SIGNALS_V4  # 列順の名前空間(既存ALL_SIGNALSは不使用、無改造)
mats_all = JS.signal_matrices(races, priors_all, NAMES, JS.CLASS_ORDINAL)
LOGISTIC_POOL = list(JS.ALL_SIGNALS_V4)
log(f"Ridge/Lasso入力候補({len(LOGISTIC_POOL)}本、個別ゲート不合格分も含む全候補): {LOGISTIC_POOL}")

recheck = json.loads(RECHECK_JSON.read_text(encoding="utf-8"))


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in JS.LEGACY_SIGNALS])


results_by_box = {}
for BOX_N in BOX_NS:
    log("\n" + "=" * 72)
    log(f"box_n={BOX_N}")
    log("=" * 72)
    ev = JE.Evaluator(races, actual, box_n=BOX_N)

    winner = json.loads((DATA_DIR / WINNER_FILES[BOX_N]).read_text(encoding="utf-8"))
    current_mats = JS.signal_matrices(races, priors_all, JS.LEGACY_SIGNALS, JS.CLASS_ORDINAL)
    current_picks = JE.score_picks(current_mats, wvec(winner["weights"]), BOX_N)
    r_current = ev.evaluate(current_picks)
    log(f"[現行本番] 市場差={r_current['excess']:+.2f}pt(固定重み、fitなし、全{len(races)}レース直接評価)")

    methods = {}
    c_selection = {}
    for method in ["ridge", "lasso"]:
        sel = JL.select_fixed_c(races, actual, mats_all, NAMES, LOGISTIC_POOL, method,
                                C_GRID, BOX_N, label="place3", n_folds=5, seed=SEED)
        best_c = sel["best_c"]
        c_selection[method] = sel
        log(f"  [{method}] C候補別5分割CV参考excess: "
            f"{ {k: round(v, 2) for k, v in sel['c_scores'].items()} }")
        log(f"  [{method}] 固定採用C={best_c}(参考値の中でのargmax。この後の外側OOFでは"
            f"再最適化しない)")

        fit_fn = JL.make_fixed_c_fit_fn(races, actual, mats_all, NAMES, LOGISTIC_POOL,
                                        method, best_c, BOX_N, label="place3")
        oof = ev.group_kfold_oof_generic(fit_fn, n_folds=N_FOLDS, seed=SEED)
        methods[method] = oof
        log(f"  [{method}] group_kfold OOF({N_FOLDS}分割) 市場差={oof['excess']:+.2f}pt  "
            f"chosen(=C、全fold固定のはず)={set(oof['fold_argmax_choices'])}")

    log("\n  --- 現行本番比のペア差分ブートストラップ ---")
    comparisons = {}
    for name in ["ridge", "lasso"]:
        diff = ev.block_bootstrap_diff(methods[name]["picks"], current_picks, seed=41)
        comparisons[name] = diff
        log(f"  {name} vs 現行本番: 差分95%CI=[{diff['lo']:+.2f},{diff['hi']:+.2f}] "
            f"mean={diff['mean']:+.2f}pt")

    # Dirichlet(既存recheckスクリプトの結果を再利用、再計算しない)
    dirichlet_ref = recheck["results_by_box"][str(BOX_N)]
    log(f"\n  [参考: Dirichlet(500パターン、jra_search500_2026_08_28_groupkfold_recheck.py)] "
        f"市場差={dirichlet_ref['group_kfold_oof']['excess']:+.2f}pt  "
        f"現行本番比CI=[{dirichlet_ref['bootstrap_gkf_vs_current']['lo']:+.2f},"
        f"{dirichlet_ref['bootstrap_gkf_vs_current']['hi']:+.2f}]")

    results_by_box[BOX_N] = {
        "current_model": r_current,
        "c_selection": {m: {"c_scores": s["c_scores"], "best_c": s["best_c"]}
                        for m, s in c_selection.items()},
        "methods": {k: {kk: vv for kk, vv in v.items() if kk != "picks"} for k, v in methods.items()},
        "vs_current_model": comparisons,
        "dirichlet_reference": {
            "excess": dirichlet_ref["group_kfold_oof"]["excess"],
            "vs_current_model": dirichlet_ref["bootstrap_gkf_vs_current"],
            "selection_optimism_decision": dirichlet_ref["decision"],
        },
    }

OUT_JSON.write_text(json.dumps({
    "logistic_pool": LOGISTIC_POOL, "n_races": len(races), "c_grid": C_GRID,
    "results_by_box": results_by_box,
}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
log(f"\n保存: {OUT_JSON}")
log(f"保存: {OUT_TXT}")
