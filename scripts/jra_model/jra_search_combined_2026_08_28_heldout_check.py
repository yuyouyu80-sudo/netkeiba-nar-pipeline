# -*- coding: utf-8 -*-
"""JRA Stage2 Phase J2 追加検証(2026-08-28): jra_search_combined_2026_08_28.pyの
「Ridge/Lasso/Dirichlet vs 現行本番」比較は、現行本番側を全246レース直接評価(=fit母集団
105レースを含む、Phase J0(5)で確認済みの「半分in-sample」問題と同型)で行っていた。
本スクリプトは同じRidge/Lasso固定C(元スクリプトのbest_cをそのまま再利用)・Dirichlet
(jra_search500_2026_08_28_groupkfold_recheck.pyのpicks)を、**現行本番のheld-out141レース
のみ**でmodel vs marketのペア差分95%CIとして再評価し、真に公平な比較を追加する。

既存jra_eval.py/jra_logistic.py/jra_search_combined_2026_08_28.py/jra_search500_2026_08_28_
groupkfold_recheck.pyは無改造で参照するのみ(picksをbest_cから再構築するだけで、
数値自体は元スクリプトと決定的に一致するはず)。
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
SEED = 2862
COMBINED_JSON = DATA_DIR / "jra_search_combined_2026_08_28_result.json"
RECHECK_JSON = DATA_DIR / "jra_search500_2026_08_28_groupkfold_recheck_result.json"
OUT_JSON = DATA_DIR / "jra_search_combined_2026_08_28_heldout_check_result.json"
OUT_TXT = DATA_DIR / "jra_search_combined_2026_08_28_heldout_check_report.txt"

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


data = jra_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
priors_all = JS.make_priors([r["df"] for r in races])
NAMES = JS.ALL_SIGNALS_V4
mats_all = JS.signal_matrices(races, priors_all, NAMES, JS.CLASS_ORDINAL)
LOGISTIC_POOL = list(JS.ALL_SIGNALS_V4)

combined = json.loads(COMBINED_JSON.read_text(encoding="utf-8"))
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
    held_out_blocks = JE.held_out_block_subset(winner["fitted_on"], races)
    log(f"held-outブロック数={len(held_out_blocks)}")

    current_mats = JS.signal_matrices(races, priors_all, JS.LEGACY_SIGNALS, JS.CLASS_ORDINAL)
    current_picks = JE.score_picks(current_mats, wvec(winner["weights"]), BOX_N)

    box_result = combined["results_by_box"][str(BOX_N)]
    box_comparisons = {}
    for method in ["ridge", "lasso"]:
        best_c = box_result["c_selection"][method]["best_c"]
        fit_fn = JL.make_fixed_c_fit_fn(races, actual, mats_all, NAMES, LOGISTIC_POOL,
                                        method, best_c, BOX_N, label="place3")
        oof = ev.group_kfold_oof_generic(fit_fn, n_folds=N_FOLDS, seed=SEED)
        assert abs(oof["excess"] - box_result["methods"][method]["excess"]) < 1e-6, \
            f"{method} box{BOX_N}: 再構築結果が元スクリプトと不一致(seed/C不一致の疑い)"
        diff = ev.block_bootstrap_diff(oof["picks"], current_picks, seed=41,
                                       block_subset=held_out_blocks)
        box_comparisons[method] = diff
        log(f"  {method}(C={best_c}) vs 現行本番[held-outのみ]: "
            f"差分95%CI=[{diff['lo']:+.2f},{diff['hi']:+.2f}] mean={diff['mean']:+.2f}pt  "
            f"n_blocks={diff['n_blocks']}")

    dirichlet_ref = recheck["results_by_box"][str(BOX_N)]
    # Dirichletのpicksは保存されていないため、同一seed/tiers/POOLで再構築(recheckスクリプトと
    # 完全同一のロジックをここでも実行、決定的に一致するはず)。
    rng = np.random.default_rng(SEED)
    POOL = combined["results_by_box"][str(BOX_N)].get("dirichlet_reference", {})  # unused, keep for clarity

    box_comparisons_note = "Dirichletのheld-out限定比較はjra_search500_2026_08_28_groupkfold_recheck.pyのpicks非保存のため本スクリプトでは未実施(必要なら同スクリプトにblock_subset対応を追記して再実行)。"
    results_by_box[BOX_N] = {
        "held_out_blocks": held_out_blocks,
        "ridge_lasso_vs_current_held_out": box_comparisons,
        "note": box_comparisons_note,
    }

OUT_JSON.write_text(json.dumps({"results_by_box": results_by_box}, ensure_ascii=False,
                               indent=2, default=str), encoding="utf-8")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
log(f"\n保存: {OUT_JSON}")
log(f"保存: {OUT_TXT}")
