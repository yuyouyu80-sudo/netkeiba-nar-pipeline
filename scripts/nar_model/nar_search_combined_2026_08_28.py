# -*- coding: utf-8 -*-
"""Phase2: 組み合わせ探索(2026-08-27、全ファクター統合・高確信度選抜計画)。

Dirichletパターン探索(既存tier方式の拡張)・Ridge・Lasso・現行本番の4手法を、
nar_eval.Evaluator.group_kfold_oof_generic(同一ブロック分割・同一評価指標)で比較する。
単一の勝者に絞らず複数の有力候補(Dirichletトップ1・Ridge最良C・Lasso最良C)を最終候補
として残す。

前提: nar_signal_gate_v5_2026_08_27.py の出力(nar_signal_gate_v5_2026_08_27_result.json)を
読み、Phase1でbox4のG1をPASSしたトラックA/B候補をDirichlet探索POOLに追加する
(Ridge/Lassoには計画通りPhase1不合格分も含む全候補=ALL_SIGNALS_V5全体を入力する)。

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
OUT_JSON = OUT_DIR / "nar_search_combined_2026_08_28_result.json"
OUT_TXT = OUT_DIR / "nar_search_combined_2026_08_28_report.txt"

BOX_NS = [5, 4, 3]
N_FOLDS = 8
SEED = 17
WEIGHT_TIERS = [(100.0, 210), (25.0, 210), (6.0, 140), (1.0, 139)]  # 700本(既存500から拡張)
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


# ============================================================ Phase1結果の読み込み
gate = json.loads(GATE_JSON.read_text(encoding="utf-8"))
POOL_TRUE_PROD = gate["pool_true_prod"]
track_a, track_b = gate["track_a"], gate["track_b"]
gate_results = gate["gate_results"]
v5_survivors = [c for c in track_a + track_b if gate_results[c]["box4"]["gate_pass"]]
log(f"POOL_TRUE_PROD({len(POOL_TRUE_PROD)}本): {POOL_TRUE_PROD}")
log(f"Phase1 box4 G1通過({len(v5_survivors)}本): {v5_survivors}")
DIRICHLET_POOL = POOL_TRUE_PROD + v5_survivors
log(f"Dirichlet探索POOL({len(DIRICHLET_POOL)}本): {DIRICHLET_POOL}")

# Ridge/Lassoは計画通りPhase1不合格分も含む全候補(V4+スクリーニング後V5)を入力する。
LOGISTIC_NAMES = POOL_TRUE_PROD + track_a + track_b
log(f"Ridge/Lasso入力候補({len(LOGISTIC_NAMES)}本、Phase1不合格分も含む): {LOGISTIC_NAMES}")

# ============================================================ データロード
data = nar_dataset.load(rebuild=True)
races, actual = data["races"], data["actual"]
priors_all = NS.make_priors(races)
log(f"レース数: {len(races)}  日付: {data['dates'][0]}〜{data['dates'][-1]}")

NAMES = NS.ALL_SIGNALS_V5  # 列順の名前空間(既存ALL_SIGNALSは不使用、無改造)
mats_all = NS.signal_matrices(races, priors_all, NAMES)


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


def equal_w(subset) -> np.ndarray:
    return wvec({n: 1.0 / len(subset) for n in subset})


def make_dirichlet_fit_fn(W: np.ndarray, box_n: int):
    """Dirichletパターン群Wの中からtrain_idx上でin-sample最良のものをargmax選択し、
    test_idxに適用するpredict_fn。

    評価はNL._fast_excess(settler.tables/mkt_stake/mkt_retをidx分だけ直接読む軽量版)を
    使う。当初ev_full.evaluate(picks_full, idx=idx)(全レース分のpicks_fullを毎回構築し
    BoxSettler.returns_for()で全レースをスキャンしてからidxで絞る設計)で実装したが、
    700パターン×8fold×3box_nの組み合わせで実測5分超でも完了せず(2026-08-27)、
    ボトルネックと判明したため置き換えた(nar_logistic.pyのmake_fit_fnで発見した問題と
    同型、こちらはロジスティック回帰ではなくDirichletパターン選択だが同じev.evaluate()の
    全件スキャンコストを踏んでいた)。"""
    ev_full = NE.Evaluator(races, actual, box_n=box_n)
    cols = [NE.NB.BET_TYPES.index(b) for b in NE.OBJ_BETS]

    def score_on(idx: np.ndarray, w: np.ndarray) -> float:
        picks_sub = NE.score_picks([mats_all[i] for i in idx], w, box_n)
        return NL._fast_excess(ev_full, cols, idx, picks_sub)

    def predict_fn(train_idx: np.ndarray, test_idx: np.ndarray):
        vals = np.array([score_on(train_idx, W[:, j]) for j in range(W.shape[1])])
        best = int(np.argmax(vals))
        test_picks = NE.score_picks([mats_all[i] for i in test_idx], W[:, best], box_n)
        return test_picks, best

    return predict_fn


def make_fixed_fit_fn(w: np.ndarray, box_n: int):
    """固定重み(現行本番)用: 何も学習せず常に同じwを返す(chosen=0固定)。"""
    def predict_fn(train_idx: np.ndarray, test_idx: np.ndarray):
        test_picks = NE.score_picks([mats_all[i] for i in test_idx], w, box_n)
        return test_picks, 0
    return predict_fn


results_by_box = {}
for box_n in BOX_NS:
    log("\n" + "=" * 72)
    log(f"box_n={box_n}")
    log("=" * 72)
    ev = NE.Evaluator(races, actual, box_n=box_n)

    rng = np.random.default_rng(SEED + box_n)
    cols = [equal_w(DIRICHLET_POOL)]  # pattern#0固定: 厳密な等重み
    for concentration, n in WEIGHT_TIERS:
        alpha = [concentration] * len(DIRICHLET_POOL)
        for _ in range(n):
            cols.append(wvec(dict(zip(DIRICHLET_POOL, rng.dirichlet(alpha)))))
    W_dirichlet = np.column_stack(cols)
    log(f"  Dirichletパターン数: {W_dirichlet.shape[1]}")

    methods = {}

    fit_fn = make_dirichlet_fit_fn(W_dirichlet, box_n)
    oof = ev.group_kfold_oof_generic(fit_fn, n_folds=N_FOLDS, seed=SEED)
    methods["dirichlet"] = oof
    log(f"  [Dirichlet] excess={oof['excess']:+.2f}pt fold_argmax_unique="
        f"{oof['fold_argmax_unique']}/{oof['n_folds']}")

    for method in ["ridge", "lasso"]:
        fit_fn = NL.make_fit_fn(races, actual, mats_all, NAMES, LOGISTIC_NAMES, method,
                                C_GRID, box_n, label="place3", inner_n_folds=5, seed=SEED)
        oof = ev.group_kfold_oof_generic(fit_fn, n_folds=N_FOLDS, seed=SEED)
        methods[method] = oof
        c_choices = oof["fold_argmax_choices"]
        log(f"  [{method}] excess={oof['excess']:+.2f}pt fold_argmax_unique="
            f"{oof['fold_argmax_unique']}/{oof['n_folds']}  C選択={c_choices}")

    fit_fn = make_fixed_fit_fn(equal_w(POOL_TRUE_PROD), box_n)
    oof = ev.group_kfold_oof_generic(fit_fn, n_folds=N_FOLDS, seed=SEED)
    methods["current_prod"] = oof
    log(f"  [現行本番] excess={oof['excess']:+.2f}pt (固定重み、fitなし)")

    log("\n  --- 現行本番比のペア差分ブートストラップ ---")
    comparisons = {}
    for name in ["dirichlet", "ridge", "lasso"]:
        diff = ev.paired_block_bootstrap(methods[name]["picks"], methods["current_prod"]["picks"])
        comparisons[name] = diff
        log(f"  {name} vs 現行本番: 差分95%CI=[{diff['lo']:+.2f},{diff['hi']:+.2f}] "
            f"mean={diff['mean']:+.2f}pt")

    results_by_box[box_n] = {
        "methods": {k: {kk: vv for kk, vv in v.items() if kk != "picks"} for k, v in methods.items()},
        "vs_current_prod": comparisons,
    }

# ============================================================ 保存
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps({
    "pool_true_prod": POOL_TRUE_PROD, "v5_survivors": v5_survivors,
    "dirichlet_pool": DIRICHLET_POOL, "logistic_names": LOGISTIC_NAMES,
    "n_races": len(races), "results_by_box": results_by_box,
}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
log(f"\n保存: {OUT_JSON}")
log(f"保存: {OUT_TXT}")
