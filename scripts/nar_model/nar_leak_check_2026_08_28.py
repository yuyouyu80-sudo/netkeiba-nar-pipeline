# -*- coding: utf-8 -*-
"""Phase4レビュー2の必須指摘#1対応: box3 DirichletのOOF市場超過+6.38pt(現行本番比、
Holm/BH/Bonferroni全通過の唯一の発見)が、Phase1→Phase2のプール選抜リークを含んでいないか
検証する。

リークの構造: corner4_positionは nar_signal_gate_v5_2026_08_27.py で「全1643レース」を
使ってPhase1ゲートを通過し(box4 G1 PASS)、その結果が nar_search_combined_2026_08_28.py の
DIRICHLET_POOL(18本 = POOL_TRUE_PROD 17本 + corner4_position)に採用されている。つまり
corner4_positionという"特徴選択"の判断自体が、Phase2のgroup_kfold_oof_generic の
全outer testフォールドと重なる情報(全データでのゲート判定)を使っている。

本スクリプトは、DIRICHLET_POOLからcorner4_positionを除いたPOOL_TRUE_PROD(17本、既存
本番と同一集合)のみで、nar_search_combined_2026_08_28.pyのbox_n=3 Dirichlet探索を完全に
同じ設定(WEIGHT_TIERS・SEED・N_FOLDS)で再実行し、現行本番比のexcessとペア差分ブートストラップ
CIを比較する。既存スクリプト・モジュールは全て無改造で参照のみ。
"""
import json
from pathlib import Path
import sys

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
COMBINED_JSON = OUT_DIR / "nar_search_combined_2026_08_28_result.json"
OUT_JSON = OUT_DIR / "nar_leak_check_2026_08_28_result.json"
OUT_TXT = OUT_DIR / "nar_leak_check_2026_08_28_report.txt"

BOX_N = 3
N_FOLDS = 8
SEED = 17  # nar_search_combined_2026_08_28.pyと同一(SEED + box_n = 20)
WEIGHT_TIERS = [(100.0, 210), (25.0, 210), (6.0, 140), (1.0, 139)]  # 同一tier構成

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


gate = json.loads(GATE_JSON.read_text(encoding="utf-8"))
POOL_TRUE_PROD = gate["pool_true_prod"]
log(f"POOL_TRUE_PROD(corner4_position抜き、{len(POOL_TRUE_PROD)}本): {POOL_TRUE_PROD}")

data = nar_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
priors_all = NS.make_priors(races)
log(f"レース数: {len(races)}")

NAMES = NS.ALL_SIGNALS_V5  # nar_search_combined_2026_08_28.pyと同一の列順名前空間
mats_all = NS.signal_matrices(races, priors_all, NAMES)


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


def equal_w(subset) -> np.ndarray:
    return wvec({n: 1.0 / len(subset) for n in subset})


def make_dirichlet_fit_fn(W: np.ndarray, box_n: int):
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
    def predict_fn(train_idx: np.ndarray, test_idx: np.ndarray):
        test_picks = NE.score_picks([mats_all[i] for i in test_idx], w, box_n)
        return test_picks, 0
    return predict_fn


ev = NE.Evaluator(races, actual, box_n=BOX_N)
rng = np.random.default_rng(SEED + BOX_N)
cols = [equal_w(POOL_TRUE_PROD)]
for concentration, n in WEIGHT_TIERS:
    alpha = [concentration] * len(POOL_TRUE_PROD)
    for _ in range(n):
        cols.append(wvec(dict(zip(POOL_TRUE_PROD, rng.dirichlet(alpha)))))
W_dirichlet = np.column_stack(cols)
log(f"Dirichletパターン数: {W_dirichlet.shape[1]}(corner4_position抜き17本プール)")

fit_fn = make_dirichlet_fit_fn(W_dirichlet, BOX_N)
oof_leak_free = ev.group_kfold_oof_generic(fit_fn, n_folds=N_FOLDS, seed=SEED)
log(f"[Dirichlet, corner4_position抜き] excess={oof_leak_free['excess']:+.2f}pt "
    f"fold_argmax_unique={oof_leak_free['fold_argmax_unique']}/{oof_leak_free['n_folds']}")

fit_fn_prod = make_fixed_fit_fn(equal_w(POOL_TRUE_PROD), BOX_N)
oof_prod = ev.group_kfold_oof_generic(fit_fn_prod, n_folds=N_FOLDS, seed=SEED)
log(f"[現行本番] excess={oof_prod['excess']:+.2f}pt (固定重み、fitなし)")

diff = ev.paired_block_bootstrap(oof_leak_free["picks"], oof_prod["picks"])
log(f"\ncorner4_position抜きDirichlet vs 現行本番: 差分95%CI=[{diff['lo']:+.2f},{diff['hi']:+.2f}] "
    f"mean={diff['mean']:+.2f}pt")

# ============================================================ 元のcorner4_position込み結果との比較
combined = json.loads(COMBINED_JSON.read_text(encoding="utf-8"))
orig = combined["results_by_box"]["3"]["vs_current_prod"]["dirichlet"]
log(f"\n[参考] corner4_position込み(元のnar_search_combined_2026_08_28.py結果):")
log(f"  vs 現行本番: 差分95%CI=[{orig['lo']:+.2f},{orig['hi']:+.2f}] mean={orig['mean']:+.2f}pt")
log(f"\n差(corner4_position込み - 抜き) = {orig['mean'] - diff['mean']:+.2f}pt")
log("この差が小さければ、+6.38ptの発見はプール選抜リークではなく既存17本の重み再最適化")
log("そのものに由来する(decompositionで示した「新シグナルの限界効果+0.68pt」と整合)。")
log("差が大きければ、リークが結果を実質的に押し上げていたことになり、報告数値の訂正が必要。")

# ============================================================ 保存
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps({
    "pool_true_prod": POOL_TRUE_PROD,
    "box_n": BOX_N,
    "leak_free_dirichlet_excess": oof_leak_free["excess"],
    "leak_free_fold_argmax_unique": oof_leak_free["fold_argmax_unique"],
    "current_prod_excess": oof_prod["excess"],
    "leak_free_vs_current_prod": diff,
    "original_with_corner4_position_vs_current_prod": orig,
    "difference_pt": orig["mean"] - diff["mean"],
}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
log(f"\n保存: {OUT_JSON}")
log(f"保存: {OUT_TXT}")
