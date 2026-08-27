# -*- coding: utf-8 -*-
"""Phase2追加診断: Dirichletパターン探索の選択バイアス(selection_optimism)確認。

nar_search_combined_2026_08_28.pyのbox_n=3(Dirichlet/Ridgeとも現行本番比で95%CI完全に
プラス)・box_n=4(Lasso僅差)について、「パターン(またはC)を選ぶという行為自体の真の
価値」をnar_eval.selection_optimism()(既存、無改造)で確認する。このプロジェクトの
過去の探索(300/500パターン等)では、group_kfold_oof/Nested LOBO OOFでは良く見えても
selection_optimismのtrue_edge_ptがほぼ常に0近傍〜負という結果が繰り返し出ており、
この確認を経ずに「有意」と判断するのは早計(SEレビューでも必ず指摘される観点)。

W_dirichletはnar_search_combined_2026_08_28.pyと同じseed・tierで再構築する(決定的、
同一結果になることをNAMES/POOL一致で保証)。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nar_dataset
import nar_eval as NE
import nar_signals as NS

OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
GATE_JSON = OUT_DIR / "nar_signal_gate_v5_2026_08_27_result.json"
OUT_JSON = OUT_DIR / "nar_check_selection_optimism_2026_08_28_result.json"
OUT_TXT = OUT_DIR / "nar_check_selection_optimism_2026_08_28_report.txt"

SEED = 17
WEIGHT_TIERS = [(100.0, 210), (25.0, 210), (6.0, 140), (1.0, 139)]
CHECK_BOX_NS = [3, 4]

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


gate = json.loads(GATE_JSON.read_text(encoding="utf-8"))
POOL_TRUE_PROD = gate["pool_true_prod"]
track_a, track_b = gate["track_a"], gate["track_b"]
gate_results = gate["gate_results"]
v5_survivors = [c for c in track_a + track_b if gate_results[c]["box4"]["gate_pass"]]
DIRICHLET_POOL = POOL_TRUE_PROD + v5_survivors
log(f"Dirichlet探索POOL({len(DIRICHLET_POOL)}本): {DIRICHLET_POOL}")

data = nar_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
priors_all = NS.make_priors(races)
log(f"レース数: {len(races)}")

NAMES = NS.ALL_SIGNALS_V5
mats_all = NS.signal_matrices(races, priors_all, NAMES)


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


def equal_w(subset) -> np.ndarray:
    return wvec({n: 1.0 / len(subset) for n in subset})


result = {}
for box_n in CHECK_BOX_NS:
    log("\n" + "=" * 72)
    log(f"box_n={box_n}: selection_optimism診断(Dirichlet 700パターン)")
    log("=" * 72)
    ev = NE.Evaluator(races, actual, box_n=box_n)

    rng = np.random.default_rng(SEED + box_n)
    cols = [equal_w(DIRICHLET_POOL)]
    for concentration, n in WEIGHT_TIERS:
        alpha = [concentration] * len(DIRICHLET_POOL)
        for _ in range(n):
            cols.append(wvec(dict(zip(DIRICHLET_POOL, rng.dirichlet(alpha)))))
    W_dirichlet = np.column_stack(cols)
    log(f"  パターン数: {W_dirichlet.shape[1]}(nar_search_combined_2026_08_28.pyと同一seed"
        f"で再構築、決定的に一致するはず)")

    opt = NE.selection_optimism(ev, mats_all, W_dirichlet, n_rep=200, seed=99)
    log(f"  選択側(見た側)の平均      : {opt['selected_side']:.2f}%")
    log(f"  未使用側での選抜値の平均    : {opt['unseen_side']:.2f}%")
    log(f"  未使用側の全パターン平均    : {opt['unseen_all_mean']:.2f}%")
    log(f"  楽観バイアス               : {opt['optimism_pt']:+.2f}pt")
    log(f"  選ぶことの真の価値(true_edge_pt): {opt['true_edge_pt']:+.2f}pt "
        f"(sd={opt['true_edge_sd']:.2f})")
    log(f"  未使用側で全パターン平均を上回る確率: {opt['win_rate']:.0%}")
    ratio = opt["true_edge_pt"] / opt["true_edge_sd"] if opt["true_edge_sd"] else float("nan")
    gate_g1 = ratio >= 2.0
    log(f"  true_edge_pt/sd比 = {ratio:.3f}  従来ゲート(>=2.0)判定: "
        f"{'PASS' if gate_g1 else 'NO'}")
    result[f"box{box_n}"] = {**opt, "true_edge_ratio": ratio, "gate_g1_pass": gate_g1}

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
log(f"\n保存: {OUT_JSON}")
