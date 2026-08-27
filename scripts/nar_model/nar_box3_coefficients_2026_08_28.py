# -*- coding: utf-8 -*-
"""Phase4レビュー1の指摘対応: box_n=3のRidge/Lasso/Dirichletモデルが実際に何の信号に
重みを置いているか(係数・重み配分)を出力する。これまでの実行では最終係数・重みが
一切保存されておらず、「予想印・展開データが組み合わせで効いているか」というユーザーの
当初仮説を検証できていなかった(レビュー1の中程度指摘の1つ)ため追加実施する。

Ridge/Lassoは全1643レースを使ったブロック5分割CVでCを選び直し(nar_search_combined_
2026_08_28.pyの外側8fold内の個別foldのCではなく、"最終的に本番投入するならこの1本"と
いう単一モデルの係数を見るのが目的のため)、全データで再学習した係数を出す。
Dirichletは同じPOOL・同じseed・同じtierで700パターンを再生成し、全データ上でin-sample
最良の1パターンの重み配分を出す(参考値、in-sample最良である旨を明記)。
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
OUT_JSON = OUT_DIR / "nar_box3_coefficients_2026_08_28_result.json"
OUT_TXT = OUT_DIR / "nar_box3_coefficients_2026_08_28_report.txt"

BOX_N = 3
SEED = 17
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]  # 下限拡張(レビュー1指摘)
WEIGHT_TIERS = [(100.0, 210), (25.0, 210), (6.0, 140), (1.0, 139)]

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


gate = json.loads(GATE_JSON.read_text(encoding="utf-8"))
POOL_TRUE_PROD = gate["pool_true_prod"]
track_a, track_b = gate["track_a"], gate["track_b"]
v5_survivors = [c for c in track_a + track_b if gate["gate_results"][c]["box4"]["gate_pass"]]
DIRICHLET_POOL = POOL_TRUE_PROD + v5_survivors
LOGISTIC_NAMES = POOL_TRUE_PROD + track_a + track_b

data = nar_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
priors_all = NS.make_priors(races)
log(f"レース数: {len(races)}")

NAMES = NS.ALL_SIGNALS_V5
mats_all = NS.signal_matrices(races, priors_all, NAMES)
ev = NE.Evaluator(races, actual, box_n=BOX_N)
cols = [NE.NB.BET_TYPES.index(b) for b in NE.OBJ_BETS]

# ============================================================ Ridge/Lasso: 全データでCを選び直す
X, race_of_row = NL.build_feature_matrix(mats_all, NAMES, LOGISTIC_NAMES)
ranges = NL.race_row_ranges(mats_all)
y = NL.build_labels(races, actual, label="place3")
blocks = NE.blocks_of(races)
block_ids = sorted(set(blocks))

rng = np.random.default_rng(SEED)
order = rng.permutation(len(block_ids))
n_folds = 5
cv_folds = [order[i::n_folds] for i in range(n_folds)]

# レビュー2指摘(中程度): OUT_JSONが定義されているのに一度も書き出されておらず、係数が
# 4桁丸めのテキストにしか残らず再利用・差分比較ができない不備があった。ここに構造化して
# 蓄積し、末尾でOUT_JSONへ書き出す。
coef_results = {}

for method, penalty in [("ridge", "l2"), ("lasso", "l1")]:
    log("\n" + "=" * 72)
    log(f"{method}(box_n=3): 全データ5分割CVでC選定 → 全データ再学習後の係数")
    log("=" * 72)
    best_C, best_score = C_GRID[0], -np.inf
    c_scores = {}
    for C in C_GRID:
        fold_scores = []
        for fold in cv_folds:
            test_blocks = {block_ids[i] for i in fold}
            test_race_idx = np.array([i for i in range(len(races)) if blocks[i] in test_blocks])
            train_race_idx = np.array([i for i in range(len(races)) if blocks[i] not in test_blocks])
            train_rows = np.where(np.isin(race_of_row, train_race_idx))[0]
            params = NL.standardize_fit(X, train_rows)
            Z_train = NL.standardize_apply(X[train_rows], params)
            model = NL.fit_logistic(Z_train, y[train_rows], penalty, C)
            Z_all = NL.standardize_apply(X, params)
            scores = NL.score_from_model(Z_all, model)
            test_picks = NL.picks_from_scores(scores, ranges, test_race_idx, BOX_N)
            fold_scores.append(NL._fast_excess(ev, cols, test_race_idx, test_picks))
        c_scores[C] = float(np.mean(fold_scores))
        if c_scores[C] > best_score:
            best_score, best_C = c_scores[C], C
    log(f"  Cグリッド別5分割CV平均excess: {c_scores}")
    log(f"  選定C={best_C}(excess={best_score:+.2f}pt ※12個のargmax選択値、上方バイアスあり。"
        f"汎化性能の推定値ではない。レビュー2指摘)")

    params = NL.standardize_fit(X, np.arange(len(X)))
    Z_all = NL.standardize_apply(X, params)
    final_model = NL.fit_logistic(Z_all, y, penalty, best_C)
    coef = final_model.coef_[0]
    order_idx = np.argsort(-np.abs(coef))
    log(f"  全データ最終係数(標準化後、|係数|降順):")
    for i in order_idx:
        marker = " <- 予想印/展開(2026-08-27新設)" if LOGISTIC_NAMES[i] in track_a + track_b else ""
        log(f"    {LOGISTIC_NAMES[i]:26s} {coef[i]:+.4f}{marker}")
    n_zero = int((np.abs(coef) < 1e-6).sum())
    zero_note = ""
    if method == "ridge":
        zero_note = "(Ridge=L2は定義上厳密な0を作らない。0本は自明でありスパース性の指標にならない。レビュー2指摘)"
    log(f"  係数が実質ゼロ({n_zero}/{len(coef)}本){zero_note}: "
        f"{[LOGISTIC_NAMES[i] for i in range(len(coef)) if abs(coef[i]) < 1e-6]}")

    coef_results[method] = {
        "c_grid": C_GRID, "c_scores": {str(k): v for k, v in c_scores.items()},
        "selected_C": best_C,
        "selected_C_excess_pt": best_score,
        "selected_C_excess_note": "12点argmax選択値。上方バイアスあり、汎化性能の推定値ではない",
        "names": LOGISTIC_NAMES,
        "coefficients": {LOGISTIC_NAMES[i]: float(coef[i]) for i in range(len(coef))},
        "is_v5_new_signal": {n: (n in track_a + track_b) for n in LOGISTIC_NAMES},
        "n_zero": n_zero,
        "zero_names": [LOGISTIC_NAMES[i] for i in range(len(coef)) if abs(coef[i]) < 1e-6],
    }

# ============================================================ Dirichlet: 全データin-sample最良パターン
log("\n" + "=" * 72)
log("Dirichlet(box_n=3): 全データin-sample最良パターンの重み配分(参考値)")
log("=" * 72)


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


def equal_w(subset) -> np.ndarray:
    return wvec({n: 1.0 / len(subset) for n in subset})


rng2 = np.random.default_rng(SEED + BOX_N)
w_cols = [equal_w(DIRICHLET_POOL)]
for concentration, n in WEIGHT_TIERS:
    alpha = [concentration] * len(DIRICHLET_POOL)
    for _ in range(n):
        w_cols.append(wvec(dict(zip(DIRICHLET_POOL, rng2.dirichlet(alpha)))))
W_dirichlet = np.column_stack(w_cols)

all_idx = np.arange(len(races))
vals = []
for j in range(W_dirichlet.shape[1]):
    picks = NE.score_picks(mats_all, W_dirichlet[:, j], BOX_N)
    vals.append(NL._fast_excess(ev, cols, all_idx, picks))
best_j = int(np.argmax(vals))
log(f"  in-sample最良パターン#{best_j}  excess={vals[best_j]:+.2f}pt(全データ評価、参考値)")
w_best = W_dirichlet[:, best_j]
order_idx = np.argsort(-w_best)
log(f"  重み配分(降順、0除く):")
for i in order_idx:
    if w_best[i] > 1e-4:
        marker = " <- 予想印/展開(2026-08-27新設)" if NAMES[i] in track_a + track_b else ""
        log(f"    {NAMES[i]:26s} {w_best[i]:.4f}{marker}")
log("  ※レビュー2指摘: 700本中のin-sample argmax選択であり、フラットDirichlet(alpha=1)の")
log("  帰無分布(ランダムdraw)の順序統計量と比べて特に極端でない(観測1位は帰無分布の期待値を")
log("  下回る)。この重み配分は「どの信号が効くか」の証拠として提示してはならない")
log("  (詳細はnar_stage1_rigor_check_2026_08_28.pyのシミュレーション結果を参照)。")

dirichlet_result = {
    "pool": DIRICHLET_POOL,
    "best_pattern_index": best_j,
    "best_pattern_excess_pt": float(vals[best_j]),
    "best_pattern_excess_note": "全データin-sample評価。参考値であり汎化性能の推定値ではない",
    "weights": {NAMES[i]: float(w_best[i]) for i in range(len(NAMES)) if w_best[i] > 1e-4},
    "caveat": "帰無分布(Dirichlet(alpha=1)のランダムdraw順序統計量)と比べて重み配分は"
              "特に極端でない。証拠として使用不可(nar_stage1_rigor_check_2026_08_28.py参照)",
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps({
    "box_n": BOX_N, "n_races": len(races),
    "ridge": coef_results.get("ridge"), "lasso": coef_results.get("lasso"),
    "dirichlet": dirichlet_result,
}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
log(f"\n保存: {OUT_JSON}")
log(f"保存: {OUT_TXT}")
