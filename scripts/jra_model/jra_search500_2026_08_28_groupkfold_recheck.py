# -*- coding: utf-8 -*-
"""JRA Stage2 Phase J1再検証(2026-08-28)。同一セッション群の別インスタンス(claude-f3)が
commit 7c57477で実施した`jra_search500_2026_08_28_v4signals.py`(POOL=LEGACY_SIGNALS10本+
Phase1生存候補corner_transition_gap、SEED=2862、WEIGHT_TIERS同一、500パターン)を、
評価方法だけ`lobo_oof`(leave-one-block-out)から`jra_eval.Evaluator.group_kfold_oof_generic`
(2026-08-28本セッションでjra_eval.pyへ追加、8分割・held-out比率1/8=12.5%)に差し替えて
再実行する。

理由: 元スクリプトの結果自体が「Nested LOBO OOFがfold毎の選択パターン2〜3/42まで退化
(=ほぼ全foldで同一パターンが選ばれる)」と自己申告しており(commit 7c57477メッセージ参照)、
これはNAR Stage1(2026-08-20)で発見した「LOBOはブロック数が多いとheld-out比率が小さすぎて
argmaxが実質固定される」という既知の構造的欠陥と同型。held-out比率を大きくした
group_kfold_oof_genericでargmaxが実際に変動するかを確認し、より信頼できるOOF市場差・
selection_optimism診断を得る。

POOL・SEED・WEIGHT_TIERS・NAMES(ALL_SIGNALS_V4)・現行本番ファイルはすべて元スクリプトと
完全一致させる(公正な比較のため)。jra_signals.py/jra_eval.py本体・元スクリプトは
無改造で参照するのみ。
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
import jra_signals as JS  # noqa: E402

N_PATTERNS = 500
SEED = 2862  # 元スクリプトと同一(公正な比較のため)
N_FOLDS = 8  # group_kfold_oof_generic用(held-out比率1/8=12.5%、LOBOの1/42=2.4%より大幅に大)
BOX_NS = (4, 5, 3)
WINNER_FILES = {5: "winner_v3.json", 4: "winner_box4.json", 3: "winner_box3.json"}
DECISION_GATE_RATIO = 2.0

GATE_JSON = DATA_DIR / "jra_signal_gate_v4_2026_08_28_result.json"
ORIG_SEARCH_JSON = DATA_DIR / "jra_search500_2026_08_28_v4signals_result.json"
OUT_JSON = DATA_DIR / "jra_search500_2026_08_28_groupkfold_recheck_result.json"
OUT_TXT = DATA_DIR / "jra_search500_2026_08_28_groupkfold_recheck_report.txt"

WEIGHT_TIERS = [(100.0, 150), (25.0, 150), (6.0, 100), (1.0, 99)]  # 元スクリプトと完全一致
assert 1 + sum(n for _, n in WEIGHT_TIERS) == N_PATTERNS

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


gate = json.loads(GATE_JSON.read_text(encoding="utf-8"))
POOL_TRUE_PROD = gate["pool_true_prod"]
phase1_survivors = [c for c in gate["candidates"]
                    if gate["gate_results"][c]["box4"]["gate_pass"]]
POOL = POOL_TRUE_PROD + phase1_survivors
log(f"POOL({len(POOL)}本、元スクリプトと同一): {POOL}")

data = jra_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
log(f"レース数: {len(races)}  日付: {data['dates'][0]}〜{data['dates'][-1]}({len(data['dates'])}日)")

priors_all = JS.make_priors([r["df"] for r in races])
NAMES = JS.ALL_SIGNALS_V4
mats_all = JS.signal_matrices(races, priors_all, NAMES, JS.CLASS_ORDINAL)


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


def equal_w(names_subset) -> np.ndarray:
    return wvec({n: 1.0 / len(names_subset) for n in names_subset})


rng = np.random.default_rng(SEED)
cols = [equal_w(POOL)]
for concentration, n in WEIGHT_TIERS:
    alpha = [concentration] * len(POOL)
    for _ in range(n):
        cols.append(wvec(dict(zip(POOL, rng.dirichlet(alpha)))))
W_POOL = np.column_stack(cols)
assert W_POOL.shape[1] == N_PATTERNS
log(f"Dirichletパターン数: {N_PATTERNS}(元スクリプトと同一seed/tiersのため決定的に一致するはず)")

orig = json.loads(ORIG_SEARCH_JSON.read_text(encoding="utf-8"))

results_by_box = {}
for BOX_N in BOX_NS:
    log("\n" + "=" * 72)
    log(f"box_n={BOX_N}")
    log("=" * 72)

    winner = json.loads((DATA_DIR / WINNER_FILES[BOX_N]).read_text(encoding="utf-8"))
    W_CURRENT = wvec(winner["weights"])

    ev = JE.Evaluator(races, actual, box_n=BOX_N)
    current_picks = JE.score_picks(mats_all, W_CURRENT, BOX_N)
    r_current = ev.evaluate(current_picks)
    log(f"現行本番(pattern{winner['pattern_id']})  複勝+ワイド={r_current['model']:.2f}%  "
        f"市場差={r_current['excess']:+.2f}pt")

    def predict_fn(train_idx, test_idx, W_POOL=W_POOL):
        st_train, rt_train = ev.settler.returns_for(
            [JE.score_picks([mats_all[i] for i in train_idx], W_POOL[:, j], BOX_N)
             for j in range(N_PATTERNS)][0:0])  # placeholder not used; real calc below
        return None  # unreachable, replaced below

    # 500パターン x train_idx でargmax選択 → test_idxへ適用(元スクリプトのfit_fnと同じ
    # 「レースごとの決済結果を先に全パターン分計算しておき、idxで絞ってcost_weighted_rateを
    # 取る」設計を踏襲。全レース分のpicksを毎回作り直すのではなく、事前計算したS/A行列から
    # train_idx/test_idxそれぞれのスコアだけ計算する軽量版。
    all_picks_full = [JE.score_picks(mats_all, W_POOL[:, j], BOX_N) for j in range(N_PATTERNS)]
    all_st_full, all_rt_full = [], []
    for p in all_picks_full:
        s, r = ev.settler.returns_for(p)
        all_st_full.append(s)
        all_rt_full.append(r)

    def predict_fn(train_idx, test_idx):
        vals = np.array([JE.cost_weighted_rate(all_st_full[j], all_rt_full[j], idx=train_idx)
                         for j in range(N_PATTERNS)])
        best = int(np.argmax(vals))
        test_picks = [all_picks_full[best][i] for i in test_idx]
        return test_picks, best

    gkf_oof = ev.group_kfold_oof_generic(predict_fn, n_folds=N_FOLDS, seed=SEED)
    log(f"\n[group_kfold_oof_generic, {N_FOLDS}分割] "
        f"複勝+ワイド={gkf_oof['model']:.2f}%  市場差={gkf_oof['excess']:+.2f}pt")
    log(f"  fold毎の選択パターンのユニーク数: {gkf_oof['fold_argmax_unique']}/{gkf_oof['n_folds']}"
        f"  選択パターン列: {gkf_oof['fold_argmax_choices']}")

    orig_lobo = orig["results_by_box"][str(BOX_N)]["nested_lobo_oof"]
    log(f"  (参考)元のLOBO OOF: 市場差={orig_lobo['excess']:+.2f}pt  "
        f"ユニーク数={orig_lobo['n_unique_patterns']}/{orig_lobo['n_folds']}")

    opt = JE.selection_optimism(ev, mats_all, W_POOL, n_rep=200, seed=2862)
    edge_ratio = opt["true_edge_pt"] / opt["true_edge_sd"] if opt["true_edge_sd"] else 0.0
    decision = "ADOPT_CANDIDATE" if edge_ratio >= DECISION_GATE_RATIO else "REJECTED"
    log(f"\n[選択バイアス診断](元スクリプトと同一、W_POOL決定的に一致のため数値も一致するはず)")
    log(f"  true_edge_pt={opt['true_edge_pt']:+.2f} (sd {opt['true_edge_sd']:.2f})  "
        f"比={edge_ratio:.3f}  判定={decision}")

    boot_gkf_vs_current = ev.block_bootstrap_diff(gkf_oof["picks"], current_picks, seed=41)
    boot_gkf_vs_market = ev.block_bootstrap(gkf_oof["picks"], n=2000, seed=31)
    log(f"\n  group_kfold OOF vs 現行本番: 差分95%CI=[{boot_gkf_vs_current['lo']:+.2f},"
        f"{boot_gkf_vs_current['hi']:+.2f}] mean={boot_gkf_vs_current['mean']:+.2f}pt")
    log(f"  group_kfold OOF 絶対回収率95%CI: [{boot_gkf_vs_market['lo']:.2f},"
        f"{boot_gkf_vs_market['hi']:.2f}]%(参考値、市場超過ゲートには使わない)")

    results_by_box[BOX_N] = {
        "current_model": r_current,
        "group_kfold_oof": {k: v for k, v in gkf_oof.items() if k != "picks"},
        "orig_lobo_oof_reference": orig_lobo,
        "selection_optimism": opt,
        "decision_gate_ratio": edge_ratio,
        "decision": decision,
        "bootstrap_gkf_vs_current": boot_gkf_vs_current,
        "bootstrap_gkf_vs_market_absolute": boot_gkf_vs_market,
    }

OUT_JSON.write_text(json.dumps({
    "pool": POOL, "n_patterns": N_PATTERNS, "n_folds": N_FOLDS, "seed": SEED,
    "results_by_box": results_by_box,
}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
log(f"\n保存: {OUT_JSON}")
log(f"保存: {OUT_TXT}")
