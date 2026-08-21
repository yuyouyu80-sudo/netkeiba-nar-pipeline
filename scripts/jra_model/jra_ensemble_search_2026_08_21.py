# -*- coding: utf-8 -*-
"""Front2: box5/4/3のアンサンブル(重みブレンド)+時系列検証(2026-08-21新設)。

Front1(jra_search500_2026_08_21_v3signals.py)は30シグナルプールでbox5/4/3を独立に500パターン
探索したが、全てREJECTED(選択バイアス診断の採否ゲート未達)だった。しかしNested LOBO OOFの
fold毎選択パターンがユニーク2〜4/36と強く退化しており、これは「単一パターンのin-sample選択が
不安定」であることを示唆する。本スクリプトは同一の500パターン母集団(同一SEED)から、単一
ベストパターンではなく多様性を考慮した上位8パターンのブレンド(jra_ensemble.py)を作ることで、
選択の不安定さ(分散)を緩和できるか検証する。

比較する4候補: (a) 単一ベストパターン(Front1と同じ選択方式)、(b) 多様性フィルタ付き上位8本
ブレンド(top_n=8, max_cos_sim=0.9、グリッド探索はしない)、(c) 現行モデル、(d) 30本等重み。
評価は3点セット: Nested LOBO OOF(退化検知込み、両手法とも)/選択バイアス診断(4候補を
W_small行列として既存selection_optimism()にそのまま渡す、新規関数は書かない)/
chronological_oof(時系列walk-forward、追加ロバスト性チェックのみ・主判定に使わない)。
採否の主判定はブロックブートストラップの差分CI(アンサンブルOOF picks - 現行モデル)。

出力: jra_ensemble_search_2026_08_21_result.json / _report.txt (data/jra_pipeline、
git管理下、研究ログ)。本番重みへの反映は専門家レビュー(ゲート1・ゲート2)を経てユーザーが
判断する。本スクリプト自体は本番重みを書き換えない。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
sys.path.insert(0, str(LIB_DIR))
import jra_backtest as JB  # noqa: E402
import jra_dataset  # noqa: E402
import jra_ensemble as JEN  # noqa: E402
import jra_eval as JE  # noqa: E402
import jra_signals as JS  # noqa: E402

N_PATTERNS = 500
SEED = 2851  # Front1(jra_search500_2026_08_21_v3signals.py)と同一シード・同一母集団を再生成
BOX_NS = (5, 4, 3)
WINNER_FILES = {5: "winner_v3.json", 4: "winner_box4.json", 3: "winner_box3.json"}
TOP_N, MAX_COS_SIM = 8, 0.9  # グリッド探索はしない(多重比較を増やさないため1組に固定)

WEIGHT_TIERS = [
    (100.0, 150),
    (25.0, 150),
    (6.0, 100),
    (1.0, 99),
]
assert 1 + sum(n for _, n in WEIGHT_TIERS) == N_PATTERNS

OUT_JSON = DATA_DIR / "jra_ensemble_search_2026_08_21_result.json"
OUT_TXT = DATA_DIR / "jra_ensemble_search_2026_08_21_report.txt"

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


data = jra_dataset.load(rebuild=True)
races, actual = data["races"], data["actual"]
NAMES = JS.ALL_SIGNALS
log(f"レース数: {len(races)}  日付: {data['dates'][0]}〜{data['dates'][-1]}({len(data['dates'])}日)")
log(f"探索プール({len(NAMES)}シグナル)。アンサンブル: top_n={TOP_N}, max_cos_sim={MAX_COS_SIM}")

priors_all = JS.make_priors([r["df"] for r in races])
mats_all = JS.signal_matrices(races, priors_all, NAMES, JS.CLASS_ORDINAL)


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


def equal_w() -> np.ndarray:
    return wvec({n: 1.0 / len(NAMES) for n in NAMES})


rng = np.random.default_rng(SEED)
cols = [equal_w()]
for concentration, n in WEIGHT_TIERS:
    alpha = [concentration] * len(NAMES)
    for _ in range(n):
        cols.append(wvec(dict(zip(NAMES, rng.dirichlet(alpha)))))
W_POOL = np.column_stack(cols)
assert W_POOL.shape[1] == N_PATTERNS
W_EQUAL = equal_w()

results_by_box = {}

for BOX_N in BOX_NS:
    log("\n" + "=" * 72)
    log(f"box_n={BOX_N}")
    log("=" * 72)

    winner = json.loads((DATA_DIR / WINNER_FILES[BOX_N]).read_text(encoding="utf-8"))
    W_CURRENT = wvec(winner["weights"])

    ev = JE.Evaluator(races, actual, box_n=BOX_N)
    mkt_picks = JE.market_picks(races, BOX_N)
    mkt = ev.evaluate(mkt_picks)
    current_picks = JE.score_picks(mats_all, W_CURRENT, BOX_N)
    r_current = ev.evaluate(current_picks)
    equal_picks = JE.score_picks(mats_all, W_EQUAL, BOX_N)
    log(f"市場  複勝+ワイド={mkt['model']:.2f}%")
    log(f"現行モデル  複勝+ワイド={r_current['model']:.2f}%  市場差={r_current['excess']:+.2f}pt")

    all_picks = [JE.score_picks(mats_all, W_POOL[:, j], BOX_N) for j in range(N_PATTERNS)]
    all_st, all_rt = [], []
    for p in all_picks:
        s, r = ev.settler.returns_for(p)
        all_st.append(s)
        all_rt.append(r)
    full_vals = np.array([JE.cost_weighted_rate(all_st[j], all_rt[j]) for j in range(N_PATTERNS)])
    best_full = int(np.argmax(full_vals))
    w_single_best_full = W_POOL[:, best_full]

    diverse_full = JEN.select_diverse_topn(W_POOL, full_vals, top_n=TOP_N, max_cos_sim=MAX_COS_SIM)
    w_blend_full = JEN.blend_weights(W_POOL, diverse_full)
    blend_picks_full = JE.score_picks(mats_all, w_blend_full, BOX_N)
    r_blend_full = ev.evaluate(blend_picks_full)
    log(f"\n[全{len(races)}レースでのin-sample単一ベスト] pattern#{best_full}  "
        f"複勝+ワイド={full_vals[best_full]:.2f}%(市場差={full_vals[best_full] - mkt['model']:+.2f}pt)")
    log(f"[全{len(races)}レースでのin-sampleアンサンブルブレンド] "
        f"選択パターン{len(diverse_full)}本{diverse_full}  "
        f"複勝+ワイド={r_blend_full['model']:.2f}%(市場差={r_blend_full['excess']:+.2f}pt)")

    def fit_fn_single(train_idx, all_st=all_st, all_rt=all_rt):
        vals = np.array([JE.cost_weighted_rate(all_st[j], all_rt[j], idx=train_idx) for j in range(N_PATTERNS)])
        best = int(np.argmax(vals))
        return W_POOL[:, best], best

    def score_fn(train_idx, all_st=all_st, all_rt=all_rt):
        vals = np.array([JE.cost_weighted_rate(all_st[j], all_rt[j], idx=train_idx) for j in range(N_PATTERNS)])
        return W_POOL, vals

    fit_fn_ensemble = JEN.ensemble_fit_fn_factory(NAMES, score_fn, top_n=TOP_N, max_cos_sim=MAX_COS_SIM)

    oof_single = ev.lobo_oof(fit_fn_single, mats_all)
    oof_ensemble = ev.lobo_oof(fit_fn_ensemble, mats_all)
    log(f"\n[Nested LOBO OOF・単一ベスト] 複勝+ワイド={oof_single['model']:.2f}%  "
        f"市場差={oof_single['excess']:+.2f}pt  "
        f"ユニークpattern {oof_single['n_unique_patterns']}/{oof_single['n_folds']}")
    log(f"[Nested LOBO OOF・アンサンブルブレンド] 複勝+ワイド={oof_ensemble['model']:.2f}%  "
        f"市場差={oof_ensemble['excess']:+.2f}pt  "
        f"ユニーク選択集合 {oof_ensemble['n_unique_patterns']}/{oof_ensemble['n_folds']}")

    # 選択バイアス診断: 4候補(単一ベスト・ブレンド・現行・等重み)をW_small行列として
    # 既存selection_optimism()にそのまま渡す(新規統計手法は導入しない)。
    W_small = np.column_stack([w_single_best_full, w_blend_full, W_CURRENT, W_EQUAL])
    opt = JE.selection_optimism(ev, mats_all, W_small, n_rep=200, seed=2871)
    log(f"\n[選択バイアス診断(4候補: 単一ベスト/ブレンド/現行/等重み)] ブロック半分割×200反復:")
    log(f"  選抜側(見た側)の平均      : {opt['selected_side']:.1f}%")
    log(f"  その候補の未使用側での成績 : {opt['unseen_side']:.1f}%")
    log(f"  未使用側の4候補平均       : {opt['unseen_all_mean']:.1f}%")
    log(f"  選ぶことの真の価値         : {opt['true_edge_pt']:+.2f}pt (sd {opt['true_edge_sd']:.2f})")

    # 時系列walk-forward(追加ロバスト性チェックのみ、主判定には使わない)。
    coof_single = ev.chronological_oof(fit_fn_single, mats_all, min_train_blocks=3)
    coof_ensemble = ev.chronological_oof(fit_fn_ensemble, mats_all, min_train_blocks=3)
    log(f"\n[時系列walk-forward(参考・ロバスト性チェック専用)] "
        f"単一ベスト: 複勝+ワイド={coof_single['model']:.2f}%  市場差={coof_single['excess']:+.2f}pt  "
        f"({coof_single['n_folds']}fold、対象{len(coof_single['tested_race_idx'])}レース)")
    log(f"  アンサンブルブレンド: 複勝+ワイド={coof_ensemble['model']:.2f}%  "
        f"市場差={coof_ensemble['excess']:+.2f}pt  "
        f"({coof_ensemble['n_folds']}fold、対象{len(coof_ensemble['tested_race_idx'])}レース)")
    chrono_lobo_agree = (coof_ensemble["excess"] > 0) == (oof_ensemble["excess"] > 0)
    log(f"  → LOBO(ランダムholdout)とwalk-forward(時系列holdout)の市場差の符号一致: "
        f"{'YES' if chrono_lobo_agree else 'NO(時間方向のリークの可能性を要確認)'}")

    # 主判定: アンサンブルOOF picks が 現行モデル/単一ベストOOF を統計的に上回るか。
    boot_ens_vs_current = ev.block_bootstrap_diff(oof_ensemble["picks"], current_picks, seed=41)
    boot_ens_vs_single = ev.block_bootstrap_diff(oof_ensemble["picks"], oof_single["picks"], seed=43)
    boot_ens_vs_market = ev.block_bootstrap(oof_ensemble["picks"], n=2000, seed=31)
    log(f"\n[主判定: アンサンブルNested LOBO OOFのブロックブートストラップCI、n=2000]")
    log(f"  市場比 95%CI[{boot_ens_vs_market['lo']:.1f}, {boot_ens_vs_market['hi']:.1f}]")
    log(f"  現行モデル比の差 95%CI[{boot_ens_vs_current['lo']:+.2f}, {boot_ens_vs_current['hi']:+.2f}]pt"
        "(下限が0を超える場合のみ、統計的に現行を上回ったと言える。採否判定の本命指標)")
    log(f"  単一ベストOOF比の差 95%CI[{boot_ens_vs_single['lo']:+.2f}, {boot_ens_vs_single['hi']:+.2f}]pt"
        "(下限が0を超える場合のみ、アンサンブルが単一パターン選択より統計的に優れると言える)")

    decision = "ADOPT_CANDIDATE" if boot_ens_vs_current["lo"] > 0 else "REJECTED"
    log(f"\n採否ゲート(現行モデル比95%CI下限>0): {decision}")

    results_by_box[BOX_N] = {
        "market": mkt["model"],
        "current_model": {"model": r_current["model"], "excess": r_current["excess"]},
        "single_best_full_population": {
            "pattern_index": best_full, "model": float(full_vals[best_full]),
            "excess": float(full_vals[best_full] - mkt["model"]),
        },
        "ensemble_blend_full_population": {
            "selected_patterns": diverse_full, "n_selected": len(diverse_full),
            "model": r_blend_full["model"], "excess": r_blend_full["excess"],
        },
        "nested_lobo_oof_single": {
            "model": oof_single["model"], "excess": oof_single["excess"],
            "n_unique_patterns": oof_single["n_unique_patterns"], "n_folds": oof_single["n_folds"],
        },
        "nested_lobo_oof_ensemble": {
            "model": oof_ensemble["model"], "excess": oof_ensemble["excess"],
            "n_unique_patterns": oof_ensemble["n_unique_patterns"], "n_folds": oof_ensemble["n_folds"],
        },
        "selection_optimism_4candidates": opt,
        "chronological_oof_single": {
            "model": coof_single["model"], "excess": coof_single["excess"],
            "n_folds": coof_single["n_folds"], "n_tested_races": int(len(coof_single["tested_race_idx"])),
        },
        "chronological_oof_ensemble": {
            "model": coof_ensemble["model"], "excess": coof_ensemble["excess"],
            "n_folds": coof_ensemble["n_folds"], "n_tested_races": int(len(coof_ensemble["tested_race_idx"])),
            "agrees_with_lobo_sign": bool(chrono_lobo_agree),
        },
        "bootstrap_ensemble_vs_market": boot_ens_vs_market,
        "bootstrap_ensemble_vs_current": boot_ens_vs_current,
        "bootstrap_ensemble_vs_single_best_oof": boot_ens_vs_single,
        "decision": decision,
    }

log("\n" + "=" * 72)
log("まとめ")
log("=" * 72)
for BOX_N in BOX_NS:
    r = results_by_box[BOX_N]
    bc = r["bootstrap_ensemble_vs_current"]
    bs = r["bootstrap_ensemble_vs_single_best_oof"]
    log(f"box_n={BOX_N}: 単一ベストOOF市場差={r['nested_lobo_oof_single']['excess']:+.2f}pt"
        f"({r['nested_lobo_oof_single']['n_unique_patterns']}/{r['nested_lobo_oof_single']['n_folds']})  "
        f"アンサンブルOOF市場差={r['nested_lobo_oof_ensemble']['excess']:+.2f}pt  "
        f"現行モデル比CI=[{bc['lo']:+.2f},{bc['hi']:+.2f}]pt  "
        f"単一ベスト比CI=[{bs['lo']:+.2f},{bs['hi']:+.2f}]pt  判定={r['decision']}")

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps({
    "n_races": len(races), "dates": data["dates"], "pool": NAMES, "n_patterns": N_PATTERNS,
    "seed": SEED, "top_n": TOP_N, "max_cos_sim": MAX_COS_SIM,
    "results_by_box": results_by_box,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
