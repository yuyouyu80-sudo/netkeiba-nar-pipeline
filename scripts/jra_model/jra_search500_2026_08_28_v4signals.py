# -*- coding: utf-8 -*-
"""Phase2: 組み合わせ探索(2026-08-28、JRA Stage2: 全ファクター統合・高確信度選抜計画の
JRA移植)。scripts/jra_model/jra_search500_2026_08_21_v3signals.pyと同一の探索手続き
(WEIGHT_TIERS等重み近傍集中サンプリング・Nested LOBO OOF・selection_optimism診断・
DECISION_GATE_RATIO=2.0)を、jra_signal_gate_v4_2026_08_28.pyのPhase1生存候補に適用する。

POOL = LEGACY_SIGNALS(10、現行本番と同一)+ Phase1生存候補(corner_transition_gapのみ、
box4でG1 PASSだがbox5/3は不一致という留保つき)。既存 jra_signals.py / jra_eval.py /
jra_dataset.py / jra_backtest.py はすべて無改造で参照するのみ。
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
SEED = 2862  # 新規シード(既存の2033/2044/2820/2821/2822/2851/2861とは別系列)
BOX_NS = (4, 5, 3)
WINNER_FILES = {5: "winner_v3.json", 4: "winner_box4.json", 3: "winner_box3.json"}
DECISION_GATE_RATIO = 2.0

GATE_JSON = DATA_DIR / "jra_signal_gate_v4_2026_08_28_result.json"
OUT_JSON = DATA_DIR / "jra_search500_2026_08_28_v4signals_result.json"
OUT_TXT = DATA_DIR / "jra_search500_2026_08_28_v4signals_report.txt"

WEIGHT_TIERS = [
    (100.0, 150),
    (25.0, 150),
    (6.0, 100),
    (1.0, 99),
]
assert 1 + sum(n for _, n in WEIGHT_TIERS) == N_PATTERNS

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


gate = json.loads(GATE_JSON.read_text(encoding="utf-8"))
POOL_TRUE_PROD = gate["pool_true_prod"]
phase1_survivors = [c for c in gate["candidates"]
                    if gate["gate_results"][c]["box4"]["gate_pass"]]
log(f"POOL_TRUE_PROD({len(POOL_TRUE_PROD)}本): {POOL_TRUE_PROD}")
log(f"Phase1 box4 G1通過({len(phase1_survivors)}本): {phase1_survivors}")
POOL = POOL_TRUE_PROD + phase1_survivors
log(f"探索対象プール({len(POOL)}本): {POOL}")

data = jra_dataset.load(rebuild=True)
races, actual = data["races"], data["actual"]
log(f"レース数: {len(races)}  日付: {data['dates'][0]}〜{data['dates'][-1]}({len(data['dates'])}日)"
    f"  頭数: {sum(len(r['df']) for r in races)}")

priors_all = JS.make_priors([r["df"] for r in races])
NAMES = JS.ALL_SIGNALS_V4  # 列順の名前空間(既存ALL_SIGNALSは不使用、無改造)
mats_all = JS.signal_matrices(races, priors_all, NAMES, JS.CLASS_ORDINAL)


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


def equal_w(names_subset) -> np.ndarray:
    d = {n: 1.0 / len(names_subset) for n in names_subset}
    return wvec(d)


rng = np.random.default_rng(SEED)
cols = [equal_w(POOL)]  # pattern#0固定: 厳密等重み
for concentration, n in WEIGHT_TIERS:
    alpha = [concentration] * len(POOL)
    for _ in range(n):
        cols.append(wvec(dict(zip(POOL, rng.dirichlet(alpha)))))
W_POOL = np.column_stack(cols)
assert W_POOL.shape[1] == N_PATTERNS

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
    log(f"上位{BOX_N}人気BOX(市場)  複勝+ワイド={mkt['model']:.2f}%")

    current_picks = JE.score_picks(mats_all, W_CURRENT, BOX_N)
    r_current = ev.evaluate(current_picks)
    log(f"現行モデル(pattern{winner['pattern_id']}, {len(races)}レースで直接評価)  "
        f"複勝+ワイド={r_current['model']:.2f}%  市場差={r_current['excess']:+.2f}pt")

    equal_picks = JE.score_picks(mats_all, equal_w(POOL), BOX_N)
    r_equal = ev.evaluate(equal_picks)
    log(f"{len(POOL)}本等重み(参考)  複勝+ワイド={r_equal['model']:.2f}%  市場差={r_equal['excess']:+.2f}pt")

    all_picks = [JE.score_picks(mats_all, W_POOL[:, j], BOX_N) for j in range(N_PATTERNS)]
    all_st, all_rt = [], []
    for p in all_picks:
        s, r = ev.settler.returns_for(p)
        all_st.append(s)
        all_rt.append(r)
    full_vals = np.array([JE.cost_weighted_rate(all_st[j], all_rt[j]) for j in range(N_PATTERNS)])
    best_full = int(np.argmax(full_vals))
    log(f"\n[全{len(races)}レースで最良の1パターン] pattern#{best_full}  "
        f"複勝+ワイド={full_vals[best_full]:.2f}%(市場差={full_vals[best_full] - mkt['model']:+.2f}pt)"
        "  ※in-sample最適化の楽観を含む、統計的検定には使わない")
    top_w = {n: float(w) for n, w in zip(NAMES, W_POOL[:, best_full]) if w > 0.005}
    log(f"  重み内訳(0.5%以上): {json.dumps(top_w, ensure_ascii=False)}")
    v4_share = sum(w for n, w in zip(NAMES, W_POOL[:, best_full]) if n in phase1_survivors)
    log(f"  V4生存候補の合計重みシェア: {v4_share:.3f} ({v4_share * 100:.1f}%)")

    def fit_fn(train_idx, all_st=all_st, all_rt=all_rt):
        vals = np.array([JE.cost_weighted_rate(all_st[j], all_rt[j], idx=train_idx) for j in range(N_PATTERNS)])
        best = int(np.argmax(vals))
        return W_POOL[:, best], best

    nested_oof = ev.lobo_oof(fit_fn, mats_all)
    n_unique = nested_oof["n_unique_patterns"]
    n_folds = nested_oof["n_folds"]
    degenerate = n_unique == 1
    log(f"\n[Nested LOBO OOF] {n_folds}ブロック(開催日×競馬場)で{N_PATTERNS}パターン探索という"
        f"手続き全体を交差検証: 複勝+ワイド={nested_oof['model']:.2f}%  市場差={nested_oof['excess']:+.2f}pt")
    log(f"  fold毎の選択パターンのユニーク数: {n_unique}/{n_folds}"
        + ("  ※全fold同一パターン=LOBO退化。この数値はin-sample評価として扱い、"
           "統計的検定には使わない" if degenerate else ""))

    opt = JE.selection_optimism(ev, mats_all, W_POOL, n_rep=200, seed=2862)
    log(f"\n[選択バイアス診断] ブロック半分割×200反復:")
    log(f"  選抜側(見た側)の平均      : {opt['selected_side']:.1f}%")
    log(f"  その候補の未使用側での成績 : {opt['unseen_side']:.1f}%")
    log(f"  未使用側の{N_PATTERNS}パターン平均       : {opt['unseen_all_mean']:.1f}%")
    log(f"  楽観バイアス               : {opt['optimism_pt']:+.1f}pt")
    log(f"  選ぶことの真の価値         : {opt['true_edge_pt']:+.2f}pt (sd {opt['true_edge_sd']:.2f})")
    log(f"  未使用側で{N_PATTERNS}パターン平均を上回る確率 : {opt['win_rate'] * 100:.0f}%")

    edge_ratio = opt["true_edge_pt"] / opt["true_edge_sd"] if opt["true_edge_sd"] else 0.0
    decision = "ADOPT_CANDIDATE" if edge_ratio >= DECISION_GATE_RATIO else "REJECTED"
    log(f"\n採否ゲート(true_edge_pt/true_edge_sd >= {DECISION_GATE_RATIO}): "
        f"{edge_ratio:.3f}  → {decision}"
        + ("  ※LOBO退化のためNested LOBO OOFではなく本診断のみで判定" if degenerate else ""))

    boot_vs_market = ev.block_bootstrap(nested_oof["picks"], n=2000, seed=31)
    boot_oof_vs_current = ev.block_bootstrap_diff(nested_oof["picks"], current_picks, seed=41)
    boot_oof_vs_equal = ev.block_bootstrap_diff(nested_oof["picks"], equal_picks, seed=43)
    log(f"\n[Nested LOBO OOF(誠実なheld-out結果)のブートストラップCI、n=2000]")
    log(f"  市場比 95%CI[{boot_vs_market['lo']:.1f}, {boot_vs_market['hi']:.1f}](複勝+ワイド%水準)")
    log(f"  現行モデル比の差 95%CI[{boot_oof_vs_current['lo']:+.2f}, {boot_oof_vs_current['hi']:+.2f}]pt"
        "(下限が0を超える場合のみ、統計的に現行を上回ったと言える。これが採否判定の本命指標)")
    log(f"  {len(POOL)}本等重み比の差 95%CI[{boot_oof_vs_equal['lo']:+.2f}, {boot_oof_vs_equal['hi']:+.2f}]pt")

    current_full_tbl = ev.full_table(current_picks)
    LOW_SAMPLE_HITS = 10
    low_sample_bets = current_full_tbl.loc[
        current_full_tbl["hit_races"] < LOW_SAMPLE_HITS, "bet_type"].tolist()
    log(f"\n[採用案(現行モデル)の全券種内訳、{len(races)}レース]\n{current_full_tbl.to_string(index=False)}")
    if low_sample_bets:
        log(f"  ※的中数{LOW_SAMPLE_HITS}未満(参考値): {low_sample_bets}")

    rejected_candidate_full_tbl = ev.full_table(nested_oof["picks"])
    log(f"\n[参考・不採用: {N_PATTERNS}パターン探索モデル(Nested LOBO OOF picks)の全券種内訳]\n"
        f"{rejected_candidate_full_tbl.to_string(index=False)}")

    results_by_box[BOX_N] = {
        "pattern_id_current": winner["pattern_id"],
        "pool": POOL,
        "n_patterns": N_PATTERNS,
        "market": mkt["model"],
        "current_model": {"model": r_current["model"], "excess": r_current["excess"]},
        "equal_weight_pool": {"model": r_equal["model"], "excess": r_equal["excess"]},
        "best_full_population": {
            "pattern_index": best_full, "model": float(full_vals[best_full]),
            "excess": float(full_vals[best_full] - mkt["model"]), "weights": top_w,
            "v4_survivor_weight_share": float(v4_share),
            "note": "in-sample最適化の楽観を含む。統計的検定には使わない。",
        },
        "nested_lobo_oof": {
            "model": nested_oof["model"], "excess": nested_oof["excess"],
            "n_unique_patterns": n_unique, "n_folds": n_folds, "degenerate": degenerate,
        },
        "selection_optimism": opt,
        "decision_gate_ratio": edge_ratio,
        "decision": decision,
        "bootstrap_oof_vs_market": boot_vs_market,
        "bootstrap_oof_vs_current": boot_oof_vs_current,
        "bootstrap_oof_vs_equal": boot_oof_vs_equal,
        "full_table_current_model": {
            "low_sample_bet_types": low_sample_bets,
            "low_sample_threshold_hits": LOW_SAMPLE_HITS,
            "rows": current_full_tbl.to_dict(orient="records"),
        },
        "full_table_rejected_candidate_oof_picks": {
            "rows": rejected_candidate_full_tbl.to_dict(orient="records"),
        },
    }

log("\n" + "=" * 72)
log("まとめ")
log("=" * 72)
for BOX_N in BOX_NS:
    r = results_by_box[BOX_N]
    boot_c = r["bootstrap_oof_vs_current"]
    log(f"box_n={BOX_N}: 現行モデル市場差={r['current_model']['excess']:+.2f}pt  "
        f"{len(POOL)}本等重み市場差={r['equal_weight_pool']['excess']:+.2f}pt  "
        f"全data探索(in-sample、参考値)市場差={r['best_full_population']['excess']:+.2f}pt  "
        f"Nested LOBO OOF市場差={r['nested_lobo_oof']['excess']:+.2f}pt"
        f"(ユニークpattern {r['nested_lobo_oof']['n_unique_patterns']}/{r['nested_lobo_oof']['n_folds']})  "
        f"選ぶことの真の価値={r['selection_optimism']['true_edge_pt']:+.2f}pt"
        f"(sd{r['selection_optimism']['true_edge_sd']:.2f})  "
        f"現行モデル比CI=[{boot_c['lo']:+.2f}, {boot_c['hi']:+.2f}]pt  判定={r['decision']}")

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps({
    "n_races": len(races), "dates": data["dates"], "n_blocks": len(set(
        f'{r["kaisai_date"]}_{r["racecourse"]}' for r in races)),
    "pool_true_prod": POOL_TRUE_PROD, "phase1_survivors": phase1_survivors, "pool": POOL,
    "n_patterns": N_PATTERNS, "seed": SEED, "weight_tiers": WEIGHT_TIERS,
    "decision_gate_ratio_threshold": DECISION_GATE_RATIO,
    "results_by_box": results_by_box,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
