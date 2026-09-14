# -*- coding: utf-8 -*-
"""box3/4/5の候補シグナルプールを初めて全統合(LEGACY10+V1-V4=41+V5=4+V6=1+
レーダー複合9(V7)=55本)して、Nested LOBO OOF + 選択バイアス診断にかける。

背景(ユーザー指示、2026-09-14): 「台帳再検証」作業を機に、詳細7カテゴリ(レーダー)の
9分割で新設した表示専用サブ軸(jra_radar_categories.DISPLAY_SUBCATEGORY_MAP)を
box3/4/5本番重み探索の候補シグナルとしても検証してほしいとの依頼を受けた。調査の結果、
2件の重要事実が判明した:
  1. レーダー複合9本は新規生データではなく、既存シグナル(LEGACY/V1-V4)をレース内で
     グループ化・再minmaxした合成値(例: radar_style_position = style+nige+
     corner4_position+corner4_gap+corner3_position+corner3_gapの合成)。
  2. CANDIDATE_SIGNALS_V4(予想印・コーナー展開、11本)が、2026-08-28の単独検証
     (ALL_SIGNALS_V4)以降、V5(血統)・V6(トラックバイアス)の統合探索チェーンに
     一度も合流していなかった(ALL_SIGNALS_V5/V6はALL_SIGNALS(LEGACY+V1+V2+V3)から
     分岐しておりV4を経由しない、jra_signals.py参照)。
ユーザーはAskUserQuestionで「V4も合流させ全55本を統合」を選択したため、本探索では
V4を含む全候補を初めて一つのプールとして統合し検証する。

手法はjra_search_track_bias_2026_09_13.py(血統・トラックバイアス検証)を完全踏襲する
(Dirichlet探索→in-sample最良→Nested LOBO OOF→選択バイアス診断→採否ゲート、
box_n=5/4/3それぞれ独立)。プールが30→55本(約1.8倍)に拡大したため、
N_PATTERNS(Dirichletサンプル数)も1000→2000に増量する。

過去の同種探索(v1〜v6・レーダー面積予想・33ラップ理論等)は例外なく市場超過を示せず
不採用となっている実績があり、本探索もREJECTが十分あり得る前提で実施する
(採否は統計的結果のみで判定し、結果を事前に決めない)。
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
import jra_backtest as JB  # noqa: E402,F401
import jra_dataset  # noqa: E402
import jra_eval as JE  # noqa: E402
import jra_radar_categories as RC  # noqa: E402
import jra_signals as JS  # noqa: E402

N_PATTERNS = 2000
SEED = 20914
BOX_NS = (5, 4, 3)
WINNER_FILES = {5: "winner_v3.json", 4: "winner_box4.json", 3: "winner_box3.json"}

OUT_JSON = DATA_DIR / "jra_search_radar_v7_2026_09_14_result.json"
OUT_TXT = DATA_DIR / "jra_search_radar_v7_2026_09_14_report.txt"

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


data = jra_dataset.load(rebuild=True)  # 前回検証(09-13)以降のデータ更新を確実に反映させる
races, actual = data["races"], data["actual"]
NAMES = JS.ALL_SIGNALS_V7  # LEGACY10+V1-V4(41)+V5(4)+V6(1)+レーダー複合V7(9) = 55シグナル
log(f"レース数: {len(races)}  日付: {data['dates']}  頭数: {sum(len(r['df']) for r in races)}")
log(f"探索対象プール({len(NAMES)}シグナル): {NAMES}")
log(f"  うちV4(印・コーナー展開、今回初めて統合チェーンに合流): {JS.CANDIDATE_SIGNALS_V4}")
log(f"  うちレーダー複合V7(今回新規・既存シグナルの再グルーピング): {JS.CANDIDATE_SIGNALS_V7}")

dfs = [r["df"] for r in races]
priors_fresh = JS.make_priors(dfs)
log(f"priors再計算: {len(priors_fresh)}キー")

mats_all = RC.signal_matrices_with_radar(races, priors_fresh, NAMES, JS.CLASS_ORDINAL)


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


def equal_w() -> np.ndarray:
    d = {n: 1.0 / len(NAMES) for n in NAMES}
    return wvec(d)


rng = np.random.default_rng(SEED)
W_POOL = np.column_stack([
    wvec(dict(zip(NAMES, rng.dirichlet([1.0] * len(NAMES))))) for _ in range(N_PATTERNS)
])
W_EQUAL = equal_w()

results_by_box = {}

for BOX_N in BOX_NS:
    log("\n" + "=" * 72)
    log(f"box_n={BOX_N}")
    log("=" * 72)

    winner = json.loads((DATA_DIR / WINNER_FILES[BOX_N]).read_text(encoding="utf-8"))
    W_CURRENT = wvec(winner["weights"])  # V4/V5/V6/V7分は0(現行モデルはLEGACY10のみ)

    ev = JE.Evaluator(races, actual, box_n=BOX_N)
    mkt_picks = JE.market_picks(races, BOX_N)
    mkt = ev.evaluate(mkt_picks)
    log(f"上位{BOX_N}人気BOX(市場)  複勝+ワイド={mkt['model']:.2f}%")

    current_picks = JE.score_picks(mats_all, W_CURRENT, BOX_N)
    r_current = ev.evaluate(current_picks)
    log(f"現行モデル(pattern{winner['pattern_id']})  "
        f"複勝+ワイド={r_current['model']:.2f}%  市場差={r_current['excess']:+.2f}pt")

    equal_picks = JE.score_picks(mats_all, W_EQUAL, BOX_N)
    r_equal = ev.evaluate(equal_picks)
    log(f"等重み({len(NAMES)}シグナル、参考)  複勝+ワイド={r_equal['model']:.2f}%  "
        f"市場差={r_equal['excess']:+.2f}pt")

    all_picks = [JE.score_picks(mats_all, W_POOL[:, j], BOX_N) for j in range(N_PATTERNS)]
    all_st, all_rt = [], []
    for p in all_picks:
        s, r = ev.settler.returns_for(p)
        all_st.append(s)
        all_rt.append(r)
    full_vals = np.array([JE.cost_weighted_rate(all_st[j], all_rt[j]) for j in range(N_PATTERNS)])
    best_full = int(np.argmax(full_vals))

    current_pct = float((full_vals < r_current["model"]).mean() * 100)
    equal_pct = float((full_vals < r_equal["model"]).mean() * 100)
    log(f"\n[早期診断] {N_PATTERNS}パターンのin-sampleスコア分布における現行モデルの位置: "
        f"{current_pct:.1f}パーセンタイル / 等重み: {equal_pct:.1f}パーセンタイル")

    log(f"\n[全{len(races)}レースで最良の1パターン] pattern#{best_full}  "
        f"複勝+ワイド={full_vals[best_full]:.2f}%(市場差={full_vals[best_full] - mkt['model']:+.2f}pt)"
        "  ※学習データそのもので選んでいるため楽観的(in-sample)な数字である点に注意")
    top_w = {n: float(w) for n, w in zip(NAMES, W_POOL[:, best_full]) if w > 0.005}
    log(f"  重み内訳(0.5%以上): {json.dumps(top_w, ensure_ascii=False)}")

    def fit_fn(train_idx, all_st=all_st, all_rt=all_rt):
        vals = np.array([JE.cost_weighted_rate(all_st[j], all_rt[j], idx=train_idx) for j in range(N_PATTERNS)])
        best = int(np.argmax(vals))
        return W_POOL[:, best], best

    nested_oof = ev.lobo_oof(fit_fn, mats_all)
    log(f"\n[Nested LOBO OOF] {N_PATTERNS}パターン探索という手続き全体をブロックで交差検証: "
        f"複勝+ワイド={nested_oof['model']:.2f}%  市場差={nested_oof['excess']:+.2f}pt")

    opt = JE.selection_optimism(ev, mats_all, W_POOL, n_rep=200, seed=2027)
    log(f"\n[選択バイアス診断] ブロック半分割×200反復:")
    log(f"  選抜側(見た側)の平均      : {opt['selected_side']:.1f}%")
    log(f"  その候補の未使用側での成績 : {opt['unseen_side']:.1f}%")
    log(f"  未使用側の{N_PATTERNS}パターン平均       : {opt['unseen_all_mean']:.1f}%")
    log(f"  楽観バイアス               : {opt['optimism_pt']:+.1f}pt")
    log(f"  選ぶことの真の価値         : {opt['true_edge_pt']:+.2f}pt (sd {opt['true_edge_sd']:.2f})")
    log(f"  未使用側で{N_PATTERNS}パターン平均を上回る確率 : {opt['win_rate'] * 100:.0f}%")

    current_table = ev.full_table(current_picks)
    best_table = ev.full_table(all_picks[best_full])
    cur_tansho = current_table[current_table["bet_type"] == "単勝"].iloc[0]
    best_tansho = best_table[best_table["bet_type"] == "単勝"].iloc[0]
    log(f"\n[単勝チェック(参考、in-sample)] 現行={cur_tansho['return_rate_pct']:.1f}%"
        f"(的中{cur_tansho['hit_rate_pct']:.1f}%) → 最良候補={best_tansho['return_rate_pct']:.1f}%"
        f"(的中{best_tansho['hit_rate_pct']:.1f}%)")

    gate_nested_beats_current = nested_oof["excess"] > r_current["excess"]
    gate_nested_positive = nested_oof["excess"] > 0
    gate_true_edge = opt["true_edge_pt"] / opt["true_edge_sd"] if opt["true_edge_sd"] else 0.0
    log(f"\n採否ゲート(true_edge/sd >= 2.0が採用の目安、過去の探索と同一基準):")
    log(f"  Nested LOBO市場差が現行モデルを上回るか: {'YES' if gate_nested_beats_current else 'NO'} "
        f"({nested_oof['excess']:+.2f} vs {r_current['excess']:+.2f})")
    log(f"  Nested LOBO市場差がプラスか            : {'YES' if gate_nested_positive else 'NO'} "
        f"({nested_oof['excess']:+.2f}pt)")
    log(f"  true_edge/sd比                          : {gate_true_edge:+.3f} "
        f"({'PASS' if gate_true_edge >= 2.0 else 'FAIL'})")

    final_picks = all_picks[best_full]
    boot = ev.block_bootstrap(final_picks, n=2000, seed=31)
    log(f"\n[全{len(races)}レース実測・本番候補重み] 複勝+ワイド={full_vals[best_full]:.2f}%  "
        f"95%CI[{boot['lo']:.1f}, {boot['hi']:.1f}]")

    def weight_share(names_subset):
        return sum(w for n, w in zip(NAMES, W_POOL[:, best_full]) if n in names_subset)

    v4_share = weight_share(JS.CANDIDATE_SIGNALS_V4)
    v5_share = weight_share(JS.CANDIDATE_SIGNALS_V5)
    v6_share = weight_share(JS.CANDIDATE_SIGNALS_V6)
    v7_share = weight_share(JS.CANDIDATE_SIGNALS_V7)
    legacy_share = weight_share(JS.LEGACY_SIGNALS)
    log(f"\n[群別の寄与(最良パターン)] LEGACY={legacy_share*100:.1f}%  "
        f"V4(印・コーナー、今回初合流)={v4_share*100:.1f}%  V5(血統)={v5_share*100:.1f}%  "
        f"V6(トラックバイアス)={v6_share*100:.1f}%  V7(レーダー複合、新規)={v7_share*100:.1f}%")
    log(f"  V7内訳:")
    for n in JS.CANDIDATE_SIGNALS_V7:
        idx = NAMES.index(n)
        log(f"    {n}: {W_POOL[idx, best_full]:.4f}")

    results_by_box[BOX_N] = {
        "pattern_id_current": winner["pattern_id"],
        "n_signals": len(NAMES),
        "market": mkt["model"],
        "current_model": {"model": r_current["model"], "excess": r_current["excess"],
                          "percentile_among_pool": current_pct},
        "equal_weight": {"model": r_equal["model"], "excess": r_equal["excess"],
                         "percentile_among_pool": equal_pct},
        "best_full_population": {
            "pattern_index": best_full, "model": float(full_vals[best_full]),
            "excess": float(full_vals[best_full] - mkt["model"]), "weights": top_w,
            "weight_share_by_wave": {"legacy": float(legacy_share), "v4": float(v4_share),
                                     "v5": float(v5_share), "v6": float(v6_share), "v7": float(v7_share)},
            "note": f"全{len(races)}レースで選んだ重みなのでin-sample最適化の楽観を含む",
        },
        "nested_lobo_oof": {"model": nested_oof["model"], "excess": nested_oof["excess"]},
        "selection_optimism": opt,
        "gates_report_only": {
            "nested_beats_current": bool(gate_nested_beats_current),
            "nested_positive": bool(gate_nested_positive),
            "true_edge_over_sd": float(gate_true_edge),
            "true_edge_gate_pass": bool(gate_true_edge >= 2.0),
        },
        "tansho_check_in_sample": {
            "current_return_rate_pct": float(cur_tansho["return_rate_pct"]),
            "current_hit_rate_pct": float(cur_tansho["hit_rate_pct"]),
            "best_return_rate_pct": float(best_tansho["return_rate_pct"]),
            "best_hit_rate_pct": float(best_tansho["hit_rate_pct"]),
        },
        "bootstrap_full_population": boot,
    }

log("\n" + "=" * 72)
log("まとめ")
log("=" * 72)
for BOX_N in BOX_NS:
    r = results_by_box[BOX_N]
    log(f"box_n={BOX_N} (現行pattern{r['pattern_id_current']}): "
        f"現行モデル市場差={r['current_model']['excess']:+.2f}pt  "
        f"等重み市場差={r['equal_weight']['excess']:+.2f}pt  "
        f"全data探索(in-sample)市場差={r['best_full_population']['excess']:+.2f}pt  "
        f"Nested LOBO OOF市場差={r['nested_lobo_oof']['excess']:+.2f}pt  "
        f"選ぶことの真の価値={r['selection_optimism']['true_edge_pt']:+.2f}pt"
        f"(sd{r['selection_optimism']['true_edge_sd']:.2f})  "
        f"true_edge/sd={r['gates_report_only']['true_edge_over_sd']:+.3f} "
        f"({'PASS' if r['gates_report_only']['true_edge_gate_pass'] else 'FAIL'})")

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps({
    "n_races": len(races), "dates": data["dates"], "pool": NAMES,
    "n_patterns": N_PATTERNS, "seed": SEED, "priors_fresh": priors_fresh,
    "results_by_box": results_by_box,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
