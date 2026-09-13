# -*- coding: utf-8 -*-
"""ユーザー依頼(2026-09-11)「BOX3の予想モデルを変更したい、現時点で最良の予想モデルを探して
ほしい」に対応する、box_n=3専用の重み再探索。jra_search_2026_08_12.pyと同じ方法論
(1000(→本スクリプトは信号数が多いため2000)パターンのDirichlet探索 + Nested LOBO OOF +
選択バイアス診断)を、box_n=3だけに絞り、かつ現在利用可能な全信号プール
(LEGACY10 + CANDIDATE9(08-11) + V2(6, 08-12) + V3(5, 08-21) + V4(11, mark/corner, 08-28) +
V5(4, 血統sire適性, 08-30) = 45信号)・現時点の全レースデータ(jra_dataset.load(rebuild=True)、
2026-08-12時点の105/177レースより大幅に増えている)で行う。

**重要な前提(2026-09-06の専門家レビューで確定済み、[[project_jra_improvement_review_2026_09_06]]
参照)**: 現状のブロック数では3pt程度の改善を検出するのに必要な検出力が足りない
(jra_power_precheck.py実測: 現状ブロック数のMDE≈17pt、3pt検出には約1,200〜1,400ブロック相当が
必要)。そのためPhase 3(モデリング変更)自体が凍結中であり、本スクリプトで見つかる「最良」は
統計的に有意な改善の保証ではなく、**参考値(reference value)** として扱うことをユーザーと
事前に合意済み。過去のbox3/box4重み探索(2026-08-11、2026-08-12拡張版含む)はいずれも
等重み・現行案を統計的に上回れず不採用となっている実績があるため、本スクリプトの結果も
同じ結論(現行/等重みと有意差なし)になる可能性が高いことを踏まえてレポートする。

出力: data/jra_pipeline/jra_search_box3_2026_09_11_result.json / _report.txt
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

N_PATTERNS = 2000  # 信号数が45と多いため、jra_search_2026_08_12.py(25信号・1000)より増やす
SEED = 2911
BOX_N = 3
WINNER_FILE = "winner_box3.json"

OUT_JSON = DATA_DIR / "jra_search_box3_2026_09_11_result.json"
OUT_TXT = DATA_DIR / "jra_search_box3_2026_09_11_report.txt"

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


data = jra_dataset.load(rebuild=True)
races, actual = data["races"], data["actual"]
NAMES = JS.ALL_SIGNALS_V5 + JS.CANDIDATE_SIGNALS_V4  # 現時点で存在する全候補信号(45個)
log(f"レース数: {len(races)}  日付: {data['dates']}  頭数: {sum(len(r['df']) for r in races)}")
log(f"探索対象プール({len(NAMES)}信号): {NAMES}")

dfs = [r["df"] for r in races]
priors_fresh = JS.make_priors(dfs)
log(f"priors再計算: {len(priors_fresh)}キー(全{len(races)}レースから)")

mats_all = JS.signal_matrices(races, priors_fresh, NAMES, JS.CLASS_ORDINAL)


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

winner = json.loads((DATA_DIR / WINNER_FILE).read_text(encoding="utf-8"))
W_CURRENT = wvec(winner["weights"])  # 候補信号分は0(現行モデルはLEGACY10信号のみ)

ev = JE.Evaluator(races, actual, box_n=BOX_N)
mkt_picks = JE.market_picks(races, BOX_N)
mkt = ev.evaluate(mkt_picks)
log(f"\n上位{BOX_N}人気BOX(市場)  複勝+ワイド={mkt['model']:.2f}%")

current_picks = JE.score_picks(mats_all, W_CURRENT, BOX_N)
r_current = ev.evaluate(current_picks)
log(f"現行モデル(pattern{winner['pattern_id']}, 2026-08-12選定、全{len(races)}レースで直接評価)  "
    f"複勝+ワイド={r_current['model']:.2f}%  市場差={r_current['excess']:+.2f}pt")

equal_picks = JE.score_picks(mats_all, W_EQUAL, BOX_N)
r_equal = ev.evaluate(equal_picks)
log(f"等重み({len(NAMES)}信号、参考)  複勝+ワイド={r_equal['model']:.2f}%  "
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
    f"複勝+ワイド={nested_oof['model']:.2f}%  市場差={nested_oof['excess']:+.2f}pt"
    f"  (foldの選択パターン数: {nested_oof['n_unique_patterns']}/{nested_oof['n_folds']}、"
    "1に近いほど『全foldで同じパターンしか選ばれない=退化』の疑い)"
    "  ※これが実際に汎化する性能の誠実な推定値")

opt = JE.selection_optimism(ev, mats_all, W_POOL, n_rep=200, seed=2912)
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

# held-out(現行モデルのfit母集団に含まれない開催日)上でのmodel vs marketペア差分95%CI
held_out_blocks = JE.held_out_block_subset(winner["fitted_on"], races)
held_out_idx = np.where(np.isin(ev.blocks, held_out_blocks))[0]
boot_ho_best_vs_mkt = ev.block_bootstrap_diff(all_picks[best_full], mkt_picks, n=2000, seed=2913,
                                              block_subset=held_out_blocks)
boot_ho_best_vs_current = ev.block_bootstrap_diff(all_picks[best_full], current_picks, n=2000, seed=2914,
                                                   block_subset=held_out_blocks)
log(f"\n[held-out({len(held_out_blocks)}ブロック、現行モデルのfit日付を除く)での参考比較]")
log(f"  最良候補 vs 市場   : 差分95%CI=[{boot_ho_best_vs_mkt['lo']:+.1f}, {boot_ho_best_vs_mkt['hi']:+.1f}]pt"
    f"  n_blocks={boot_ho_best_vs_mkt['n_blocks']}")
log(f"  最良候補 vs 現行モデル: 差分95%CI=[{boot_ho_best_vs_current['lo']:+.1f}, "
    f"{boot_ho_best_vs_current['hi']:+.1f}]pt  n_blocks={boot_ho_best_vs_current['n_blocks']}")

gate_nested_beats_current = nested_oof["excess"] > r_current["excess"]
gate_nested_positive = nested_oof["excess"] > 0
log(f"\n採否ゲート(参考、検出力不足のため統計的な採否判断には使わない):")
log(f"  Nested LOBO市場差が現行モデルを上回るか: {'YES' if gate_nested_beats_current else 'NO'} "
    f"({nested_oof['excess']:+.2f} vs {r_current['excess']:+.2f})")
log(f"  Nested LOBO市場差がプラスか            : {'YES' if gate_nested_positive else 'NO'} "
    f"({nested_oof['excess']:+.2f}pt)")

final_picks = all_picks[best_full]
boot = ev.block_bootstrap(final_picks, n=2000, seed=2915)
log(f"\n[全{len(races)}レース実測・最良候補重み] 複勝+ワイド={full_vals[best_full]:.2f}%  "
    f"95%CI[{boot['lo']:.1f}, {boot['hi']:.1f}]")

signal_groups = {
    "LEGACY": JS.LEGACY_SIGNALS, "CANDIDATE_v1(08-11)": JS.CANDIDATE_SIGNALS,
    "CANDIDATE_v2(08-12)": JS.CANDIDATE_SIGNALS_V2, "CANDIDATE_v3(08-21)": JS.CANDIDATE_SIGNALS_V3,
    "CANDIDATE_v4(08-28,mark/corner)": JS.CANDIDATE_SIGNALS_V4,
    "CANDIDATE_v5(08-30,sire適性)": JS.CANDIDATE_SIGNALS_V5,
}
group_share = {
    label: float(sum(w for n, w in zip(NAMES, W_POOL[:, best_full]) if n in names))
    for label, names in signal_groups.items()
}
log(f"\n[信号グループ別寄与(最良パターン)]")
for label, share in group_share.items():
    log(f"  {label}: {share:.3f} ({share*100:.1f}%)")

result = {
    "n_races": len(races), "dates": data["dates"], "pool": NAMES,
    "n_patterns": N_PATTERNS, "seed": SEED, "priors_fresh": priors_fresh,
    "market": mkt["model"],
    "current_model": {"model": r_current["model"], "excess": r_current["excess"],
                      "percentile_among_pool": current_pct},
    "equal_weight": {"model": r_equal["model"], "excess": r_equal["excess"],
                     "percentile_among_pool": equal_pct},
    "best_full_population": {
        "pattern_index": best_full, "model": float(full_vals[best_full]),
        "excess": float(full_vals[best_full] - mkt["model"]), "weights": top_w,
        "signal_group_share": group_share,
        "note": f"全{len(races)}レースで選んだ重みなのでin-sample最適化の楽観を含む(参考値)",
    },
    "nested_lobo_oof": {"model": nested_oof["model"], "excess": nested_oof["excess"],
                       "n_unique_patterns": nested_oof["n_unique_patterns"],
                       "n_folds": nested_oof["n_folds"]},
    "selection_optimism": opt,
    "held_out_vs_market": boot_ho_best_vs_mkt,
    "held_out_vs_current": boot_ho_best_vs_current,
    "gates_report_only": {
        "nested_beats_current": bool(gate_nested_beats_current),
        "nested_positive": bool(gate_nested_positive),
    },
    "tansho_check_in_sample": {
        "current_return_rate_pct": float(cur_tansho["return_rate_pct"]),
        "current_hit_rate_pct": float(cur_tansho["hit_rate_pct"]),
        "best_return_rate_pct": float(best_tansho["return_rate_pct"]),
        "best_hit_rate_pct": float(best_tansho["hit_rate_pct"]),
    },
    "bootstrap_full_population": boot,
    "caveat": "2026-09-06専門家レビューでPhase3(モデリング変更)は検出力不足のため凍結中と"
              "判定済み(jra_power_precheck.py: 現状ブロック数のMDE約17pt、3pt検出には約"
              "1200-1400ブロック相当が必要)。本結果はユーザー合意の上での参考値であり、"
              "統計的に有意な改善の保証ではない。",
}

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
