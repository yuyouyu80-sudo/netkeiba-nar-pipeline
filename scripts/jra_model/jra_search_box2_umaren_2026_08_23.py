# -*- coding: utf-8 -*-
"""ユーザー依頼(2026-08-23): 「馬連版もnewspaperデータを使用して、より良いパターンを見つけて
ください」を受けた探索。市場アンカー型(Harville式・newspaper不使用)の馬連特化モデリングは
7ゲート中2/7のみ通過で不採用となったため、既存のbox5(pattern83)/box4(pattern19)/
box3(pattern95)と同じ枠組み — newspaperの全シグナル(jra_signals.py、25シグナル)を重み付き
合成して上位N頭を選ぶ「box」モデル — の box_n=2 版を新設し、目的関数を馬連に絞って重み探索する。

box5/4/3は現在も「複勝+ワイド」を目的関数に選ばれた重みであり、馬連専用に最適化されたことは
一度もない。box_nを2に絞ると馬連は「上位2頭を的中させる」という単勝的中と同程度の厳しさの
券種になる。

**この母集団(jra_dataset.py、211レース・12開催日)は既にbox5/4/3の複数回の重み探索・
Step1(市場アンカー型)で繰り返し使われてきた「使い切り」の母集団である**ことを明記した上で、
既存のNested LOBO OOF + 選択バイアス診断という検証済みの厳格な手法(NAR nar_search300由来、
jra_search_2026_08_12.py等で使用実績あり)にそのまま乗せて多重比較・過学習リスクを検出する。

box2は新設のため現行モデルとの比較は無く、「上位2人気BOX(市場)」の馬連回収率とだけ比較する。
本スクリプトはあくまで研究ログであり、predict_box2.py等の本番一式は作らない(ゲートを一定以上
クリアした場合に限り、別途ユーザーに提案する)。

出力: jra_search_box2_umaren_2026_08_23_result.json / _report.txt (data/jra_pipeline、研究ログ)。
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

N_PATTERNS = 1000  # 既存box5/4/3探索(jra_search_2026_08_12.py)の候補数に倣う
SEED = 2308  # 新規採番(既存探索のシードと衝突しないように)
BOX_N = 2
BETS = ["馬連"]

OUT_JSON = DATA_DIR / "jra_search_box2_umaren_2026_08_23_result.json"
OUT_TXT = DATA_DIR / "jra_search_box2_umaren_2026_08_23_report.txt"

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


data = jra_dataset.load(rebuild=True)
races, actual = data["races"], data["actual"]
NAMES = JS.ALL_SIGNALS  # 既存10 + 候補9(2026-08-11) + 候補6(2026-08-12) = 25シグナル
log(f"レース数: {len(races)}  日付: {data['dates']}  頭数: {sum(len(r['df']) for r in races)}")
log("※この母集団はbox5/4/3の複数回の重み探索・Step1(市場アンカー型)で既に繰り返し使われて"
    "いる「使い切り」の母集団である点に注意(多重比較・過学習リスクの土台情報)。")
log(f"探索対象プール({len(NAMES)}シグナル): {NAMES}")
log(f"目的関数: {BETS}  box_n={BOX_N}  "
    f"理論ブレークイーブン回収率={JE.breakeven_pct(BOX_N, bets=BETS):.2f}%")

dfs = [r["df"] for r in races]
priors_fresh = JS.make_priors(dfs)
log(f"priors再計算: {len(priors_fresh)}キー({len(races)}レース全体から)")

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

ev = JE.Evaluator(races, actual, box_n=BOX_N)
mkt_picks = JE.market_picks(races, BOX_N)
mkt = ev.evaluate(mkt_picks, bets=BETS)
log(f"\n上位{BOX_N}人気BOX(市場)  馬連={mkt['model']:.2f}%")

equal_picks = JE.score_picks(mats_all, W_EQUAL, BOX_N)
r_equal = ev.evaluate(equal_picks, bets=BETS)
log(f"等重み({len(NAMES)}シグナル、参考)  馬連={r_equal['model']:.2f}%  市場差={r_equal['excess']:+.2f}pt")

all_picks = [JE.score_picks(mats_all, W_POOL[:, j], BOX_N) for j in range(N_PATTERNS)]
all_st, all_rt = [], []
for p in all_picks:
    s, r = ev.settler.returns_for(p)
    all_st.append(s)
    all_rt.append(r)
full_vals = np.array([JE.cost_weighted_rate(all_st[j], all_rt[j], bets=BETS) for j in range(N_PATTERNS)])
best_full = int(np.argmax(full_vals))

equal_pct = float((full_vals < r_equal["model"]).mean() * 100)
log(f"\n[早期診断] {N_PATTERNS}パターンのin-sampleスコア分布における等重みの位置: {equal_pct:.1f}パーセンタイル")

log(f"\n[全{len(races)}レースで最良の1パターン] pattern#{best_full}  "
    f"馬連={full_vals[best_full]:.2f}%(市場差={full_vals[best_full] - mkt['model']:+.2f}pt)"
    "  ※学習データそのもので選んでいるため楽観的(in-sample)な数字である点に注意")
top_w = {n: float(w) for n, w in zip(NAMES, W_POOL[:, best_full]) if w > 0.005}
log(f"  重み内訳(0.5%以上): {json.dumps(top_w, ensure_ascii=False)}")


def fit_fn(train_idx, all_st=all_st, all_rt=all_rt):
    vals = np.array([JE.cost_weighted_rate(all_st[j], all_rt[j], bets=BETS, idx=train_idx)
                     for j in range(N_PATTERNS)])
    best = int(np.argmax(vals))
    return W_POOL[:, best], best


nested_oof = ev.lobo_oof(fit_fn, mats_all, bets=BETS)
log(f"\n[Nested LOBO OOF] {N_PATTERNS}パターン探索という手続き全体をブロックで交差検証: "
    f"馬連={nested_oof['model']:.2f}%  市場差={nested_oof['excess']:+.2f}pt"
    "  ※これが実際に汎化する性能の誠実な推定値")

opt = JE.selection_optimism(ev, mats_all, W_POOL, n_rep=200, seed=2308, bets=BETS)
log("\n[選択バイアス診断] ブロック半分割×200反復:")
log(f"  選抜側(見た側)の平均      : {opt['selected_side']:.1f}%")
log(f"  その候補の未使用側での成績 : {opt['unseen_side']:.1f}%")
log(f"  未使用側の{N_PATTERNS}パターン平均       : {opt['unseen_all_mean']:.1f}%")
log(f"  楽観バイアス               : {opt['optimism_pt']:+.1f}pt")
log(f"  選ぶことの真の価値         : {opt['true_edge_pt']:+.2f}pt (sd {opt['true_edge_sd']:.2f})")
log(f"  未使用側で{N_PATTERNS}パターン平均を上回る確率 : {opt['win_rate'] * 100:.0f}%")

# 単勝(参考指標、box_nが2なので単勝の的中は「1位を当てる」に近い厳しさ)。
equal_table = ev.full_table(equal_picks)
best_table = ev.full_table(all_picks[best_full])
eq_tansho = equal_table[equal_table["bet_type"] == "単勝"].iloc[0]
best_tansho = best_table[best_table["bet_type"] == "単勝"].iloc[0]
log(f"\n[単勝チェック(参考、in-sample)] 等重み={eq_tansho['return_rate_pct']:.1f}%"
    f"(的中{eq_tansho['hit_rate_pct']:.1f}%) → 最良候補={best_tansho['return_rate_pct']:.1f}%"
    f"(的中{best_tansho['hit_rate_pct']:.1f}%)")

gate1_nested_positive = nested_oof["excess"] > 0
gate2_true_edge_positive = opt["true_edge_pt"] > 0 and opt["win_rate"] > 0.5
final_picks = all_picks[best_full]
boot = ev.block_bootstrap(final_picks, bets=BETS, n=2000, seed=2308)
gate3_boot_ci_lo_above_breakeven = boot["lo"] > 100.0
log(f"\n[全{len(races)}レース実測・本番候補重み] 馬連={full_vals[best_full]:.2f}%  "
    f"95%CI[{boot['lo']:.1f}, {boot['hi']:.1f}]  ※in-sample楽観を含むため参考")

log("\n採否ゲート(参考、記述的):")
log(f"  1. Nested LOBO OOF馬連市場差がプラスか      : {'YES' if gate1_nested_positive else 'NO'} "
    f"({nested_oof['excess']:+.2f}pt)")
log(f"  2. 選択バイアス診断のtrue_edge_ptが有意にプラスか: {'YES' if gate2_true_edge_positive else 'NO'} "
    f"(true_edge={opt['true_edge_pt']:+.2f}pt, win_rate={opt['win_rate']*100:.0f}%)")
log(f"  3. 全レースブートストラップCI下限が100%を超えるか(参考): "
    f"{'YES' if gate3_boot_ci_lo_above_breakeven else 'NO'} (CI下限={boot['lo']:.1f}%)")

n_gates_passed = sum([gate1_nested_positive, gate2_true_edge_positive, gate3_boot_ci_lo_above_breakeven])
decision = (
    f"採用検討可(参考ゲート{n_gates_passed}/3通過、本番化は別途提案が必要)"
    if gate1_nested_positive and gate2_true_edge_positive
    else f"不採用(参考ゲート{n_gates_passed}/3のみ通過、主判定のNested LOBO OOFが基準未達)"
)
log(f"\n=> 判定: {decision}")

candidate_weight_share = sum(w for n, w in zip(NAMES, W_POOL[:, best_full]) if n in JS.CANDIDATE_SIGNALS)
candidate_v2_weight_share = sum(
    w for n, w in zip(NAMES, W_POOL[:, best_full]) if n in JS.CANDIDATE_SIGNALS_V2)
log(f"\n[候補シグナルの寄与] 最良パターンにおける候補9(2026-08-11)合計重み: "
    f"{candidate_weight_share:.3f} ({candidate_weight_share*100:.1f}%)  "
    f"候補6(2026-08-12, surf_ketto/jockey系)合計重み: {candidate_v2_weight_share:.3f} "
    f"({candidate_v2_weight_share*100:.1f}%)")

result = {
    "n_races": len(races), "dates": data["dates"], "pool": NAMES,
    "n_patterns": N_PATTERNS, "seed": SEED, "box_n": BOX_N, "bets": BETS,
    "breakeven_pct": JE.breakeven_pct(BOX_N, bets=BETS),
    "note_population_reuse": "この母集団はbox5/4/3・Step1の複数回の探索で既に使い切られている",
    "market": mkt["model"],
    "equal_weight": {"model": r_equal["model"], "excess": r_equal["excess"],
                     "percentile_among_1000": equal_pct},
    "best_full_population": {
        "pattern_index": best_full, "model": float(full_vals[best_full]),
        "excess": float(full_vals[best_full] - mkt["model"]), "weights": top_w,
        "candidate_signal_weight_share": float(candidate_weight_share),
        "candidate_v2_signal_weight_share": float(candidate_v2_weight_share),
        "note": f"全{len(races)}レースで選んだ重みなのでin-sample最適化の楽観を含む",
    },
    "nested_lobo_oof": {"model": nested_oof["model"], "excess": nested_oof["excess"]},
    "selection_optimism": opt,
    "tansho_check_in_sample": {
        "equal_return_rate_pct": float(eq_tansho["return_rate_pct"]),
        "equal_hit_rate_pct": float(eq_tansho["hit_rate_pct"]),
        "best_return_rate_pct": float(best_tansho["return_rate_pct"]),
        "best_hit_rate_pct": float(best_tansho["hit_rate_pct"]),
    },
    "bootstrap_full_population": boot,
    "gates": {
        "nested_positive": bool(gate1_nested_positive),
        "true_edge_positive": bool(gate2_true_edge_positive),
        "boot_ci_lo_above_breakeven": bool(gate3_boot_ci_lo_above_breakeven),
        "n_gates_passed": int(n_gates_passed),
        "n_gates_total": 3,
    },
    "decision": decision,
}

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
