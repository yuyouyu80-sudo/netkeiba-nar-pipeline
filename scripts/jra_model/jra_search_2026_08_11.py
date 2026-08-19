# -*- coding: utf-8 -*-
"""ユーザー依頼(2026-08-11): JRA通常戦(新馬・未勝利を除く)の収集データが105レース(pattern83
採用時)から177レースに増えたことを受け、既存10シグナル+候補9シグナル(course/concerned/
interval/kinryo/nige/margin/timediff/agari/holdtime、未使用データ調査から選定)の計19シグナルで
box_n=5/4/3それぞれ独立に重みを再探索する。

NAR側(scripts/nar_model/nar_search300_2026_08_01.py)で確立した手法をそのまま踏襲する:
  1. 全レースで1000パターンから最良を1つ選ぶ(in-sample、参考値。楽観バイアスを含む)。
  2. Nested LOBO(開催日×競馬場のブロックを1つ除いた残りだけで1000パターンから最良を選び、
     除いたブロックで評価。全ブロックで繰り返しpooled集計)で「1000パターン探索という
     手続き全体」を交差検証した誠実な汎化性能を推定する。
  3. 選択バイアス診断(ブロック半分割×200反復)で「選ぶ」こと自体の真の価値を測る。

JRAとNARのアーキテクチャ差分: box_n=5/4/3はNAR同様それぞれ独立モデル(predict.py=pattern83/
winner_v3.json、predict_box4.py=pattern19/winner_box4.json、predict_box3.py=pattern95/
winner_box3.json)であることが実装時に判明した(当初「1本のランキングを切り詰める」設計だと
誤認していたが、box4/box3が既に独立の重みファイルを持っていることをコード確認して訂正)。
よって本スクリプトもNARのnar_search300と同じく、box_n=5/4/3それぞれ独立にEvaluatorを作り
独立に最良パターンを探索する。

出力: jra_search_2026_08_11_result.json / _report.txt (data/jra_pipeline、研究ログ)。
本番重み(data/jra_pipeline/winner_v3.json 等)への反映は本スクリプトの外で別途行う。
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
import jra_eval as JE  # noqa: E402
import jra_signals as JS  # noqa: E402

N_PATTERNS = 1000  # NARが候補シグナル追加時(17→22シグナル相当)に1000パターンを使った前例に倣う
SEED = 2033
BOX_NS = (5, 4, 3)
WINNER_FILES = {5: "winner_v3.json", 4: "winner_box4.json", 3: "winner_box3.json"}

OUT_JSON = DATA_DIR / "jra_search_2026_08_11_result.json"
OUT_TXT = DATA_DIR / "jra_search_2026_08_11_report.txt"

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


data = jra_dataset.load(rebuild=True)
races, actual = data["races"], data["actual"]
NAMES = JS.ALL_SIGNALS  # 既存10 + 候補9 = 19シグナル
log(f"レース数: {len(races)}  日付: {data['dates']}  頭数: {sum(len(r['df']) for r in races)}")
log(f"探索対象プール({len(NAMES)}シグナル): {NAMES}")

# priorsは現在の全レースから再計算する(pattern83等の記録時点のpriorsとは別物。
# 探索側にのみ使い、data/jra_pipeline/winner_*.jsonの既存記録は変更しない)。
dfs = [r["df"] for r in races]
priors_fresh = JS.make_priors(dfs)
log(f"priors再計算: {len(priors_fresh)}キー(177レース全体から)")

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

results_by_box = {}

for BOX_N in BOX_NS:
    log("\n" + "=" * 72)
    log(f"box_n={BOX_N}")
    log("=" * 72)

    winner = json.loads((DATA_DIR / WINNER_FILES[BOX_N]).read_text(encoding="utf-8"))
    W_CURRENT = wvec(winner["weights"])  # 候補シグナル分は0(現行モデルは既存10シグナルのみ)

    ev = JE.Evaluator(races, actual, box_n=BOX_N)
    mkt_picks = JE.market_picks(races, BOX_N)
    mkt = ev.evaluate(mkt_picks)
    log(f"上位{BOX_N}人気BOX(市場)  複勝+ワイド={mkt['model']:.2f}%")

    current_picks = JE.score_picks(mats_all, W_CURRENT, BOX_N)
    r_current = ev.evaluate(current_picks)
    log(f"現行モデル(pattern{winner['pattern_id']}, 177レースで直接評価)  "
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

    # 早期の安価な診断: 現行モデル・等重みが1000パターン中どのパーセンタイルに相当するか
    current_pct = float((full_vals < r_current["model"]).mean() * 100)
    equal_pct = float((full_vals < r_equal["model"]).mean() * 100)
    log(f"\n[早期診断] 1000パターンのin-sampleスコア分布における現行モデルの位置: "
        f"{current_pct:.1f}パーセンタイル / 等重み: {equal_pct:.1f}パーセンタイル")

    log(f"\n[全{len(races)}レースで最良の1パターン] pattern#{best_full}  "
        f"複勝+ワイド={full_vals[best_full]:.2f}%(市場差={full_vals[best_full] - mkt['model']:+.2f}pt)"
        "  ※学習データそのもので選んでいるため楽観的(in-sample)な数字である点に注意")
    top_w = {n: float(w) for n, w in zip(NAMES, W_POOL[:, best_full]) if w > 0.005}
    log(f"  重み内訳(0.5%以上): {json.dumps(top_w, ensure_ascii=False)}")

    def fit_fn(train_idx, all_st=all_st, all_rt=all_rt):
        vals = np.array([JE.cost_weighted_rate(all_st[j], all_rt[j], idx=train_idx) for j in range(N_PATTERNS)])
        return W_POOL[:, int(np.argmax(vals))]

    nested_oof = ev.lobo_oof(fit_fn, mats_all)
    log(f"\n[Nested LOBO OOF] {N_PATTERNS}パターン探索という手続き全体をブロックで交差検証: "
        f"複勝+ワイド={nested_oof['model']:.2f}%  市場差={nested_oof['excess']:+.2f}pt"
        "  ※これが実際に汎化する性能の誠実な推定値")

    opt = JE.selection_optimism(ev, mats_all, W_POOL, n_rep=200, seed=2027)
    log(f"\n[選択バイアス診断] ブロック半分割×200反復:")
    log(f"  選抜側(見た側)の平均      : {opt['selected_side']:.1f}%")
    log(f"  その候補の未使用側での成績 : {opt['unseen_side']:.1f}%")
    log(f"  未使用側の{N_PATTERNS}パターン平均       : {opt['unseen_all_mean']:.1f}%")
    log(f"  楽観バイアス               : {opt['optimism_pt']:+.1f}pt")
    log(f"  選ぶことの真の価値         : {opt['true_edge_pt']:+.2f}pt (sd {opt['true_edge_sd']:.2f})")
    log(f"  未使用側で{N_PATTERNS}パターン平均を上回る確率 : {opt['win_rate'] * 100:.0f}%")

    # 単勝(勝率の直接指標)の非劣化チェック: 最良パターンをfull dataで採用したとして、
    # 現行モデル比で単勝が悪化していないか確認する(採否ゲートの一部)。
    current_table = ev.full_table(current_picks)
    best_table = ev.full_table(all_picks[best_full])
    cur_tansho = current_table[current_table["bet_type"] == "単勝"].iloc[0]
    best_tansho = best_table[best_table["bet_type"] == "単勝"].iloc[0]
    log(f"\n[単勝チェック(参考、in-sample)] 現行={cur_tansho['return_rate_pct']:.1f}%"
        f"(的中{cur_tansho['hit_rate_pct']:.1f}%) → 最良候補={best_tansho['return_rate_pct']:.1f}%"
        f"(的中{best_tansho['hit_rate_pct']:.1f}%)")

    gate_nested_beats_current = nested_oof["excess"] > r_current["excess"]
    gate_nested_positive = nested_oof["excess"] > 0
    log(f"\n採否ゲート(参考、最終判断は複数指標を総合して行う):")
    log(f"  Nested LOBO市場差が現行モデルを上回るか: {'YES' if gate_nested_beats_current else 'NO'} "
        f"({nested_oof['excess']:+.2f} vs {r_current['excess']:+.2f})")
    log(f"  Nested LOBO市場差がプラスか            : {'YES' if gate_nested_positive else 'NO'} "
        f"({nested_oof['excess']:+.2f}pt)")

    final_picks = all_picks[best_full]
    boot = ev.block_bootstrap(final_picks, n=2000, seed=31)
    log(f"\n[全{len(races)}レース実測・本番候補重み] 複勝+ワイド={full_vals[best_full]:.2f}%  "
        f"95%CI[{boot['lo']:.1f}, {boot['hi']:.1f}]")

    # 候補シグナル別寄与(最良パターンにおいて候補シグナルの重みがどの程度乗ったか)
    candidate_weight_share = sum(w for n, w in zip(NAMES, W_POOL[:, best_full]) if n in JS.CANDIDATE_SIGNALS)
    log(f"\n[候補シグナルの寄与] 最良パターンにおける候補9シグナル合計重み: {candidate_weight_share:.3f} "
        f"(全体の{candidate_weight_share*100:.1f}%)")

    results_by_box[BOX_N] = {
        "pattern_id_current": winner["pattern_id"],
        "n_signals": len(NAMES),
        "market": mkt["model"],
        "current_model": {"model": r_current["model"], "excess": r_current["excess"],
                          "percentile_among_1000": current_pct},
        "equal_weight": {"model": r_equal["model"], "excess": r_equal["excess"],
                         "percentile_among_1000": equal_pct},
        "best_full_population": {
            "pattern_index": best_full, "model": float(full_vals[best_full]),
            "excess": float(full_vals[best_full] - mkt["model"]), "weights": top_w,
            "candidate_signal_weight_share": float(candidate_weight_share),
            "note": f"全{len(races)}レースで選んだ重みなのでin-sample最適化の楽観を含む",
        },
        "nested_lobo_oof": {"model": nested_oof["model"], "excess": nested_oof["excess"]},
        "selection_optimism": opt,
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
        f"(sd{r['selection_optimism']['true_edge_sd']:.2f})")

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps({
    "n_races": len(races), "dates": data["dates"], "pool": NAMES,
    "n_patterns": N_PATTERNS, "seed": SEED, "priors_fresh": priors_fresh,
    "results_by_box": results_by_box,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
