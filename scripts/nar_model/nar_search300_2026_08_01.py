# -*- coding: utf-8 -*-
"""ユーザー依頼(2026-08-01): 地方競馬の取得データ増加(7/23-7/31、253レース)を受け、
2026-07-29のnar_search300_2026_07_29.py(126レース、box_n=4/3のみ)と同じ手法・同じ
乱数シードで300パターンを生成し直し、box_n=5(予想5頭)を新たに加えた3サイズで再評価する。

box_n=5はこれまでbox4の重みを流用していたが(既存設計)、今回のユーザー依頼により
box5にも独立した最良パターンを持たせる。dead signal検出・探索対象プールはbox間で
共有(データそのものの充填率で決まるため box_n に依存しない)。

  1. 全253レースで300パターンから最良を1つ選ぶ(in-sample、参考値)。
  2. Nested LOBO(開催日×競馬場のブロックを1つ除いた残りだけで300パターンから最良を選び、
     除いたブロックで評価。全ブロックで繰り返しpooled集計)で「300パターン探索という
     手続き全体」を交差検証した誠実な汎化性能を推定する。
  3. 選択バイアス診断(ブロック半分割×200反復)で「選ぶ」こと自体の真の価値を測る。

出力: nar_search300_2026_08_01_result.json / _report.txt (scratchpad、研究ログ)。
本番重み(data/nar_pipeline/winner_box{5,4,3}_nar.json)への反映は本スクリプトの外で
別途行う。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nar_pipeline"
sys.path.insert(0, str(LIB_DIR))
import nar_backtest as NB  # noqa: E402
import nar_dataset  # noqa: E402
import nar_eval as NE  # noqa: E402
import nar_signals as NS  # noqa: E402

N_PATTERNS = 300
SEED = 2029  # 2026-07-29のnar_search300_2026_07_29.pyと同じシード(「同じ300パターン」を再現)
BOX_NS = (5, 4, 3)

OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
OUT_JSON = OUT_DIR / "nar_search300_2026_08_01_result.json"
OUT_TXT = OUT_DIR / "nar_search300_2026_08_01_report.txt"

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


data = nar_dataset.load(rebuild=True)
races, actual = data["races"], data["actual"]
priors_all = NS.make_priors(races)
dead = NS.detect_dead(races, priors_all)
alive_base = [n for n in NS.ALL_SIGNALS
              if n not in dead and n not in NS.CANDIDATE_SIGNALS and n not in NS.CANDIDATE_SIGNALS_V2
              and n not in NS.CANDIDATE_SIGNALS_V3]
new_v2_alive = [n for n in NS.CANDIDATE_SIGNALS_V2 if n not in dead]
POOL = alive_base + new_v2_alive  # 17シグナル(2026-07-29と同じ構成)

log(f"レース数: {len(races)}  日付: {data['dates']}  頭数: {sum(len(r['df']) for r in races)}")
log(f"死にシグナル({len(dead)}): {dead}")
log(f"探索対象プール(生存{len(POOL)}, 等重み廃止・差のある重みを探索): {POOL}")
log(f"パターン数: {N_PATTERNS}  乱数シード: {SEED}(2026-07-29と同一)")

NAMES = NS.ALL_SIGNALS
mats_all = NS.signal_matrices(races, priors_all, NAMES)


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


def equal_w(subset) -> np.ndarray:
    d = {n: 1.0 / len(subset) for n in subset}
    return wvec(d)


rng = np.random.default_rng(SEED)
W_POOL = np.column_stack([
    wvec(dict(zip(POOL, rng.dirichlet([1.0] * len(POOL))))) for _ in range(N_PATTERNS)
])

W_BASE_EQUAL = equal_w(POOL)  # box5/4/3共通の基準(現行box3/4のalive_signalsはPOOLと同一構成)

results_by_box = {}

for BOX_N in BOX_NS:
    log("\n" + "=" * 72)
    log(f"box_n={BOX_N}")
    log("=" * 72)
    ev = NE.Evaluator(races, actual, box_n=BOX_N)
    mkt_picks = NE.market_picks(races, BOX_N)
    mkt = ev.evaluate(mkt_picks)
    log(f"上位{BOX_N}人気BOX(市場)  複勝+ワイド={mkt['model']:.2f}%")

    base_picks = NE.score_picks(mats_all, W_BASE_EQUAL, BOX_N)
    r_base = ev.evaluate(base_picks)
    log(f"基準(生存{len(POOL)}シグナル等重み)  複勝+ワイド={r_base['model']:.2f}%  市場差={r_base['excess']:+.2f}pt")

    all_picks = [NE.score_picks(mats_all, W_POOL[:, j], BOX_N) for j in range(N_PATTERNS)]
    all_st, all_rt = [], []
    for p in all_picks:
        s, r = ev.settler.returns_for(p)
        all_st.append(s)
        all_rt.append(r)
    full_vals = np.array([NE.cost_weighted_rate(all_st[j], all_rt[j]) for j in range(N_PATTERNS)])
    best_full = int(np.argmax(full_vals))
    log(f"\n[全{len(races)}レースで最良の1パターン] pattern#{best_full}  "
        f"複勝+ワイド={full_vals[best_full]:.2f}%(市場差={full_vals[best_full] - mkt['model']:+.2f}pt)"
        "  ※学習データそのもので選んでいるため楽観的(in-sample)な数字である点に注意")
    top_w = {n: float(w) for n, w in zip(NAMES, W_POOL[:, best_full]) if w > 0.005}
    log(f"  重み内訳(0.5%以上): {json.dumps(top_w, ensure_ascii=False)}")

    def fit_fn(train_idx, all_st=all_st, all_rt=all_rt):
        vals = np.array([NE.cost_weighted_rate(all_st[j], all_rt[j], idx=train_idx) for j in range(N_PATTERNS)])
        return W_POOL[:, int(np.argmax(vals))]

    nested_oof = ev.lobo_oof(fit_fn, mats_all)
    log(f"\n[Nested LOBO OOF] 300パターン探索という手続き全体をブロックで交差検証: "
        f"複勝+ワイド={nested_oof['model']:.2f}%  市場差={nested_oof['excess']:+.2f}pt"
        "  ※これが実際に汎化する性能の誠実な推定値")

    opt = NE.selection_optimism(ev, mats_all, W_POOL, n_rep=200, seed=2027)
    log(f"\n[選択バイアス診断] ブロック半分割×200反復:")
    log(f"  選抜側(見た側)の平均      : {opt['selected_side']:.1f}%")
    log(f"  その候補の未使用側での成績 : {opt['unseen_side']:.1f}%")
    log(f"  未使用側の{N_PATTERNS}パターン平均       : {opt['unseen_all_mean']:.1f}%")
    log(f"  楽観バイアス               : {opt['optimism_pt']:+.1f}pt")
    log(f"  選ぶことの真の価値         : {opt['true_edge_pt']:+.2f}pt (sd {opt['true_edge_sd']:.2f})")
    log(f"  未使用側で{N_PATTERNS}パターン平均を上回る確率 : {opt['win_rate'] * 100:.0f}%")

    gate_nested_beats_base = nested_oof["excess"] > r_base["excess"]
    gate_nested_positive = nested_oof["excess"] > 0
    log(f"\n参考ゲート(ブロッキングではなく報告のみ):")
    log(f"  Nested LOBO市場差が現行(等重み)を上回るか: {'YES' if gate_nested_beats_base else 'NO'} "
        f"({nested_oof['excess']:+.2f} vs {r_base['excess']:+.2f})")
    log(f"  Nested LOBO市場差がプラスか              : {'YES' if gate_nested_positive else 'NO'} "
        f"({nested_oof['excess']:+.2f}pt)")

    final_picks = all_picks[best_full]
    boot = ev.block_bootstrap(final_picks, n=2000, seed=31)
    log(f"\n[全{len(races)}レース実測・本番候補重み] 複勝+ワイド={full_vals[best_full]:.2f}%  "
        f"95%CI[{boot['lo']:.1f}, {boot['hi']:.1f}]")

    results_by_box[BOX_N] = {
        "pool": POOL,
        "n_patterns": N_PATTERNS,
        "market": mkt["model"],
        "baseline_equal_weight": {"model": r_base["model"], "excess": r_base["excess"]},
        "best_full_population": {
            "pattern_index": best_full, "model": float(full_vals[best_full]),
            "excess": float(full_vals[best_full] - mkt["model"]), "weights": top_w,
            "note": f"全{len(races)}レースで選んだ重みなのでin-sample最適化の楽観を含む",
        },
        "nested_lobo_oof": {"model": nested_oof["model"], "excess": nested_oof["excess"]},
        "selection_optimism": opt,
        "gates_report_only": {
            "nested_beats_baseline": bool(gate_nested_beats_base),
            "nested_positive": bool(gate_nested_positive),
        },
        "bootstrap_full_population": boot,
    }

log("\n" + "=" * 72)
log("まとめ")
log("=" * 72)
for BOX_N in BOX_NS:
    r = results_by_box[BOX_N]
    log(f"box_n={BOX_N}: 基準(等重み)市場差={r['baseline_equal_weight']['excess']:+.2f}pt  "
        f"全data探索(in-sample)市場差={r['best_full_population']['excess']:+.2f}pt  "
        f"Nested LOBO OOF市場差={r['nested_lobo_oof']['excess']:+.2f}pt  "
        f"選ぶことの真の価値={r['selection_optimism']['true_edge_pt']:+.2f}pt"
        f"(sd{r['selection_optimism']['true_edge_sd']:.2f})")

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps({
    "n_races": len(races), "dates": data["dates"], "dead_signals": dead, "pool": POOL,
    "n_patterns": N_PATTERNS, "seed": SEED,
    "results_by_box": results_by_box,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
