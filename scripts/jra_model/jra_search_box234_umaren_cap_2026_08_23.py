# -*- coding: utf-8 -*-
"""ユーザー依頼(2026-08-23続き): 「馬連newspaper、最良パターン探索についてですが、BOX2〜4で
探してください。払い戻し2万円以上のレースは除外して回収率の計算もお願いします。」

先行実施のjra_search_box2_umaren_2026_08_23.py(box_n=2のみ、キャップ無し)を拡張し、
  1. box_n = 2, 3, 4 の3サイズをそれぞれ独立に探索(box5/4/3と同じ「box_nごとに独立モデル」
     というJRAアーキテクチャを踏襲)。
  2. 払戻2万円以上のレースを除外して回収率を計算(主判定)。jra_eval.pyに追加した
     max_payout引数(実現値ベースのキャップ)を使う。市場ベンチマーク・等重み・in-sample
     最良パターン・Nested LOBO OOF・選択バイアス診断・ブートストラップの全てに同一ルールで
     適用する(=同一ルールでの横並び比較なので、2026-08-23の馬連Harvilleモデル検証で問題に
     なった「real vs simulated armの非対称性」は生じない。あちらは同一picksを2000通りの乱数列に
     当てはめる順列検定だったため実現値キャップが非対称に効いたが、本スクリプトは複数の戦略
     (市場/モデル/OOF)を同一の実現値キャップの下で横並び比較するだけなので対称)。
  参考として、キャップ無し(未加工)の数字も併記する。

母集団は既存のbox5/4/3・Step1・box2探索で複数回使われてきた「使い切り」の211レースのまま
(jra_dataset.py)。過学習リスクは前回のbox2結果(in-sample市場+277pt、Nested LOBO OOFでは
市場-26.5pt)で既に強く示唆されているため、今回もNested LOBO OOF + 選択バイアス診断を
主判定として重視する。

出力: jra_search_box234_umaren_cap_2026_08_23_result.json / _report.txt (data/jra_pipeline、研究ログ)。
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

N_PATTERNS = 1000
SEED = 2309  # box2単独版(2308)と衝突しないよう新規採番
BOX_NS = (2, 3, 4)
BETS = ["馬連"]
MAX_PAYOUT = 20000.0  # ユーザー依頼: 払戻2万円以上のレースは除外

OUT_JSON = DATA_DIR / "jra_search_box234_umaren_cap_2026_08_23_result.json"
OUT_TXT = DATA_DIR / "jra_search_box234_umaren_cap_2026_08_23_report.txt"

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


data = jra_dataset.load(rebuild=True)
races, actual = data["races"], data["actual"]
NAMES = JS.ALL_SIGNALS
log(f"レース数: {len(races)}  日付: {data['dates']}  頭数: {sum(len(r['df']) for r in races)}")
log("※この母集団はbox5/4/3・Step1・box2探索で既に繰り返し使われている「使い切り」の母集団。")
log(f"探索対象プール({len(NAMES)}シグナル): {NAMES}")
log(f"目的関数: {BETS}  払戻キャップ: {MAX_PAYOUT:.0f}円以上のレースを除外して回収率計算(主判定)")

dfs = [r["df"] for r in races]
priors_fresh = JS.make_priors(dfs)
log(f"priors再計算: {len(priors_fresh)}キー({len(races)}レース全体から)")

mats_all = JS.signal_matrices(races, priors_fresh, NAMES, JS.CLASS_ORDINAL)


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


def equal_w() -> np.ndarray:
    d = {n: 1.0 / len(NAMES) for n in NAMES}
    return wvec(d)


W_EQUAL = equal_w()
results_by_box = {}

for BOX_N in BOX_NS:
    log("\n" + "=" * 72)
    log(f"box_n={BOX_N}  理論ブレークイーブン回収率={JE.breakeven_pct(BOX_N, bets=BETS):.2f}%")
    log("=" * 72)

    rng = np.random.default_rng(SEED + BOX_N)  # box_nごとに異なるが再現可能なシード
    W_POOL = np.column_stack([
        wvec(dict(zip(NAMES, rng.dirichlet([1.0] * len(NAMES))))) for _ in range(N_PATTERNS)
    ])

    ev = JE.Evaluator(races, actual, box_n=BOX_N)
    mkt_picks = JE.market_picks(races, BOX_N)
    mkt_cap = ev.evaluate(mkt_picks, bets=BETS, max_payout=MAX_PAYOUT)
    mkt_nocap = ev.evaluate(mkt_picks, bets=BETS)
    log(f"上位{BOX_N}人気BOX(市場)  馬連(cap)={mkt_cap['model']:.2f}%  馬連(no-cap)={mkt_nocap['model']:.2f}%")

    equal_picks = JE.score_picks(mats_all, W_EQUAL, BOX_N)
    r_equal_cap = ev.evaluate(equal_picks, bets=BETS, max_payout=MAX_PAYOUT)
    r_equal_nocap = ev.evaluate(equal_picks, bets=BETS)
    log(f"等重み({len(NAMES)}シグナル、参考)  馬連(cap)={r_equal_cap['model']:.2f}%"
        f"(市場差{r_equal_cap['excess']:+.2f}pt)  馬連(no-cap)={r_equal_nocap['model']:.2f}%"
        f"(市場差{r_equal_nocap['excess']:+.2f}pt)")

    all_picks = [JE.score_picks(mats_all, W_POOL[:, j], BOX_N) for j in range(N_PATTERNS)]
    all_st, all_rt = [], []
    for p in all_picks:
        s, r = ev.settler.returns_for(p)
        all_st.append(s)
        all_rt.append(r)
    full_vals_cap = np.array([
        JE.cost_weighted_rate(all_st[j], all_rt[j], bets=BETS, max_payout=MAX_PAYOUT)
        for j in range(N_PATTERNS)])
    full_vals_nocap = np.array([
        JE.cost_weighted_rate(all_st[j], all_rt[j], bets=BETS) for j in range(N_PATTERNS)])
    best_full = int(np.argmax(full_vals_cap))  # 選定基準はキャップ後の値(主判定と一致させる)

    log(f"\n[全{len(races)}レースで最良の1パターン(選定基準=cap後)] pattern#{best_full}  "
        f"馬連(cap)={full_vals_cap[best_full]:.2f}%(市場差={full_vals_cap[best_full] - mkt_cap['model']:+.2f}pt)"
        f"  馬連(no-cap参考)={full_vals_nocap[best_full]:.2f}%"
        "  ※学習データそのもので選んでいるため楽観的(in-sample)な数字である点に注意")
    top_w = {n: float(w) for n, w in zip(NAMES, W_POOL[:, best_full]) if w > 0.005}
    log(f"  重み内訳(0.5%以上): {json.dumps(top_w, ensure_ascii=False)}")

    def fit_fn_cap(train_idx, all_st=all_st, all_rt=all_rt):
        vals = np.array([JE.cost_weighted_rate(all_st[j], all_rt[j], bets=BETS, idx=train_idx,
                                                max_payout=MAX_PAYOUT) for j in range(N_PATTERNS)])
        best = int(np.argmax(vals))
        return W_POOL[:, best], best

    def fit_fn_nocap(train_idx, all_st=all_st, all_rt=all_rt):
        vals = np.array([JE.cost_weighted_rate(all_st[j], all_rt[j], bets=BETS, idx=train_idx)
                         for j in range(N_PATTERNS)])
        best = int(np.argmax(vals))
        return W_POOL[:, best], best

    nested_oof_cap = ev.lobo_oof(fit_fn_cap, mats_all, bets=BETS, max_payout=MAX_PAYOUT)
    log(f"\n[Nested LOBO OOF・cap後(主判定)] 馬連={nested_oof_cap['model']:.2f}%  "
        f"市場差={nested_oof_cap['excess']:+.2f}pt")

    nested_oof_nocap = ev.lobo_oof(fit_fn_nocap, mats_all, bets=BETS)
    log(f"[Nested LOBO OOF・no-cap(参考)] 馬連={nested_oof_nocap['model']:.2f}%  "
        f"市場差={nested_oof_nocap['excess']:+.2f}pt")

    opt = JE.selection_optimism(ev, mats_all, W_POOL, n_rep=200, seed=SEED + BOX_N,
                                bets=BETS, max_payout=MAX_PAYOUT)
    log("\n[選択バイアス診断・cap後] ブロック半分割×200反復:")
    log(f"  選抜側(見た側)の平均      : {opt['selected_side']:.1f}%")
    log(f"  その候補の未使用側での成績 : {opt['unseen_side']:.1f}%")
    log(f"  未使用側の{N_PATTERNS}パターン平均       : {opt['unseen_all_mean']:.1f}%")
    log(f"  楽観バイアス               : {opt['optimism_pt']:+.1f}pt")
    log(f"  選ぶことの真の価値         : {opt['true_edge_pt']:+.2f}pt (sd {opt['true_edge_sd']:.2f})")
    log(f"  未使用側で{N_PATTERNS}パターン平均を上回る確率 : {opt['win_rate'] * 100:.0f}%")

    final_picks = all_picks[best_full]
    boot_cap = ev.block_bootstrap(final_picks, bets=BETS, n=2000, seed=SEED + BOX_N, max_payout=MAX_PAYOUT)
    boot_nocap = ev.block_bootstrap(final_picks, bets=BETS, n=2000, seed=SEED + BOX_N)
    log(f"\n[全{len(races)}レース実測・本番候補重み・cap後] 馬連={full_vals_cap[best_full]:.2f}%  "
        f"95%CI[{boot_cap['lo']:.1f}, {boot_cap['hi']:.1f}]  ※in-sample楽観を含むため参考")
    log(f"[同・no-cap(参考)] 馬連={full_vals_nocap[best_full]:.2f}%  "
        f"95%CI[{boot_nocap['lo']:.1f}, {boot_nocap['hi']:.1f}]")

    gate1_nested_positive = nested_oof_cap["excess"] > 0
    gate2_true_edge_positive = opt["true_edge_pt"] > 0 and opt["win_rate"] > 0.5
    gate3_boot_ci_lo_above_breakeven = boot_cap["lo"] > 100.0
    log("\n採否ゲート(参考、記述的、cap後の数字で判定):")
    log(f"  1. Nested LOBO OOF馬連市場差がプラスか      : {'YES' if gate1_nested_positive else 'NO'} "
        f"({nested_oof_cap['excess']:+.2f}pt)")
    log(f"  2. 選択バイアス診断のtrue_edge_ptが有意にプラスか: {'YES' if gate2_true_edge_positive else 'NO'} "
        f"(true_edge={opt['true_edge_pt']:+.2f}pt, win_rate={opt['win_rate']*100:.0f}%)")
    log(f"  3. 全レースブートストラップCI下限が100%を超えるか(参考): "
        f"{'YES' if gate3_boot_ci_lo_above_breakeven else 'NO'} (CI下限={boot_cap['lo']:.1f}%)")

    n_gates_passed = sum([gate1_nested_positive, gate2_true_edge_positive, gate3_boot_ci_lo_above_breakeven])
    decision = (
        f"採用検討可(参考ゲート{n_gates_passed}/3通過、本番化は別途提案が必要)"
        if gate1_nested_positive and gate2_true_edge_positive
        else f"不採用(参考ゲート{n_gates_passed}/3のみ通過、主判定のNested LOBO OOFが基準未達)"
    )
    log(f"\n=> box_n={BOX_N} 判定: {decision}")

    candidate_weight_share = sum(w for n, w in zip(NAMES, W_POOL[:, best_full]) if n in JS.CANDIDATE_SIGNALS)
    candidate_v2_weight_share = sum(
        w for n, w in zip(NAMES, W_POOL[:, best_full]) if n in JS.CANDIDATE_SIGNALS_V2)

    results_by_box[BOX_N] = {
        "n_signals": len(NAMES),
        "max_payout_cap": MAX_PAYOUT,
        "market": {"cap": mkt_cap["model"], "nocap": mkt_nocap["model"]},
        "equal_weight": {
            "cap": {"model": r_equal_cap["model"], "excess": r_equal_cap["excess"]},
            "nocap": {"model": r_equal_nocap["model"], "excess": r_equal_nocap["excess"]},
        },
        "best_full_population": {
            "pattern_index": best_full,
            "cap": {"model": float(full_vals_cap[best_full]),
                   "excess": float(full_vals_cap[best_full] - mkt_cap["model"])},
            "nocap": {"model": float(full_vals_nocap[best_full]),
                     "excess": float(full_vals_nocap[best_full] - mkt_nocap["model"])},
            "weights": top_w,
            "candidate_signal_weight_share": float(candidate_weight_share),
            "candidate_v2_signal_weight_share": float(candidate_v2_weight_share),
            "note": f"全{len(races)}レースで選んだ重みなのでin-sample最適化の楽観を含む(選定基準=cap後)",
        },
        "nested_lobo_oof": {
            "cap": {"model": nested_oof_cap["model"], "excess": nested_oof_cap["excess"]},
            "nocap": {"model": nested_oof_nocap["model"], "excess": nested_oof_nocap["excess"]},
        },
        "selection_optimism_cap": opt,
        "bootstrap_full_population": {"cap": boot_cap, "nocap": boot_nocap},
        "gates": {
            "nested_positive": bool(gate1_nested_positive),
            "true_edge_positive": bool(gate2_true_edge_positive),
            "boot_ci_lo_above_breakeven": bool(gate3_boot_ci_lo_above_breakeven),
            "n_gates_passed": int(n_gates_passed),
            "n_gates_total": 3,
        },
        "decision": decision,
    }

log("\n" + "=" * 72)
log("まとめ(cap後=払戻2万円以上のレースを除外した回収率)")
log("=" * 72)
for BOX_N in BOX_NS:
    r = results_by_box[BOX_N]
    log(f"box_n={BOX_N}: 市場(cap)={r['market']['cap']:.1f}%  "
        f"等重み(cap)={r['equal_weight']['cap']['model']:.1f}%(市場差{r['equal_weight']['cap']['excess']:+.1f}pt)  "
        f"全data探索(cap,in-sample)={r['best_full_population']['cap']['model']:.1f}%"
        f"(市場差{r['best_full_population']['cap']['excess']:+.1f}pt)  "
        f"Nested LOBO OOF(cap)={r['nested_lobo_oof']['cap']['model']:.1f}%"
        f"(市場差{r['nested_lobo_oof']['cap']['excess']:+.1f}pt)  "
        f"選ぶことの真の価値={r['selection_optimism_cap']['true_edge_pt']:+.1f}pt  "
        f"判定={r['decision']}")

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps({
    "n_races": len(races), "dates": data["dates"], "pool": NAMES,
    "n_patterns": N_PATTERNS, "seed_base": SEED, "box_ns": list(BOX_NS),
    "max_payout_cap": MAX_PAYOUT, "priors_fresh": priors_fresh,
    "results_by_box": results_by_box,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
