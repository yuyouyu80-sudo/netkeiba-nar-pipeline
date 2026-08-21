# -*- coding: utf-8 -*-
"""ユーザー依頼(2026-08-21「勝率・回収率を上げる工夫をお願いします」): NARのCANDIDATE_SIGNALS_V4
(hold_just/hold_wide/jockey_change/class_drop/weight_trend)をJRAへ移植した5新規シグナル
(jra_signals.CANDIDATE_SIGNALS_V3)を含む30本のプールで、box5/4/3現行モデルを上回るか
再探索する。box買い(BoxSettler)向け。同日の軸流し向けは
jra_axis_search500_2026_08_21_v3signals.py を参照。

過去3回の独立探索(box25シグナル拡張・軸流しワイド目的・軸複勝的中率目的)がいずれも
「市場に対する統計的優位性なし」でREJECTEDだった実績を踏まえ、既に確立済みの手法をそのまま
踏襲する(車輪の再発明をしない):
  (1) 重み生成はWEIGHT_TIERS(等重み近傍集中サンプリング)、jra_axis_search_2026_08_21.pyと同一設計。
      旧jra_search_2026_08_11/12.pyのDirichlet([1]*n)一様サンプリング(欠陥あり)は使わない。
  (2) Nested LOBO OOFの退化検知(n_unique_patterns/n_folds)を必須で行う。
  (3) 選択バイアス診断(ブロック半分割×200反復)のtrue_edge_pt/true_edge_sd比を主判定に使う
      (採否ゲート: >=2.0)。
  (4) ブートストラップは選択循環を避けるため、Nested LOBO OOFのheld-out picksに対して行う。

出力: jra_search500_2026_08_21_v3signals_result.json / _report.txt (data/jra_pipeline、
git管理下、研究ログ)。本番重み(data/jra_pipeline/winner_v3.json等)への反映は、専門家レビュー
(ゲート1: 本スクリプト実行前の設計レビュー、ゲート2: 実行後の結果レビュー)を経てユーザーが
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
import jra_eval as JE  # noqa: E402
import jra_signals as JS  # noqa: E402

N_PATTERNS = 500
SEED = 2851  # 2026-08-21新規シード(既存の2033/2044/2820/2821/2822とは別系列)
BOX_NS = (5, 4, 3)
WINNER_FILES = {5: "winner_v3.json", 4: "winner_box4.json", 3: "winner_box3.json"}
DECISION_GATE_RATIO = 2.0

WEIGHT_TIERS = [
    (100.0, 150),
    (25.0, 150),
    (6.0, 100),
    (1.0, 99),
]
assert 1 + sum(n for _, n in WEIGHT_TIERS) == N_PATTERNS

OUT_JSON = DATA_DIR / "jra_search500_2026_08_21_v3signals_result.json"
OUT_TXT = DATA_DIR / "jra_search500_2026_08_21_v3signals_report.txt"

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


data = jra_dataset.load(rebuild=True)
races, actual = data["races"], data["actual"]
NAMES = JS.ALL_SIGNALS  # 25(既存)+5(候補第3弾、2026-08-21新設)=30シグナル
log(f"レース数: {len(races)}  日付: {data['dates'][0]}〜{data['dates'][-1]}({len(data['dates'])}日)"
    f"  頭数: {sum(len(r['df']) for r in races)}")
log(f"探索対象プール({len(NAMES)}シグナル、うち候補第3弾5本: {JS.CANDIDATE_SIGNALS_V3}): {NAMES}")
log(f"パターン数: {N_PATTERNS}  乱数シード: {SEED}")
log(f"重み生成tiers(濃度, 本数): {WEIGHT_TIERS} + 厳密等重み1本")

priors_all = JS.make_priors([r["df"] for r in races])
log(f"priors再計算: {len(priors_all)}キー({len(races)}レース全体から)")

mats_all = JS.signal_matrices(races, priors_all, NAMES, JS.CLASS_ORDINAL)

# 新規5シグナルの充足率・既存holdtimeとの相関を実行時に必ず記録する(Front1設計レビュー指摘、
# hold_just/hold_wideは既存holdtimeと同一データソース由来のため多重共線性を確認する)。
S_all = np.vstack([m["S"] for m in mats_all])
A_all = np.vstack([m["A"] for m in mats_all])
signal_diagnostics = {}
for n in JS.CANDIDATE_SIGNALS_V3:
    i = NAMES.index(n)
    avail_pct = float(A_all[:, i].mean() * 100)
    signal_diagnostics[n] = {"available_pct": avail_pct}
log("\n[候補第3弾シグナルの充足率]")
for n, d in signal_diagnostics.items():
    log(f"  {n:14s} 非NaN率={d['available_pct']:5.1f}%")

i_hj, i_hw, i_ht = NAMES.index("hold_just"), NAMES.index("hold_wide"), NAMES.index("holdtime")
both = (A_all[:, i_hj] > 0) & (A_all[:, i_ht] > 0)
corr_hj_ht = float(np.corrcoef(S_all[both, i_hj], S_all[both, i_ht])[0, 1]) if both.sum() > 5 else float("nan")
both2 = (A_all[:, i_hw] > 0) & (A_all[:, i_ht] > 0)
corr_hw_ht = float(np.corrcoef(S_all[both2, i_hw], S_all[both2, i_ht])[0, 1]) if both2.sum() > 5 else float("nan")
log(f"[多重共線性チェック] hold_just と 既存holdtime の相関係数: {corr_hj_ht:.3f}"
    f"(n={int(both.sum())})")
log(f"[多重共線性チェック] hold_wide と 既存holdtime の相関係数: {corr_hw_ht:.3f}"
    f"(n={int(both2.sum())})")
signal_diagnostics["corr_hold_just_vs_holdtime"] = corr_hj_ht
signal_diagnostics["corr_hold_wide_vs_holdtime"] = corr_hw_ht


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


def equal_w(names_subset=None) -> np.ndarray:
    subset = names_subset if names_subset is not None else NAMES
    d = {n: 1.0 / len(subset) for n in subset}
    return wvec(d)


rng = np.random.default_rng(SEED)
cols = [equal_w()]  # pattern#0固定: 30本厳密等重み
for concentration, n in WEIGHT_TIERS:
    alpha = [concentration] * len(NAMES)
    for _ in range(n):
        cols.append(wvec(dict(zip(NAMES, rng.dirichlet(alpha)))))
W_POOL = np.column_stack(cols)
assert W_POOL.shape[1] == N_PATTERNS

W_EQUAL_30 = equal_w()
W_EQUAL_25 = equal_w(JS.LEGACY_SIGNALS + JS.CANDIDATE_SIGNALS + JS.CANDIDATE_SIGNALS_V2)

results_by_box = {}

for BOX_N in BOX_NS:
    log("\n" + "=" * 72)
    log(f"box_n={BOX_N}")
    log("=" * 72)

    winner = json.loads((DATA_DIR / WINNER_FILES[BOX_N]).read_text(encoding="utf-8"))
    W_CURRENT = wvec(winner["weights"])  # 候補シグナル分は0(現行モデルはLEGACY10のみ)

    ev = JE.Evaluator(races, actual, box_n=BOX_N)
    mkt_picks = JE.market_picks(races, BOX_N)
    mkt = ev.evaluate(mkt_picks)
    log(f"上位{BOX_N}人気BOX(市場)  複勝+ワイド={mkt['model']:.2f}%")

    current_picks = JE.score_picks(mats_all, W_CURRENT, BOX_N)
    r_current = ev.evaluate(current_picks)
    log(f"現行モデル(pattern{winner['pattern_id']}, {len(races)}レースで直接評価)  "
        f"複勝+ワイド={r_current['model']:.2f}%  市場差={r_current['excess']:+.2f}pt")

    equal25_picks = JE.score_picks(mats_all, W_EQUAL_25, BOX_N)
    r_equal25 = ev.evaluate(equal25_picks)
    log(f"25本等重み(候補第3弾を含めない、参考)  複勝+ワイド={r_equal25['model']:.2f}%  "
        f"市場差={r_equal25['excess']:+.2f}pt")

    equal30_picks = JE.score_picks(mats_all, W_EQUAL_30, BOX_N)
    r_equal30 = ev.evaluate(equal30_picks)
    log(f"30本等重み(候補第3弾を含む、参考)  複勝+ワイド={r_equal30['model']:.2f}%  "
        f"市場差={r_equal30['excess']:+.2f}pt")

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
        "  ※学習データそのもので選んでいるため楽観的(in-sample)な数字である点に注意")
    top_w = {n: float(w) for n, w in zip(NAMES, W_POOL[:, best_full]) if w > 0.005}
    log(f"  重み内訳(0.5%以上): {json.dumps(top_w, ensure_ascii=False)}")
    v3_share = sum(w for n, w in zip(NAMES, W_POOL[:, best_full]) if n in JS.CANDIDATE_SIGNALS_V3)
    log(f"  候補第3弾5本の合計重みシェア: {v3_share:.3f} ({v3_share * 100:.1f}%)")

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

    opt = JE.selection_optimism(ev, mats_all, W_POOL, n_rep=200, seed=2861)
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
    boot_oof_vs_equal30 = ev.block_bootstrap_diff(nested_oof["picks"], equal30_picks, seed=43)
    log(f"\n[Nested LOBO OOF(誠実なheld-out結果)のブートストラップCI、n=2000]")
    log(f"  市場比 95%CI[{boot_vs_market['lo']:.1f}, {boot_vs_market['hi']:.1f}](複勝+ワイド%水準)")
    log(f"  現行モデル比の差 95%CI[{boot_oof_vs_current['lo']:+.2f}, {boot_oof_vs_current['hi']:+.2f}]pt"
        "(下限が0を超える場合のみ、統計的に現行を上回ったと言える。これが採否判定の本命指標)")
    log(f"  30本等重み比の差 95%CI[{boot_oof_vs_equal30['lo']:+.2f}, {boot_oof_vs_equal30['hi']:+.2f}]pt")

    # 採用案(現行モデル)の全券種内訳。レポートに掲載するのはこちら(不採用の探索候補と
    # 取り違えないこと、2026-08-21競馬予想家レビュー指摘の教訓を踏襲)。
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
        "pool": NAMES,
        "n_patterns": N_PATTERNS,
        "market": mkt["model"],
        "current_model": {"model": r_current["model"], "excess": r_current["excess"]},
        "equal_weight_25_without_v3": {"model": r_equal25["model"], "excess": r_equal25["excess"]},
        "equal_weight_30_with_v3": {"model": r_equal30["model"], "excess": r_equal30["excess"]},
        "best_full_population": {
            "pattern_index": best_full, "model": float(full_vals[best_full]),
            "excess": float(full_vals[best_full] - mkt["model"]), "weights": top_w,
            "v3_signal_weight_share": float(v3_share),
            "note": f"全{len(races)}レースで選んだ重みなのでin-sample最適化の楽観を含む。"
                    "統計的検定には使わない(post-selection inferenceの誤りを避けるため)。",
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
        "bootstrap_oof_vs_equal30": boot_oof_vs_equal30,
        "full_table_current_model": {
            "note": "採用案(現行モデル)の全券種内訳。レポート掲載用。",
            "low_sample_bet_types": low_sample_bets,
            "low_sample_threshold_hits": LOW_SAMPLE_HITS,
            "rows": current_full_tbl.to_dict(orient="records"),
        },
        "full_table_rejected_candidate_oof_picks": {
            "note": f"不採用になった{N_PATTERNS}パターン探索モデル(Nested LOBO OOF picks)の内訳。"
                    "参考値。採用案の数値と混同しないこと。",
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
        f"30本等重み市場差={r['equal_weight_30_with_v3']['excess']:+.2f}pt  "
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
    "pool": NAMES, "n_patterns": N_PATTERNS, "seed": SEED, "weight_tiers": WEIGHT_TIERS,
    "decision_gate_ratio_threshold": DECISION_GATE_RATIO,
    "signal_diagnostics_v3": signal_diagnostics,
    "results_by_box": results_by_box,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
