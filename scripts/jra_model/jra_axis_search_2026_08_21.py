# -*- coding: utf-8 -*-
"""ユーザー依頼(2026-08-21): JRA予想の「1頭軸+相手2〜4頭」流し買い(馬連・ワイド・3連複・
馬単・3連単)で勝率・回収率が最高になる軸馬選定モデルを重み探索する。box買いと違い軸流しは
「軸馬が的中の要」なので、box5/4/3向けに選定された現行重み(winner_v3.json等)がそのまま
最適とは限らない。axis5/4/3(軸+相手4/3/2頭=合計5/4/3頭)それぞれ独立に探索する。

[[project_nar_search500_v4_signals_2026_08_20_finding]]で確立した修正済み手法をそのまま
踏襲する(車輪の再発明をしない):
  (1) 重み生成は Dirichlet([1]*n) の一様サンプリング(旧jra_search_2026_08_11.pyの欠陥、
      等重み近傍を一度も生成しない)ではなく、等重み(パターン#0固定)+ 高濃度Dirichletによる
      等重み近傍集中サンプリング主体の混合(WEIGHT_TIERS)にする。
  (2) Nested LOBO OOFが「候補間の差が大きい場合にin-sample評価へ退化する」新種の落とし穴
      (2026-08-20/21にNAR側で発見)を検知するため、各foldで実際に選ばれたパターンの
      ユニーク数(n_unique_patterns)を必ず記録する。全fold同一パターンの場合はNested LOBO OOFを
      統計的検定に使わず、選択バイアス診断(ブロック半分割)のtrue_edge_pt/sdを主指標にする。
  (3) ブートストラップは選択循環(post-selection inference)を避けるため、Nested LOBO OOFの
      held-out picksに対して行う(in-sampleのargmaxパターンには行わない)。

探索プールはjra_signals.ALL_SIGNALS(25シグナル、box探索と同一)をそのまま使う。今回は賭け方の
新設が目的であり新規シグナルは実装しない(box探索で候補拡張は不採用済みだが、目的関数が
異なる軸流しでは別の重みが最適になりうるため、同じプールで独立に探索する)。

出力: jra_axis_search_2026_08_21_result.json / _report.txt (data/jra_pipeline、git管理下、
研究ログ)。本番重み(採用時: data/jra_pipeline/winner_axis{5,4,3}.json)への反映は、専門家
レビュー(ゲート1: 本スクリプト実行前の設計レビュー、ゲート2: 実行後の結果レビュー)を経て
ユーザーが判断する。本スクリプト自体は本番重みを書き換えない。
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
import jra_axis_backtest as AB  # noqa: E402
import jra_axis_eval as AE  # noqa: E402
import jra_dataset  # noqa: E402
import jra_signals as JS  # noqa: E402

N_PATTERNS = 500
SEED = 2821  # 2026-08-21、新規シード(NARのnar_search500_2026_08_20.py=2820とは別系列)
BOX_NS = (5, 4, 3)  # 軸+相手の合計頭数(axis5=軸+相手4、axis4=軸+相手3、axis3=軸+相手2)
CURRENT_BOX_WEIGHT_FILES = {5: "winner_v3.json", 4: "winner_box4.json", 3: "winner_box3.json"}
DECISION_GATE_RATIO = 2.0  # true_edge_pt / true_edge_sd >= 2.0(winner_v3.jsonに記録済みの既存基準)

# 2026-08-21 統計学者ゲート1レビュー指摘への対応: winner_v3/box4/box3.jsonはいずれも同一の
# fitted_on(105レース、6開催日)を記録している。この6日は「現行box重み」自体の選定に
# 直接使われた母集団であり、今回の軸流し評価(211レース、12開催日)にそのまま含めて
# 「現行box重み転用+18〜23pt」を算出すると二重に楽観的な数字になる(選定に使ったデータで
# 選定結果を評価している)。よって「元の重み決定に一度も使われていない開催日だけ」の
# 部分集合でも別途評価する。
ORIGINAL_FIT_DATES = {"20260711", "20260712", "20260718", "20260719", "20260725", "20260726"}
# さらに保守的に、2026-08-12の(不採用に終わった)拡張候補再検証探索が参照した10開催日
# (177レース、jra_search_2026_08_12_result.json)も除いた、どの重み関連プロセスにも
# 一度も使われていない開催日のみの部分集合。
DIAGNOSTIC_SEARCH_DATES = ORIGINAL_FIT_DATES | {"20260801", "20260802", "20260808", "20260809"}

# 重み生成の混合比率: (Dirichlet濃度パラメータ, パターン数)。nar_search500_2026_08_20.pyと同一設計。
WEIGHT_TIERS = [
    (100.0, 150),
    (25.0, 150),
    (6.0, 100),
    (1.0, 99),
]
assert 1 + sum(n for _, n in WEIGHT_TIERS) == N_PATTERNS

OUT_JSON = DATA_DIR / "jra_axis_search_2026_08_21_result.json"
OUT_TXT = DATA_DIR / "jra_axis_search_2026_08_21_report.txt"

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


data = jra_dataset.load(rebuild=True)
races, actual = data["races"], data["actual"]
NAMES = JS.ALL_SIGNALS  # 25シグナル、box探索と同一プール(新規シグナルは実装しない)
log(f"レース数: {len(races)}  日付: {data['dates'][0]}〜{data['dates'][-1]}({len(data['dates'])}日)"
    f"  頭数: {sum(len(r['df']) for r in races)}")
log(f"探索対象プール({len(NAMES)}シグナル、box探索と同一): {NAMES}")
log(f"パターン数: {N_PATTERNS}  乱数シード: {SEED}")
log(f"重み生成tiers(濃度, 本数): {WEIGHT_TIERS} + 厳密等重み1本")
log(f"対象券種(jra_axis_backtest.BET_TYPES_AXIS、7区分): {AB.BET_TYPES_AXIS}")
log(f"重み探索の目的関数: {AE.OBJ_BETS_AXIS}のコスト加重回収率(低分散のため。"
    "馬単・3連単は1点あたり配当の分散が極端に大きくin-sample探索がノイズに支配されるため除外)")

priors_all = JS.make_priors([r["df"] for r in races])
log(f"priors再計算: {len(priors_all)}キー({len(races)}レース全体から)")

mats_all = JS.signal_matrices(races, priors_all, NAMES, JS.CLASS_ORDINAL)


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


def equal_w() -> np.ndarray:
    d = {n: 1.0 / len(NAMES) for n in NAMES}
    return wvec(d)


rng = np.random.default_rng(SEED)
cols = [equal_w()]  # pattern#0固定: 厳密な等重み
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
    log(f"axis{BOX_N}(軸+相手{BOX_N - 1}頭、合計{BOX_N}頭)")
    log("=" * 72)

    ev = AE.Evaluator(races, actual, box_n=BOX_N)
    mkt_picks = AE.market_picks(races, BOX_N)
    mkt = ev.evaluate(mkt_picks)
    log(f"軸=1番人気・相手=2〜{BOX_N}番人気(市場)  ワイド={mkt['model']:.2f}%")

    # 現行box{N}の重み(10シグナルのみ、他は0埋め)をそのまま軸流しに転用した場合の成績。
    current_box_w = json.loads((DATA_DIR / CURRENT_BOX_WEIGHT_FILES[BOX_N]).read_text(encoding="utf-8"))
    W_CURRENT_BOX = wvec(current_box_w["weights"])
    current_picks = AE.score_picks(mats_all, W_CURRENT_BOX, BOX_N)
    r_current = ev.evaluate(current_picks)
    log(f"現行box{BOX_N}重み({CURRENT_BOX_WEIGHT_FILES[BOX_N]})を軸流しに転用(全{len(races)}レース、"
        f"うち重み自体のfit母集団と重複するレースを含む)  "
        f"ワイド={r_current['model']:.2f}%  市場差={r_current['excess']:+.2f}pt")

    # fit母集団と重複しない開催日だけに絞った、二重に楽観的でない評価。
    date_arr = np.array([r["kaisai_date"] for r in races])
    idx_post_fit = np.where(~np.isin(date_arr, list(ORIGINAL_FIT_DATES)))[0]
    idx_post_diag = np.where(~np.isin(date_arr, list(DIAGNOSTIC_SEARCH_DATES)))[0]
    blocks_post_fit = sorted(set(ev.blocks[idx_post_fit]))
    blocks_post_diag = sorted(set(ev.blocks[idx_post_diag])) if len(idx_post_diag) else []

    r_current_post_fit = ev.evaluate(current_picks, idx=idx_post_fit)
    mkt_post_fit = ev.evaluate(mkt_picks, idx=idx_post_fit)
    log(f"  → 元の重み決定(105レース、{sorted(ORIGINAL_FIT_DATES)})と重複しない"
        f"{len(idx_post_fit)}レース({len(blocks_post_fit)}ブロック)のみで再評価: "
        f"ワイド={r_current_post_fit['model']:.2f}%  "
        f"市場(同レース集合、{mkt_post_fit['model']:.2f}%)差={r_current_post_fit['model'] - mkt_post_fit['model']:+.2f}pt")

    if len(idx_post_diag) >= 20:  # ブロック数が少なすぎる場合は参考にならないため足切り
        r_current_post_diag = ev.evaluate(current_picks, idx=idx_post_diag)
        mkt_post_diag = ev.evaluate(mkt_picks, idx=idx_post_diag)
        log(f"  → さらに保守的に、2026-08-12の再検証探索(177レース、{sorted(DIAGNOSTIC_SEARCH_DATES)})"
            f"にも一度も使われていない{len(idx_post_diag)}レース({len(blocks_post_diag)}ブロック)のみ: "
            f"ワイド={r_current_post_diag['model']:.2f}%  "
            f"市場差={r_current_post_diag['model'] - mkt_post_diag['model']:+.2f}pt"
            "  (※ブロック数が少なく参考値)")
    else:
        r_current_post_diag = None
        log(f"  → 2026-08-12の再検証探索にも使われていないレースは{len(idx_post_diag)}件のみ"
            "(ブロック数不足のため参考値としても算出しない)")

    # 現行box重み転用が「fit母集団と重複しない開催日だけ」でも本当に市場を上回るかを
    # ブロックブートストラップで検定する(点推定だけでなく信頼区間を出す)。
    boot_current_vs_market_post_fit = ev.block_bootstrap_diff(
        current_picks, mkt_picks, n=2000, seed=51, block_subset=blocks_post_fit)
    log(f"  → 上記{len(blocks_post_fit)}ブロックでの現行重み転用−市場 差の95%CI="
        f"[{boot_current_vs_market_post_fit['lo']:+.2f}, {boot_current_vs_market_post_fit['hi']:+.2f}]pt"
        "(下限が0を超える場合のみ、fit母集団と重複しないデータでも統計的に市場を上回ると言える)")

    equal_picks = AE.score_picks(mats_all, W_EQUAL, BOX_N)
    r_equal = ev.evaluate(equal_picks)
    log(f"{len(NAMES)}本等重み(参考)  ワイド={r_equal['model']:.2f}%  市場差={r_equal['excess']:+.2f}pt")

    all_picks = [AE.score_picks(mats_all, W_POOL[:, j], BOX_N) for j in range(N_PATTERNS)]
    all_st, all_rt = [], []
    for p in all_picks:
        s, r = ev.settler.returns_for(p)
        all_st.append(s)
        all_rt.append(r)
    full_vals = np.array([AE.cost_weighted_rate(all_st[j], all_rt[j]) for j in range(N_PATTERNS)])
    best_full = int(np.argmax(full_vals))
    log(f"\n[全{len(races)}レースで最良の1パターン] pattern#{best_full}  "
        f"ワイド={full_vals[best_full]:.2f}%(市場差={full_vals[best_full] - mkt['model']:+.2f}pt)"
        "  ※学習データそのもので選んでいるため楽観的(in-sample)な数字である点に注意")
    top_w = {n: float(w) for n, w in zip(NAMES, W_POOL[:, best_full]) if w > 0.005}
    log(f"  重み内訳(0.5%以上): {json.dumps(top_w, ensure_ascii=False)}")

    def fit_fn(train_idx, all_st=all_st, all_rt=all_rt):
        vals = np.array([AE.cost_weighted_rate(all_st[j], all_rt[j], idx=train_idx) for j in range(N_PATTERNS)])
        best = int(np.argmax(vals))
        return W_POOL[:, best], best

    nested_oof = ev.lobo_oof(fit_fn, mats_all)
    n_unique = nested_oof["n_unique_patterns"]
    n_folds = nested_oof["n_folds"]
    degenerate = n_unique == 1
    log(f"\n[Nested LOBO OOF] {n_folds}ブロック(開催日×競馬場)で{N_PATTERNS}パターン探索という"
        f"手続き全体を交差検証: ワイド={nested_oof['model']:.2f}%  市場差={nested_oof['excess']:+.2f}pt")
    log(f"  fold毎の選択パターンのユニーク数: {n_unique}/{n_folds}"
        + ("  ※全fold同一パターン=LOBO退化(2026-08-20/21発見の落とし穴)。"
           "この数値はin-sample評価として扱い、統計的検定には使わない" if degenerate else ""))

    opt = AE.selection_optimism(ev, mats_all, W_POOL, n_rep=200, seed=2028)
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
    log(f"  市場比 95%CI[{boot_vs_market['lo']:.1f}, {boot_vs_market['hi']:.1f}](ワイド%水準)")
    log(f"  現行box{BOX_N}重み転用比の差 95%CI[{boot_oof_vs_current['lo']:+.2f}, "
        f"{boot_oof_vs_current['hi']:+.2f}]pt(下限が0を超える場合のみ、統計的に現行を上回ったと"
        "言える。これが採否判定の本命指標)")
    log(f"  {len(NAMES)}本等重み比の差 95%CI[{boot_oof_vs_equal['lo']:+.2f}, {boot_oof_vs_equal['hi']:+.2f}]pt")

    # 採用案(現行box重み転用)の7券種全部の的中率・回収率テーブル。レポートに掲載するのは
    # こちら(2026-08-21競馬予想家レビュー指摘: 不採用の探索候補の内訳と取り違えないこと)。
    current_full_tbl = ev.full_table(current_picks)
    LOW_SAMPLE_HITS = 10  # この的中数未満は「参考値」として明示的に注記する
    low_sample_bets = current_full_tbl.loc[
        current_full_tbl["hit_races"] < LOW_SAMPLE_HITS, "bet_type"].tolist()
    log(f"\n[採用案(現行box{BOX_N}重み転用)の7券種内訳、{len(races)}レース]\n"
        f"{current_full_tbl.to_string(index=False)}")
    if low_sample_bets:
        log(f"  ※的中数{LOW_SAMPLE_HITS}未満(参考値、回収率の解釈に注意): {low_sample_bets}")

    # 参考: 不採用になった探索候補(Nested LOBO OOFのheld-out picks)側の同テーブル。
    # 採用案の数値と混同しないよう別キーで保存する。
    rejected_candidate_full_tbl = ev.full_table(nested_oof["picks"])
    log(f"\n[参考・不採用: 500パターン探索モデル(Nested LOBO OOF picks)の7券種内訳]\n"
        f"{rejected_candidate_full_tbl.to_string(index=False)}")

    results_by_box[BOX_N] = {
        "pool": NAMES,
        "n_patterns": N_PATTERNS,
        "market": mkt["model"],
        "current_box_weight_reused": {
            "file": CURRENT_BOX_WEIGHT_FILES[BOX_N],
            "model": r_current["model"], "excess": r_current["excess"],
            "caveat": "この数値は現行box重み自体のfit母集団(105レース、6開催日)を含む全"
                      f"{len(races)}レースでの評価であり、選定に使ったデータで選定結果を評価する"
                      "二重の楽観を含む(2026-08-21統計学者レビュー指摘)。下記post_fit_onlyが"
                      "公正な評価。",
            "post_fit_only": {
                "n_races": int(len(idx_post_fit)), "n_blocks": len(blocks_post_fit),
                "dates_excluded": sorted(ORIGINAL_FIT_DATES),
                "model": r_current_post_fit["model"],
                "excess_vs_market_same_subset": r_current_post_fit["model"] - mkt_post_fit["model"],
                "bootstrap_excess_vs_market_ci": boot_current_vs_market_post_fit,
            },
            "post_diagnostic_search_only": None if r_current_post_diag is None else {
                "n_races": int(len(idx_post_diag)), "n_blocks": len(blocks_post_diag),
                "dates_excluded": sorted(DIAGNOSTIC_SEARCH_DATES),
                "model": r_current_post_diag["model"],
                "excess_vs_market_same_subset": r_current_post_diag["model"] - mkt_post_diag["model"],
                "note": "ブロック数が少なく参考値",
            },
        },
        "equal_weight_25signals": {"model": r_equal["model"], "excess": r_equal["excess"]},
        "best_full_population": {
            "pattern_index": best_full, "model": float(full_vals[best_full]),
            "excess": float(full_vals[best_full] - mkt["model"]), "weights": top_w,
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
        "bootstrap_oof_vs_current_box_weight": boot_oof_vs_current,
        "bootstrap_oof_vs_equal_weight": boot_oof_vs_equal,
        "full_table_current_box_weight": {
            "note": "採用案(現行box重み転用)の7券種内訳。レポート掲載用。",
            "low_sample_bet_types": low_sample_bets,
            "low_sample_threshold_hits": LOW_SAMPLE_HITS,
            "rows": current_full_tbl.to_dict(orient="records"),
        },
        "full_table_rejected_candidate_oof_picks": {
            "note": "不採用になった500パターン探索モデル(Nested LOBO OOF picks)の内訳。参考値。"
                    "採用案の数値と混同しないこと。",
            "rows": rejected_candidate_full_tbl.to_dict(orient="records"),
        },
    }

log("\n" + "=" * 72)
log("まとめ")
log("=" * 72)
for BOX_N in BOX_NS:
    r = results_by_box[BOX_N]
    boot_c = r["bootstrap_oof_vs_current_box_weight"]
    pf = r["current_box_weight_reused"]["post_fit_only"]
    pf_ci = pf["bootstrap_excess_vs_market_ci"]
    log(f"axis{BOX_N}: 現行box重み転用(全レース、fit母集団含む・楽観的) "
        f"市場差={r['current_box_weight_reused']['excess']:+.2f}pt / "
        f"fit母集団除外後({pf['n_races']}レース、{pf['n_blocks']}ブロック、公正な評価) "
        f"市場差={pf['excess_vs_market_same_subset']:+.2f}pt CI=[{pf_ci['lo']:+.2f}, {pf_ci['hi']:+.2f}]pt  "
        f"全data探索(in-sample、参考値)市場差={r['best_full_population']['excess']:+.2f}pt  "
        f"Nested LOBO OOF市場差={r['nested_lobo_oof']['excess']:+.2f}pt"
        f"(ユニークpattern {r['nested_lobo_oof']['n_unique_patterns']}/{r['nested_lobo_oof']['n_folds']})  "
        f"選ぶことの真の価値={r['selection_optimism']['true_edge_pt']:+.2f}pt"
        f"(sd{r['selection_optimism']['true_edge_sd']:.2f})  "
        f"現行box重み転用比CI=[{boot_c['lo']:+.2f}, {boot_c['hi']:+.2f}]pt  判定={r['decision']}")

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps({
    "n_races": len(races), "dates": data["dates"], "n_blocks": len(set(
        f'{r["kaisai_date"]}_{r["racecourse"]}' for r in races)),
    "pool": NAMES, "n_patterns": N_PATTERNS, "seed": SEED, "weight_tiers": WEIGHT_TIERS,
    "obj_bets": AE.OBJ_BETS_AXIS, "decision_gate_ratio_threshold": DECISION_GATE_RATIO,
    "results_by_box": results_by_box,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
