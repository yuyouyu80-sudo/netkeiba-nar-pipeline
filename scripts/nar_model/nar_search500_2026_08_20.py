# -*- coding: utf-8 -*-
"""ユーザー依頼(2026-08-20): インターネット由来の勝率アップ手法を新規シグナル化
(nar_signals.py の CANDIDATE_SIGNALS_V4)した上で、重み配分パターンを500通り試し、
的中率・回収率が現行(等重み)を上回るパターンを探す。box5/box4/box3の3サイズで実施。

過去2回(2026-07-29, 2026-08-01)の300パターン自由探索は、専門家レビュー(統計学者+
システムエンジニア)により「構造的に不成立」と判定され撤回済み([[project_nar_search300_v2_reverted_2026_08_01_finding]])。
主因は2つ:
  (1) 重み生成が Dirichlet([1.0]*n) の一様サンプリングで、等重み近傍を一度も生成しない
      設計上の偏り(現行の等重みが300パターン中97パーセンタイルという逆転現象を招いた)。
  (2) 実効ブロック数(開催日×競馬場)が23と少なく、300パターンという多重比較に対して
      検出力が不足していた。

本スクリプトはこの2点を修正する:
  (1) 重み生成を「等重み(パターン#0固定)+ 等重み近傍の局所摂動(高濃度Dirichlet)
      主体 + 中程度・広域探索を少数」の混合にする(WEIGHT_TIERS参照)。
  (2) 現在の検証可能データは1385レース・127ブロックまで拡大している(実測、2026-08-20
      時点)。前回の23ブロックから約5.5倍で、検出力不足という前提条件が変化している。

探索対象プール(POOL)は「生存シグナル」から、過去に個別のLOBO OOFで既に不採用と判定
済みの interval/kinryo/waku_recent2d を除外する(これらはDirichlet(1)自由探索の欠陥とは
無関係な、単独ファクター追加テストで却下されたものなので再検討の対象にしない)。
CANDIDATE_SIGNALS_V4(2026-08-20新規、hold_just/hold_wide/jockey_change/class_drop/
weight_trend)は今回はじめて探索対象に含める。

50シグナル×500パターンは前回の17シグナル×300パターンより多重比較が厳しくなる
(候補が増えるほど「たまたま良く見える」パターンが出現しやすい)。Nested LOBO OOF・
選択バイアス診断・ブロックブートストラップCIの3点セットは前回と同一の基準で必須実施する。

出力: nar_search500_2026_08_20_result.json / _report.txt (scratchpad、研究ログ、
nar_search300_2026_08_01.py と同じ置き場所)。本番重み(data/nar_pipeline/winner_box{5,4,3}_nar.json)
への反映は、専門家レビュー(ゲート1: 本スクリプト実行前の設計レビュー、ゲート2: 本スクリプト
実行後の結果レビュー)を経てユーザーが判断する。本スクリプト自体は本番重みを書き換えない。
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

N_PATTERNS = 500
SEED = 2820  # 2026-08-20、新規シード(前回2029とは別系列であることを明示するため)
BOX_NS = (5, 4, 3)
# 重み生成の混合比率: (Dirichlet濃度パラメータ, パターン数)。濃度が高いほど等重み近傍に
# 集中する。パターン#0は別途、厳密な等重みを1本追加する(濃度∞相当)。
WEIGHT_TIERS = [
    (100.0, 150),  # 等重みのごく近傍(微小な偏差のみ)
    (25.0, 150),   # 中程度の偏差
    (6.0, 100),    # やや広い探索
    (1.0, 99),     # 旧来のDirichlet(1)一様探索(比較用に少数だけ残す)
]
assert 1 + sum(n for _, n in WEIGHT_TIERS) == N_PATTERNS

OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
OUT_JSON = OUT_DIR / "nar_search500_2026_08_20_result.json"
OUT_TXT = OUT_DIR / "nar_search500_2026_08_20_report.txt"

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


data = nar_dataset.load(rebuild=True)
races, actual = data["races"], data["actual"]
priors_all = NS.make_priors(races)
dead = NS.detect_dead(races, priors_all)
EXCLUDED_PREVIOUSLY_REJECTED = NS.CANDIDATE_SIGNALS + NS.CANDIDATE_SIGNALS_V3  # interval/kinryo/waku_recent2d
alive_base = [n for n in NS.LEGACY_SIGNALS + NS.NEW_SIGNALS if n not in dead]
v2_alive = [n for n in NS.CANDIDATE_SIGNALS_V2 if n not in dead]
v4_alive = [n for n in NS.CANDIDATE_SIGNALS_V4 if n not in dead]
# 2026-08-20ゲート1(統計学者レビュー)指摘: 「基準」を22本(V4込み)だけにすると、
# 実際の本番(data/nar_pipeline/winner_box*_nar.json、17本等重み)と混同されるおそれが
# あった。POOL_TRUE_PRODを別途持ち、「17本(真の現行本番) / 22本(V4込み・未検証の新基準)
# / 探索結果」の3点比較にする。
POOL_TRUE_PROD = alive_base + v2_alive  # 17本、winner_box*_nar.jsonの現行本番と同一構成
POOL = POOL_TRUE_PROD + v4_alive  # 22本、今回の探索対象プール

log(f"レース数: {len(races)}  日付: {data['dates'][0]}〜{data['dates'][-1]}({len(data['dates'])}日)"
    f"  頭数: {sum(len(r['df']) for r in races)}")
log(f"死にシグナル({len(dead)}): {dead}")
log(f"既に個別テストで不採用済み・今回も除外({len(EXCLUDED_PREVIOUSLY_REJECTED)}): "
    f"{EXCLUDED_PREVIOUSLY_REJECTED}")
log(f"真の現行本番プール(POOL_TRUE_PROD、{len(POOL_TRUE_PROD)}本、winner_box*_nar.jsonと同一構成): "
    f"{POOL_TRUE_PROD}")
log(f"探索対象プール(POOL、生存{len(POOL)}、うち2026-08-20新規{len(v4_alive)}本): {POOL}")
log(f"目標50シグナルに対し実際にプールへ組み込めたのは{len(POOL)}本。理由: "
    "NAR側で構造的に空の項目(speed/apt/train/bms、surf_ketto系等)・既に個別却下済みの"
    "項目を除くと安全に探索できる本数はこれだけだった(nar_signals.pyのコメント参照)。")
log(f"パターン数: {N_PATTERNS}  乱数シード: {SEED}")
log(f"重み生成tiers(濃度, 本数): {WEIGHT_TIERS} + 厳密等重み1本")

NAMES = NS.ALL_SIGNALS
mats_all = NS.signal_matrices(races, priors_all, NAMES)


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


def equal_w(subset) -> np.ndarray:
    d = {n: 1.0 / len(subset) for n in subset}
    return wvec(d)


rng = np.random.default_rng(SEED)
cols = [equal_w(POOL)]  # pattern#0固定: 厳密な等重み
for concentration, n in WEIGHT_TIERS:
    alpha = [concentration] * len(POOL)
    for _ in range(n):
        cols.append(wvec(dict(zip(POOL, rng.dirichlet(alpha)))))
W_POOL = np.column_stack(cols)
assert W_POOL.shape[1] == N_PATTERNS

W_BASE_EQUAL = equal_w(POOL)
W_TRUE_PROD = equal_w(POOL_TRUE_PROD)

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
    log(f"{len(POOL)}本(V4込み)等重み・未検証の新基準  "
        f"複勝+ワイド={r_base['model']:.2f}%  市場差={r_base['excess']:+.2f}pt")

    true_prod_picks = NE.score_picks(mats_all, W_TRUE_PROD, BOX_N)
    r_true_prod = ev.evaluate(true_prod_picks)
    log(f"{len(POOL_TRUE_PROD)}本等重み・真の現行本番(winner_box{BOX_N}_nar.jsonと同一構成)  "
        f"複勝+ワイド={r_true_prod['model']:.2f}%  市場差={r_true_prod['excess']:+.2f}pt")

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
    log(f"\n[Nested LOBO OOF] {len(ev.block_ids)}ブロック(開催日×競馬場)で500パターン探索という"
        f"手続き全体を交差検証: 複勝+ワイド={nested_oof['model']:.2f}%  市場差={nested_oof['excess']:+.2f}pt"
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
    gate_nested_beats_true_prod = nested_oof["excess"] > r_true_prod["excess"]
    gate_nested_positive = nested_oof["excess"] > 0
    log(f"\n参考ゲート(ブロッキングではなく報告のみ、最終判断はフェーズ3の専門家レビューで行う):")
    log(f"  Nested LOBO市場差が{len(POOL)}本基準を上回るか: {'YES' if gate_nested_beats_base else 'NO'} "
        f"({nested_oof['excess']:+.2f} vs {r_base['excess']:+.2f})")
    log(f"  Nested LOBO市場差が{len(POOL_TRUE_PROD)}本・真の現行本番を上回るか: "
        f"{'YES' if gate_nested_beats_true_prod else 'NO'} "
        f"({nested_oof['excess']:+.2f} vs {r_true_prod['excess']:+.2f})")
    log(f"  Nested LOBO市場差がプラスか              : {'YES' if gate_nested_positive else 'NO'} "
        f"({nested_oof['excess']:+.2f}pt)")

    # 2026-08-20ゲート1(統計学者レビュー)の致命的指摘への対応: 当初は「in-sampleの
    # argmaxで選んだ最良パターン」を、選抜に使ったのと同じ全データからブロックブートストラップ
    # しており、選択循環(post-selection inference、選んでから同じデータで検定する誤り)に
    # 陥っていた。Nested LOBO OOFのpicks(各ブロックをtrain側だけで選んだ重みで予測した、
    # 正当なheld-out結果)を使って差分をブートストラップする形に修正した。
    def block_bootstrap_diff(picks_a, picks_b, n=2000, seed=41):
        st_a, rt_a = ev.settler.returns_for(picks_a)
        st_b, rt_b = ev.settler.returns_for(picks_b)
        by_block = {b: np.where(ev.blocks == b)[0] for b in ev.block_ids}
        ids = list(ev.block_ids)
        obj_cols = [NB.BET_TYPES.index(b) for b in NE.OBJ_BETS]
        rng_ci = np.random.default_rng(seed)
        diffs = np.empty(n)
        for k in range(n):
            chosen = rng_ci.choice(len(ids), size=len(ids), replace=True)
            idx = np.concatenate([by_block[ids[c]] for c in chosen])
            sa, ra = st_a[np.ix_(idx, obj_cols)].sum(), rt_a[np.ix_(idx, obj_cols)].sum()
            sb, rb = st_b[np.ix_(idx, obj_cols)].sum(), rt_b[np.ix_(idx, obj_cols)].sum()
            va = ra / sa * 100 if sa else 0.0
            vb = rb / sb * 100 if sb else 0.0
            diffs[k] = va - vb
        return {"mean": float(diffs.mean()), "lo": float(np.percentile(diffs, 2.5)),
                "hi": float(np.percentile(diffs, 97.5))}

    boot_vs_market = ev.block_bootstrap(nested_oof["picks"], n=2000, seed=31)
    boot_oof_vs_base = block_bootstrap_diff(nested_oof["picks"], base_picks, seed=41)
    boot_oof_vs_true_prod = block_bootstrap_diff(nested_oof["picks"], true_prod_picks, seed=43)
    log(f"\n[Nested LOBO OOF(誠実なheld-out結果)のブートストラップCI、n=2000]")
    log(f"  市場比 95%CI[{boot_vs_market['lo']:.1f}, {boot_vs_market['hi']:.1f}](複勝+ワイド%水準)")
    log(f"  {len(POOL)}本基準比の差 95%CI[{boot_oof_vs_base['lo']:+.2f}, {boot_oof_vs_base['hi']:+.2f}]pt")
    log(f"  {len(POOL_TRUE_PROD)}本・真の現行本番比の差 95%CI[{boot_oof_vs_true_prod['lo']:+.2f}, "
        f"{boot_oof_vs_true_prod['hi']:+.2f}]pt (下限が0を超える場合のみ、統計的に現行本番を"
        "上回ったと言える。これが採否判定の本命指標)")
    log(f"  参考(in-sample最良パターン、全{len(races)}レース): "
        f"複勝+ワイド={full_vals[best_full]:.2f}%  ※選択循環を避けるため統計的検定には使わない")

    results_by_box[BOX_N] = {
        "pool": POOL, "pool_true_prod": POOL_TRUE_PROD,
        "n_patterns": N_PATTERNS,
        "market": mkt["model"],
        "baseline_new_candidate_equal_weight": {"model": r_base["model"], "excess": r_base["excess"]},
        "true_production_equal_weight": {"model": r_true_prod["model"], "excess": r_true_prod["excess"]},
        "best_full_population": {
            "pattern_index": best_full, "model": float(full_vals[best_full]),
            "excess": float(full_vals[best_full] - mkt["model"]), "weights": top_w,
            "note": f"全{len(races)}レースで選んだ重みなのでin-sample最適化の楽観を含む。"
                    "統計的検定には使わない(post-selection inferenceの誤りを避けるため)。",
        },
        "nested_lobo_oof": {"model": nested_oof["model"], "excess": nested_oof["excess"]},
        "selection_optimism": opt,
        "gates_report_only": {
            "nested_beats_new_candidate_baseline": bool(gate_nested_beats_base),
            "nested_beats_true_production": bool(gate_nested_beats_true_prod),
            "nested_positive": bool(gate_nested_positive),
        },
        "bootstrap_oof_vs_market": boot_vs_market,
        "bootstrap_oof_vs_new_candidate_baseline": boot_oof_vs_base,
        "bootstrap_oof_vs_true_production": boot_oof_vs_true_prod,
    }

log("\n" + "=" * 72)
log("まとめ")
log("=" * 72)
for BOX_N in BOX_NS:
    r = results_by_box[BOX_N]
    boot_tp = r["bootstrap_oof_vs_true_production"]
    log(f"box_n={BOX_N}: 真の現行本番({len(r['pool_true_prod'])}本)市場差="
        f"{r['true_production_equal_weight']['excess']:+.2f}pt  "
        f"全data探索(in-sample、参考値)市場差={r['best_full_population']['excess']:+.2f}pt  "
        f"Nested LOBO OOF市場差={r['nested_lobo_oof']['excess']:+.2f}pt  "
        f"選ぶことの真の価値={r['selection_optimism']['true_edge_pt']:+.2f}pt"
        f"(sd{r['selection_optimism']['true_edge_sd']:.2f})  "
        f"真の現行本番比CI=[{boot_tp['lo']:+.2f}, {boot_tp['hi']:+.2f}]pt")

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps({
    "n_races": len(races), "dates": data["dates"], "n_blocks": len(set(
        f'{r["kaisai_date"]}_{r["racecourse"]}' for r in races)),
    "dead_signals": dead, "excluded_previously_rejected": EXCLUDED_PREVIOUSLY_REJECTED,
    "pool": POOL, "n_patterns": N_PATTERNS, "seed": SEED, "weight_tiers": WEIGHT_TIERS,
    "results_by_box": results_by_box,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
