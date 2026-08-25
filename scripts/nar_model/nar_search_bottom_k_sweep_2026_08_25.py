# -*- coding: utf-8 -*-
"""NAR「凡走(下位着順)」予測モデルの拡張検証(2026-08-25)。

前回(nar_search_bottom_2026_08_24.py、K=4固定「4着以内に入らない」、全4セル不採用)に対し、
ユーザーから2点の拡張依頼があった:
  (1) K=3〜8(「3着以内に来ない」〜「8着以内に来ない」)それぞれ独立にパターンを算出する。
  (2) 既存シグナルの反転流用ではなく凡走専用シグナルを新設する
      (pace_clash_risk/layoff_return_risk/class_rank_vs_field/distance_stretch_risk)。

Opus 5システムエンジニア役サブエージェントによる批判的レビュー(実データ検証込み)を経て、
以下の設計で実施する:
  - box_n固定・スコアベース選出のため、隣接Kの結果(picks)がほぼ同一になり、K=3〜8は
    見かけ上6個の独立検定に見えて実効的にはもっと少ない(cross-K相関として実測・報告する)。
    → K=3(ユーザーが述べた将来応用「3着以内に来ない馬の除外」と直接対応)を単一の
    事前登録主要仮説(confirmatory)とし、K=4〜8は探索的(exploratory)スイープとして
    明確に区別する。
  - ゲート定義は事前に固定: true_edge_pt>0 かつ win_rate>=90% かつ
    グループK分割OOF picksの市場比ペア差分ブートストラップCI下限>0。
    (前回スクリプトのtrust_oof=fold_argmaxユニーク数>=3は138ブロックのleave-one-block-out
    では構造的にほぼ通過不能だったため、held-out比率が実際に意味を持つグループK分割OOFを
    新設しゲートをこちらに一本化する。Nested LOBO OOFは参考値として併記する)。
  - 依存する3つのサイドカーファイル(nar_bottom_dataset.py/nar_bottom_eval.py/
    nar_bottom_signals.py)への変更はすべてデフォルト引数付きの後方互換な追加のみで、
    nar_search_bottom_2026_08_24.py 自体は無改造。本スクリプト完了後にそちらを再実行し、
    数値が完全一致することを回帰テストとして確認する(このスクリプトの外側で実施)。
  - build_matricesはKに依存しない(Kはどのレースを残すか・ラベルだけを変える)ため、
    Track×系統ごとに1回(計4回)だけ呼び、K=3〜8はBE.label_and_filter()のインデックスで
    スライスして使い回す(24回の重い呼び出し→4回に短縮)。

出力: scratchpad配下のJSON/TXT(研究ログ)。本番data/nar_pipeline/配下・共有キャッシュ
(nar_dataset_cache.pkl)には一切書き込まない。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
sys.path.insert(0, str(LIB_DIR))
import nar_bottom_dataset as BD  # noqa: E402
import nar_bottom_eval as BE  # noqa: E402
import nar_bottom_signals as BS  # noqa: E402
import nar_signals as NS  # noqa: E402
import verify_provenance as VP  # noqa: E402

OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
OUT_JSON = OUT_DIR / "nar_search_bottom_k_sweep_2026_08_25_result.json"
OUT_TXT = OUT_DIR / "nar_search_bottom_k_sweep_2026_08_25_report.txt"

N_PATTERNS = 200
SEED = 2825
BOX_N = 4
K_PRIMARY = 3
K_RANGE = list(range(3, 9))  # 3..8
VIF_THRESHOLD = 0.45
SIGN_RHO_THRESHOLD = 0.02
WEIGHT_TIERS = [
    (100.0, 60),  # 等重みのごく近傍
    (25.0, 60),   # 中程度の偏差
    (6.0, 50),    # やや広い探索
    (1.0, 29),    # 旧来のDirichlet(1)一様探索(比較用に少数だけ残す)
]
assert 1 + sum(n for _, n in WEIGHT_TIERS) == N_PATTERNS

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


# =====================================================================
# Phase 0: 診断
# =====================================================================
log("=" * 72)
log("Phase 0: 診断")
log("=" * 72)

data = BD.build(verbose=False, min_field_size=2)  # K=8まで対応する最も広い母集団
races_a = BD.track_a(data)
races_b = BD.track_b(data)
log(f"Track A(全期間) races: {len(races_a)}")
log(f"Track B(2026-07-25以降) races: {len(races_b)}")
log(f"skipped: {len(data['skipped'])}")

priors_all = NS.make_priors(races_a)
dead = NS.detect_dead(races_a, priors_all)
alive_base = [n for n in NS.ALL_SIGNALS if n not in dead]
log(f"\n死にシグナル({len(dead)}): {dead}")
log(f"既存プール(反転流用、{len(alive_base)}本): {alive_base}")

priors_bottom = BS.make_priors_bottom(races_b)
POOL_NEW_TRACK_A_CAND = BS.NEW_SIGNALS_TRACK_A + BS.NEW_SIGNALS_TRACK_A_BOTTOM_V2
POOL_NEW_TRACK_B_CAND = BS.NEW_SIGNALS_ALL_BOTTOM_V2
dead_bottom_a = BS.detect_dead_bottom(races_a, priors_bottom, names=POOL_NEW_TRACK_A_CAND)
dead_bottom_b = BS.detect_dead_bottom(races_b, priors_bottom, names=POOL_NEW_TRACK_B_CAND)
log(f"\n新規候補シグナル({len(POOL_NEW_TRACK_A_CAND)}本、Track A対象)死亡チェック: {dead_bottom_a}")
log(f"新規候補シグナル({len(POOL_NEW_TRACK_B_CAND)}本、Track B対象)死亡チェック: {dead_bottom_b}")


# --- 新規4本シグナルの充足率・分散診断
def _raw_new_signal_series(entries, name):
    vals = []
    for e in entries:
        cc = NS.class_ordinal(e["race_name"])
        sig = BS.build_bottom_signals(e["df"], cc, priors_bottom, track_b=e.get("track_b", False),
                                      kaisai_date=e.get("kaisai_date"))
        vals.append(sig[name])
    return pd.concat(vals, ignore_index=True)


def _raw_base_signal_series(entries, name):
    vals = []
    for e in entries:
        cc = NS.class_ordinal(e["race_name"])
        sig = NS.build_signals(e["df"], cc, priors_all)
        vals.append(sig[name])
    return pd.concat(vals, ignore_index=True)


log("\n[新規4本シグナルの診断]")
for name, ents, tlabel in [
    ("pace_clash_risk", races_a, "TrackA"), ("layoff_return_risk", races_a, "TrackA"),
    ("class_rank_vs_field", races_a, "TrackA"), ("distance_stretch_risk", races_b, "TrackB"),
]:
    s = _raw_new_signal_series(ents, name)
    fill = s.notna().mean() * 100
    log(f"  {name}({tlabel}, N={len(s)}): 充足率={fill:.1f}%  "
        f"mean={s.mean():.3f} std={s.std():.3f} min={s.min():.3f} max={s.max():.3f}")

# --- past1_dateフォーマット確認、SCALE定数の実測分布(既にnar_bottom_signals.pyへ確定値を
# 埋め込み済み。ここでは分布を再測定し、採用値との整合を記録する)
past1_raw = pd.concat([NS._col(r["df"], "past1_date") for r in races_a], ignore_index=True)
past1_dt = pd.to_datetime(past1_raw, format="%Y.%m.%d", errors="coerce")
fmt_ok = past1_dt.notna().mean() * 100
n_blank = (past1_raw.isna() | (past1_raw.astype(str).str.strip() == "")).mean() * 100
log(f"\npast1_dateフォーマット: \"%Y.%m.%d\"解釈成功率={fmt_ok:.1f}%  空欄={n_blank:.1f}%")

kaisai_series = pd.concat(
    [pd.Series([r["kaisai_date"]] * r["field_size"]) for r in races_a], ignore_index=True
)
kd_dt = pd.to_datetime(kaisai_series, format="%Y%m%d", errors="coerce")
gap = (kd_dt - past1_dt).dt.days
gap_valid = gap[gap > 0].dropna()
log(f"gap_days分布(N={len(gap_valid)}): p50={gap_valid.quantile(.5):.0f} p90={gap_valid.quantile(.9):.0f} "
    f"p99={gap_valid.quantile(.99):.0f} max={gap_valid.max():.0f}  "
    f"→ LAYOFF_SCALE_DAYS={BS.LAYOFF_SCALE_DAYS}(採用値、nar_bottom_signals.pyに実装済み)")
log(f"gap_days<=0(異常値・NaN化対象): {int((gap <= 0).sum())}件 / {gap.notna().sum()}件中")

runs_b = pd.concat(
    [NS._num(NS._col(r["df"], "data_distance_slot1_runs")) for r in races_b], ignore_index=True
).dropna()
log(f"同距離経験本数(data_distance_slot1_runs)分布(TrackB, N={len(runs_b)}): "
    f"p50={runs_b.quantile(.5):.0f} p90={runs_b.quantile(.9):.0f} p99={runs_b.quantile(.99):.0f} "
    f"max={runs_b.max():.0f}  → DISTANCE_STRETCH_SCALE_RUNS={BS.DISTANCE_STRETCH_SCALE_RUNS}(採用値)")


def _mean_class_1to3(df):
    cs = []
    for i in (1, 2, 3):
        c = NS._col(df, f"past{i}_race_name").map(NS.class_ordinal)
        c = c.fillna(NS._col(df, f"past{i}_race_class").map(NS.class_ordinal))
        cs.append(c)
    return pd.concat(cs, axis=1).mean(axis=1, skipna=True)


diffs = []
allnan_races = 0
for r in races_a:
    own = _mean_class_1to3(r["df"])
    fld = own.mean(skipna=True)
    if pd.isna(fld):
        allnan_races += 1
        continue
    diffs.append(own - fld)
diff_all = pd.concat(diffs, ignore_index=True).dropna() if diffs else pd.Series([], dtype=float)
log(f"class_rank_vs_field 差分分布(TrackA, N={len(diff_all)}): "
    f"p10={diff_all.quantile(.1):.2f} p50={diff_all.quantile(.5):.2f} p90={diff_all.quantile(.9):.2f}  "
    f"全NaNレース={allnan_races}/{len(races_a)}({allnan_races / len(races_a) * 100:.1f}%)  "
    f"→ CLASS_RANK_SCALE={BS.CLASS_RANK_SCALE}(採用値)")

# --- pace_clash_riskのVIF/相関ゲート(既存style/nigeとの相関、閾値超なら自動除外)
pcr = _raw_new_signal_series(races_a, "pace_clash_risk")
style_s = _raw_base_signal_series(races_a, "style")
nige_s = _raw_base_signal_series(races_a, "nige")
corr_style = float(pcr.corr(style_s, method="spearman"))
corr_nige = float(pcr.corr(nige_s, method="spearman"))
pinned_rate = float((pcr == pcr.mode().iloc[0]).mean() * 100) if pcr.notna().any() else float("nan")
log(f"\n[pace_clash_risk VIFゲート] style相関={corr_style:+.3f}  nige相関={corr_nige:+.3f}  "
    f"最頻値への張り付き率={pinned_rate:.1f}%  閾値=|rho|>={VIF_THRESHOLD}")
pace_clash_excluded = (abs(corr_style) >= VIF_THRESHOLD) or (abs(corr_nige) >= VIF_THRESHOLD)
log(f"  判定: {'自動除外(プールから外す)' if pace_clash_excluded else '許容(プールに残す)'}")
excluded_by_vif = {"pace_clash_risk"} if pace_clash_excluded else set()

POOL_2A = alive_base + [n for n in POOL_NEW_TRACK_A_CAND if n not in dead_bottom_a and n not in excluded_by_vif]
POOL_2B = alive_base + [n for n in POOL_NEW_TRACK_B_CAND if n not in dead_bottom_b and n not in excluded_by_vif]
POOL_3A = POOL_2A + ["ninki"]
POOL_3B = POOL_2B + ["ninki"]
log(f"\nVIF除外後: 系統2 TrackA={len(POOL_2A)}本  系統2 TrackB={len(POOL_2B)}本  "
    f"系統3 TrackA={len(POOL_3A)}本  系統3 TrackB={len(POOL_3B)}本")


def build_matrices(entries: list, names: list) -> list:
    """既存(反転流用)+新規候補+ninki を1つの(S, A)行列に統合する。Kに依存しない
    (レース選択・ラベルはK=3〜8ごとにBE.label_and_filter()で後から絞る)。"""
    mats = []
    for e in entries:
        current_class = NS.class_ordinal(e["race_name"])
        base_sig = NS.build_signals(e["df"], current_class, priors_all)
        new_sig = BS.build_bottom_signals(e["df"], current_class, priors_bottom,
                                          track_b=e.get("track_b", False), kaisai_date=e.get("kaisai_date"))
        ninki = pd.to_numeric(e["df"]["bias_ninki"], errors="coerce")
        merged = {**base_sig, **new_sig, "ninki": NS._minmax(-ninki)}
        cols = [merged[n].to_numpy(dtype=float) for n in names]
        M = np.column_stack(cols)
        A = (~np.isnan(M)).astype(float)
        S = np.nan_to_num(M, nan=0.0)
        mats.append({"S": S, "A": A})
    return mats


# --- 符号規約チェック(K=3ラベル基準、自動除外ルール付き)
sub_races_k3, sub_labels_k3, sub_idx_k3 = BE.label_and_filter(races_b, K_PRIMARY, BOX_N)
mats_signcheck_full = build_matrices(races_b, POOL_3B)
mats_signcheck = [mats_signcheck_full[i] for i in sub_idx_k3]
sign_check = BE.signal_label_correlation(sub_races_k3, mats_signcheck, POOL_3B, labels=sub_labels_k3)
log(f"\n符号規約チェック(K={K_PRIMARY}ラベル基準、spearman相関。規約が正しければ負のはず):")
auto_excluded_sign = []
for n, rho in sign_check.items():
    if rho is not None and rho > SIGN_RHO_THRESHOLD:
        auto_excluded_sign.append(n)
    log(f"  {n}: {rho:+.3f}" if rho is not None else f"  {n}: N/A(標本不足)")
log(f"  自動除外(符号疑義、正相関かつ|rho|>{SIGN_RHO_THRESHOLD}): "
    f"{auto_excluded_sign if auto_excluded_sign else 'なし'}")
for excl in auto_excluded_sign:
    for pool in (POOL_2A, POOL_2B, POOL_3A, POOL_3B):
        if excl in pool:
            pool.remove(excl)
log(f"符号除外後 最終プール: 系統2 TrackA={len(POOL_2A)}本  系統2 TrackB={len(POOL_2B)}本  "
    f"系統3 TrackA={len(POOL_3A)}本  系統3 TrackB={len(POOL_3B)}本")
log(f"  系統2 TrackA内訳: {POOL_2A}")
log(f"  系統2 TrackB内訳: {POOL_2B}")

# --- provenance(市場ベースラインの取得タイミング非対称性、参考値)
try:
    prov = VP.build_provenance_table()
    verdict_map = dict(zip(prov["race_id"], prov["verdict"]))
    counts = {"pre_race": 0, "post_race": 0, "unrecorded": 0}
    for r in races_a:
        v = verdict_map.get(r["race_id"], "unrecorded")
        if v not in counts:
            v = "unrecorded"
        counts[v] += 1
    log(f"\nprovenance内訳(母集団N={len(races_a)}): {counts}")
except Exception as exc:  # noqa: BLE001
    counts = None
    log(f"\nprovenance確認をスキップ(理由: {exc})")


# --- K別 reality-check テーブル(box_n=4固定、ランダム/市場/オラクル上限/伸びしろ)
def oracle_precision(sub_labels: list, box_n: int) -> float:
    stake_total, hit_total = 0, 0
    for lab in sub_labels:
        n = len(lab)
        k_sel = min(box_n, n)
        pos = int(np.nansum(lab))
        stake_total += k_sel
        hit_total += min(pos, k_sel)
    return hit_total / stake_total * 100 if stake_total else 0.0


def random_baseline(sub_labels: list, box_n: int) -> float:
    stake_total, hit_total = 0.0, 0.0
    for lab in sub_labels:
        n = len(lab)
        k_sel = min(box_n, n)
        pos = float(np.nansum(lab))
        stake_total += k_sel
        hit_total += k_sel * pos / n if n else 0.0
    return hit_total / stake_total * 100 if stake_total else 0.0


log(f"\n[事前確率チェック] box_n={BOX_N}固定、K別のランダム/市場/オラクル上限/伸びしろ:")
reality_check = {"A": {}, "B": {}}
for track_key, entries_full in (("A", races_a), ("B", races_b)):
    for k in K_RANGE:
        sub_races, sub_labels, _ = BE.label_and_filter(entries_full, k, BOX_N)
        ev_rc = BE.Evaluator(sub_races, BOX_N, labels=sub_labels)
        mkt_rc = ev_rc.evaluate(ev_rc.mkt_picks)["model"]
        rnd = random_baseline(sub_labels, BOX_N)
        orc = oracle_precision(sub_labels, BOX_N)
        pos_rate = float(np.mean([np.nanmean(l) for l in sub_labels])) * 100
        reality_check[track_key][k] = {
            "n_races": len(sub_races), "pos_rate_pct": pos_rate,
            "random": rnd, "market": mkt_rc, "oracle": orc, "headroom_pt": orc - mkt_rc,
        }
        log(f"  Track{track_key} K={k}: N={len(sub_races)}"
            f"({len(sub_races) / len(entries_full) * 100:.0f}%)  正例率={pos_rate:.1f}%  "
            f"ランダム={rnd:.2f}%  市場={mkt_rc:.2f}%  オラクル上限={orc:.2f}%  "
            f"伸びしろ={orc - mkt_rc:+.2f}pt")


# =====================================================================
# 本探索: 系統2/系統3 × Track A/B × K=3〜8(box_nは4固定)
# =====================================================================
def equal_w(pool: list) -> np.ndarray:
    return np.array([1.0 / len(pool)] * len(pool))


def make_weight_patterns(pool: list, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = [equal_w(pool)]
    for concentration, n in WEIGHT_TIERS:
        alpha = [concentration] * len(pool)
        for _ in range(n):
            cols.append(rng.dirichlet(alpha))
    W = np.column_stack(cols)
    assert W.shape[1] == N_PATTERNS
    return W


def run_cell(label: str, entries_full: list, mats_full: list, pool: list, W: np.ndarray,
            k: int, box_n: int, seed: int, confirmatory: bool):
    log(f"\n--- {label}  K={k}(box_n={box_n}, プール{len(pool)}本, "
        f"{'confirmatory' if confirmatory else 'exploratory'}) ---")
    sub_races, sub_labels, sub_idx = BE.label_and_filter(entries_full, k, box_n)
    sub_mats = [mats_full[i] for i in sub_idx]
    ev = BE.Evaluator(sub_races, box_n, labels=sub_labels)
    mkt = ev.evaluate(ev.mkt_picks)
    log(f"N={len(sub_races)}  市場precision={mkt['model']:.2f}%")

    eq_picks = BE.score_picks_bottom(sub_mats, equal_w(pool), box_n)
    r_eq = ev.evaluate(eq_picks)
    log(f"等重み: precision={r_eq['model']:.2f}%  市場差={r_eq['excess']:+.2f}pt")

    all_picks = [BE.score_picks_bottom(sub_mats, W[:, j], box_n) for j in range(N_PATTERNS)]
    all_st, all_rt = [], []
    for p in all_picks:
        s, r = ev.hits_for(p)
        all_st.append(s)
        all_rt.append(r)
    full_vals = np.array([ev.precision(all_st[j], all_rt[j]) for j in range(N_PATTERNS)])
    best_full = int(np.argmax(full_vals))
    top_w = {n: float(w) for n, w in zip(pool, W[:, best_full]) if w > 0.005}
    log(f"[in-sample最良](参考) pattern#{best_full}  precision={full_vals[best_full]:.2f}%"
        f"(市場差={full_vals[best_full] - mkt['model']:+.2f}pt)")

    def fit_fn(train_idx, all_st=all_st, all_rt=all_rt):
        vals = np.array([ev.precision(all_st[j], all_rt[j], idx=train_idx) for j in range(N_PATTERNS)])
        chosen = int(np.argmax(vals))
        return W[:, chosen], chosen

    lobo = ev.lobo_oof(fit_fn, sub_mats)
    gkf = ev.group_kfold_oof(fit_fn, sub_mats, n_folds=8, seed=seed + 3)
    log(f"[Nested LOBO OOF]({len(ev.block_ids)}ブロック、参考値) precision={lobo['model']:.2f}%  "
        f"市場差={lobo['excess']:+.2f}pt  fold argmaxユニーク数={lobo['fold_argmax_unique']}/{len(ev.block_ids)}")
    log(f"[グループK分割OOF](n_folds={gkf['n_folds']}、本命) precision={gkf['model']:.2f}%  "
        f"市場差={gkf['excess']:+.2f}pt  fold argmaxユニーク数={gkf['fold_argmax_unique']}/{gkf['n_folds']}")

    opt = BE.selection_optimism(ev, sub_mats, W, n_rep=200, seed=seed + 7)
    log(f"[選択バイアス診断] true_edge_pt={opt['true_edge_pt']:+.2f}pt(sd{opt['true_edge_sd']:.2f})  "
        f"win_rate={opt['win_rate'] * 100:.0f}%")

    boot_vs_market = ev.paired_block_bootstrap(gkf["picks"], ev.mkt_picks, seed=seed + 11)
    log(f"[グループK分割OOF picksの市場比ペア差分ブートストラップCI, n=2000] "
        f"95%CI[{boot_vs_market['lo']:+.2f}, {boot_vs_market['hi']:+.2f}]pt")

    recall = ev.recall_and_clean(gkf["picks"])
    log(f"  参考指標: recall={recall['recall_pct']:.1f}%  clean_box_rate={recall['clean_box_rate_pct']:.1f}%")

    gate = None
    if confirmatory:
        gate = bool(opt["true_edge_pt"] > 0 and opt["win_rate"] >= 0.90 and boot_vs_market["lo"] > 0)
        log(f"  事前登録ゲート(true_edge_pt>0 かつ win_rate>=90% かつ 市場比ペア差分CI下限>0): "
            f"{'YES(採用検討の1条件を満たす)' if gate else 'NO'}")

    result = {
        "k": k, "n_races": len(sub_races), "pool_size": len(pool),
        "market": mkt["model"],
        "equal_weight": {"model": r_eq["model"], "excess": r_eq["excess"]},
        "best_full_population": {
            "pattern_index": best_full, "model": float(full_vals[best_full]),
            "excess": float(full_vals[best_full] - mkt["model"]), "weights": top_w,
        },
        "nested_lobo_oof": {"model": lobo["model"], "excess": lobo["excess"],
                            "fold_argmax_unique": lobo["fold_argmax_unique"], "n_blocks": len(ev.block_ids)},
        "group_kfold_oof": {"model": gkf["model"], "excess": gkf["excess"],
                            "fold_argmax_unique": gkf["fold_argmax_unique"], "n_folds": gkf["n_folds"]},
        "selection_optimism": opt,
        "bootstrap_vs_market_paired": boot_vs_market,
        "recall_and_clean": recall,
        "confirmatory": confirmatory,
        "gate": gate,
    }
    return result, full_vals


results = {}
cross_k_full_vals = {}
TRACKS = [("Track A(全期間)", races_a, POOL_2A, POOL_3A, "A"),
          ("Track B(2026-07-25以降)", races_b, POOL_2B, POOL_3B, "B")]

for track_label, entries_full, pool2, pool3, track_key in TRACKS:
    log("\n" + "=" * 72)
    log(f"{track_label}  (N={len(entries_full)})")
    log("=" * 72)
    results[track_key] = {"n_races": len(entries_full), "by_k": {}}

    # --- build_matricesの再利用最適化: Track×系統ごとに1回だけ(Kに依存しない)
    mats2 = build_matrices(entries_full, pool2)
    mats3 = build_matrices(entries_full, pool3)
    # --- 重みパターン行列もTrack×系統ごとに1回だけ生成し、全K間で同一パターンを使い回す
    # (cross-K相関診断: パターンjが指すシグナル重みがK間で完全に同一であることを保証する)
    W2 = make_weight_patterns(pool2, SEED + (0 if track_key == "A" else 100))
    W3 = make_weight_patterns(pool3, SEED + (1000 if track_key == "A" else 1100))

    for k in K_RANGE:
        confirmatory = (k == K_PRIMARY)
        r2, fv2 = run_cell(f"{track_label} 系統2(oddsなし)", entries_full, mats2, pool2, W2,
                           k, BOX_N, SEED + k, confirmatory)
        r3, fv3 = run_cell(f"{track_label} 系統3(odds込み)", entries_full, mats3, pool3, W3,
                           k, BOX_N, SEED + 500 + k, confirmatory)
        results[track_key]["by_k"][k] = {"system2": r2, "system3": r3}
        cross_k_full_vals[(track_key, "system2", k)] = fv2
        cross_k_full_vals[(track_key, "system3", k)] = fv3

    # --- 感度分析(K=3・系統3のみ、box_n=4): walk-forward / 馬ID group-split / 天井効果
    sub_races3, sub_labels3, sub_idx3 = BE.label_and_filter(entries_full, K_PRIMARY, BOX_N)
    sub_mats3 = [mats3[i] for i in sub_idx3]

    train_idx, test_idx, mid_date = BE.walk_forward_split(sub_races3)
    if len(train_idx) >= 10 and len(test_idx) >= 10:
        ev_wf = BE.Evaluator(sub_races3, BOX_N, labels=sub_labels3)
        picks_wf = [BE.score_picks_bottom(sub_mats3, W3[:, j], BOX_N) for j in range(N_PATTERNS)]
        st_wf, rt_wf = [], []
        for p in picks_wf:
            s, r = ev_wf.hits_for(p)
            st_wf.append(s)
            rt_wf.append(r)
        vals_train = np.array([ev_wf.precision(st_wf[j], rt_wf[j], idx=train_idx) for j in range(N_PATTERNS)])
        best_train = int(np.argmax(vals_train))
        excess_train = vals_train[best_train] - ev_wf.evaluate(ev_wf.mkt_picks, idx=train_idx)["model"]
        val_test = ev_wf.precision(st_wf[best_train], rt_wf[best_train], idx=test_idx)
        mkt_test = ev_wf.evaluate(ev_wf.mkt_picks, idx=test_idx)["model"]
        excess_test = val_test - mkt_test
        log(f"\n[walk-forward](K={K_PRIMARY}、系統3、{track_label}、分割日={mid_date}) "
            f"train={excess_train:+.2f}pt→test={excess_test:+.2f}pt  "
            f"符号一致={'YES' if (excess_train > 0) == (excess_test > 0) else 'NO'}")
        results[track_key]["walk_forward_k3"] = {
            "mid_date": mid_date, "excess_train": float(excess_train), "excess_test": float(excess_test),
            "sign_agree": bool((excess_train > 0) == (excess_test > 0)),
        }
    else:
        log(f"\n[walk-forward] {track_label}: train/testいずれかが小さすぎるためスキップ")
        results[track_key]["walk_forward_k3"] = None

    idx_a, idx_b = BE.horse_group_split(sub_races3, seed=SEED + 600)
    ev_gs = BE.Evaluator(sub_races3, BOX_N, labels=sub_labels3)
    picks_gs = [BE.score_picks_bottom(sub_mats3, W3[:, j], BOX_N) for j in range(N_PATTERNS)]
    st_gs, rt_gs = [], []
    for p in picks_gs:
        s, r = ev_gs.hits_for(p)
        st_gs.append(s)
        rt_gs.append(r)
    vals_a = np.array([ev_gs.precision(st_gs[j], rt_gs[j], idx=idx_a) for j in range(N_PATTERNS)])
    best_a = int(np.argmax(vals_a))
    val_b_of_best_a = ev_gs.precision(st_gs[best_a], rt_gs[best_a], idx=idx_b)
    mean_b = float(np.mean([ev_gs.precision(st_gs[j], rt_gs[j], idx=idx_b) for j in range(N_PATTERNS)]))
    log(f"\n[馬ID group-split感度分析](K={K_PRIMARY}、系統3、{track_label}) "
        f"グループAで選んだパターンのグループBでの成績={val_b_of_best_a:.2f}%  "
        f"グループB全パターン平均={mean_b:.2f}%  差={val_b_of_best_a - mean_b:+.2f}pt")
    results[track_key]["horse_group_split_k3"] = {
        "selected_on_a_eval_on_b": val_b_of_best_a, "unseen_b_mean": mean_b,
        "true_edge_pt": val_b_of_best_a - mean_b,
    }

    entries_ceiling = [r for r in sub_races3 if r["field_size"] >= BOX_N + 4]
    if len(entries_ceiling) >= 20:
        ceil_idx = [i for i, r in enumerate(sub_races3) if r["field_size"] >= BOX_N + 4]
        ev_c = BE.Evaluator(entries_ceiling, BOX_N, labels=[sub_labels3[i] for i in ceil_idx])
        mkt_c = ev_c.evaluate(ev_c.mkt_picks)
        mats_c = [sub_mats3[i] for i in ceil_idx]
        eq_c = ev_c.evaluate(BE.score_picks_bottom(mats_c, equal_w(pool3), BOX_N))
        log(f"\n[天井効果の副分析](K={K_PRIMARY}、field_size>={BOX_N + 4}、"
            f"N={len(entries_ceiling)}/{len(sub_races3)}) 市場={mkt_c['model']:.2f}%  "
            f"系統3等重み={eq_c['model']:.2f}%(市場差={eq_c['excess']:+.2f}pt)")
        results[track_key]["ceiling_subpop_k3"] = {
            "n_races": len(entries_ceiling), "market": mkt_c["model"],
            "system3_equal_weight_excess": eq_c["excess"],
        }
    else:
        results[track_key]["ceiling_subpop_k3"] = None


# =====================================================================
# cross-K相関診断(多重比較の実効検定数の根拠)
# =====================================================================
log("\n" + "=" * 72)
log("cross-K相関診断(隣接K間、多重比較の実効検定数の根拠)")
log("=" * 72)
cross_k_report = {}
all_adjacent_r = []
for track_key in ("A", "B"):
    for sys_key in ("system2", "system3"):
        vals_by_k = {k: cross_k_full_vals[(track_key, sys_key, k)] for k in K_RANGE}
        corrs = []
        for k in K_RANGE[:-1]:
            r = float(np.corrcoef(vals_by_k[k], vals_by_k[k + 1])[0, 1])
            corrs.append(r)
            all_adjacent_r.append(r)
            log(f"  Track{track_key}/{sys_key}: K={k}⇔K={k + 1}  pearson r={r:+.3f}")
        cross_k_report[f"{track_key}_{sys_key}"] = corrs
mean_adjacent_r = float(np.mean(all_adjacent_r))
log(f"\n全隣接ペア平均相関: {mean_adjacent_r:+.3f}"
    f"(SEレビューの事前実測値0.75〜0.84と比較。picksがbox_n固定・スコアベース選出のため"
    f"K間でほぼ同一になることの実データでの再確認。K=3〜8は見かけ上6個の独立検定に見えて"
    f"実効的にはもっと少ないことの根拠として、K=3を単一の事前登録主要仮説とする設計を採用した)")


# =====================================================================
# まとめ
# =====================================================================
log("\n" + "=" * 72)
log("まとめ")
log("=" * 72)
log(f"\n[K={K_PRIMARY}(主要仮説、confirmatory)]")
for track_key, track_label in (("A", "Track A(全期間)"), ("B", "Track B(2026-07-25以降)")):
    cell = results[track_key]["by_k"][K_PRIMARY]
    for sys_key, sys_label in (("system2", "系統2(oddsなし)"), ("system3", "系統3(odds込み)")):
        s = cell[sys_key]
        boot = s["bootstrap_vs_market_paired"]
        log(f"{track_label} / {sys_label}: 市場={s['market']:.2f}%  "
            f"グループK分割OOF市場差={s['group_kfold_oof']['excess']:+.2f}pt  "
            f"true_edge_pt={s['selection_optimism']['true_edge_pt']:+.2f}pt  "
            f"win_rate={s['selection_optimism']['win_rate'] * 100:.0f}%  "
            f"市場比ペア差分CI=[{boot['lo']:+.2f}, {boot['hi']:+.2f}]pt  "
            f"ゲート={'YES' if s['gate'] else 'NO'}")
    wf = results[track_key].get("walk_forward_k3")
    if wf:
        log(f"  walk-forward符号一致: {'YES' if wf['sign_agree'] else 'NO'}"
            f"(train={wf['excess_train']:+.2f}pt→test={wf['excess_test']:+.2f}pt)")

log(f"\n最終判定(K={K_PRIMARY}、事前登録ルール: 系統2または系統3で ゲートYES かつ "
    "walk-forward符号一致 の場合のみ「採用検討」):")
verdicts = {}
for track_key, track_label in (("A", "Track A(全期間)"), ("B", "Track B(2026-07-25以降)")):
    cell = results[track_key]["by_k"][K_PRIMARY]
    wf = results[track_key].get("walk_forward_k3")
    wf_ok = bool(wf and wf["sign_agree"]) if wf else False
    any_gate = any(cell[sk]["gate"] for sk in ("system2", "system3"))
    verdicts[track_key] = "採用検討" if (any_gate and wf_ok) else "不採用"
    log(f"  {track_label}: {verdicts[track_key]}")

log(f"\n[K=4〜8(探索的スイープ、参考値。個別のゲート判定は行わない)]")
for k in K_RANGE:
    if k == K_PRIMARY:
        continue
    for track_key, track_label in (("A", "Track A"), ("B", "Track B")):
        cell = results[track_key]["by_k"][k]
        for sys_key, sys_label in (("system2", "系統2"), ("system3", "系統3")):
            s = cell[sys_key]
            log(f"  {track_label}/{sys_label} K={k}: 市場={s['market']:.2f}%  "
                f"グループK分割OOF市場差={s['group_kfold_oof']['excess']:+.2f}pt  "
                f"true_edge_pt={s['selection_optimism']['true_edge_pt']:+.2f}pt  "
                f"win_rate={s['selection_optimism']['win_rate'] * 100:.0f}%")

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps({
    "n_races_track_a": len(races_a), "n_races_track_b": len(races_b),
    "dead_signals": dead, "pool_2a": POOL_2A, "pool_2b": POOL_2B,
    "n_patterns": N_PATTERNS, "seed": SEED, "weight_tiers": WEIGHT_TIERS,
    "k_primary": K_PRIMARY, "k_range": K_RANGE,
    "pace_clash_vif": {"corr_style": corr_style, "corr_nige": corr_nige,
                       "pinned_rate_pct": pinned_rate, "excluded": pace_clash_excluded},
    "sign_check_k3": sign_check, "auto_excluded_sign": auto_excluded_sign,
    "reality_check": reality_check, "cross_k_correlation": cross_k_report,
    "mean_adjacent_cross_k_r": mean_adjacent_r, "verdicts_k_primary": verdicts,
    "provenance_counts": counts,
    "results": results,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
