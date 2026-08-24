# -*- coding: utf-8 -*-
"""NAR「5位以下(4着以内に入らない)」予測モデルの検証(2026-08-24)。

ユーザー依頼: 「NARで、取得したデータを生かして、5位以下になる確率を最大にしたパターンの
検証とレポートをお願いします。現在の、地方競馬 馬柱データによる着順予想とは別のレポートを
お願いします。できるだけ取得したデータを有効に利用したものにしてください。新馬戦を除く、
今日までの1504レースを対象とします。」

Opus 5サブエージェントによる批判的レビュー(実データ検証込み)を経て、以下の設計で実施する:
  - 母集団を2層に分割: Track A(全期間)/ Track B(2026-07-25以降、newspaperのdata_*系
    スキーマが100%充足するサブ集団)。スキーマがこの日付で完全分断しているため、混ぜてpoolすると
    異なる構造のモデルを1つの数値に平均する誤りになる。
  - 主指標は selection_optimism の true_edge_pt(前回NAR探索8/20の教訓: 保留率の大きい
    選択バイアス診断を優先する)。Nested LOBO OOFは併記するが fold別argmaxユニーク数を
    必ず確認する(2以下ならin-sample評価と実質同値)。
  - box_nは4を主軸として統計的検定(Nested LOBO OOF・selection_optimism・ペア差分
    ブロックブートストラップ)を行う。3/5はin-sample参考値のみ(検定しない)。
  - 3系統(系統1=市場のみ/系統2=新規・既存データのみ/系統3=系統2+ninki統合)のうち、
    系統1 vs 系統3 のペア差分ブートストラップを増分検定の本命とする。系統2単独の市場対比は
    参考値として併記する(「市場情報の上に取得済みデータが増分価値を持つか」を直接測るため)。
  - Dirichletパターン数は200(多重比較の圧縮、前回500から削減)。

事前登録した採否ルール(実行前に固定): 系統2または系統3のNested LOBO OOF picksが、市場との
ペア差分ブロックブートストラップ95%CI下限で0を超え、かつfold別argmaxユニーク数が3以上、かつ
walk-forward検証(Track Bのみ)でも市場超過の符号が一致する場合のみ「採用検討」とする。

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
OUT_JSON = OUT_DIR / "nar_search_bottom_2026_08_24_result.json"
OUT_TXT = OUT_DIR / "nar_search_bottom_2026_08_24_report.txt"

N_PATTERNS = 200
SEED = 2824
BOX_N_PRIMARY = 4
BOX_NS_REFERENCE = (3, 5)
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
# Phase 0: 診断(市場との差・スキーマ分断・符号チェック・重複列チェック・provenance)
# =====================================================================
log("=" * 72)
log("Phase 0: 診断")
log("=" * 72)

data = BD.build(verbose=False)
races_a = BD.track_a(data)
races_b = BD.track_b(data)
log(f"Track A(全期間、20260711〜) races: {len(races_a)}")
log(f"Track B(2026-07-25以降) races: {len(races_b)}")
log(f"skipped: {len(data['skipped'])}")
sizes = pd.Series([r["field_size"] for r in races_a])
log(f"field size: min={sizes.min()} median={sizes.median()} max={sizes.max()} mean={sizes.mean():.1f}")
pos_rate = pd.Series([r["df"]["label_bottom"].mean() for r in races_a]).mean()
log(f"label_bottom=1 平均割合(レース内平均、全馬ランダムに近い基準): {pos_rate * 100:.1f}%")

priors_all = NS.make_priors(races_a)
dead = NS.detect_dead(races_a, priors_all)
alive_base = [n for n in NS.ALL_SIGNALS if n not in dead]
log(f"\n死にシグナル({len(dead)}): {dead}")
log(f"既存プール(反転流用、{len(alive_base)}本): {alive_base}")

priors_bottom = BS.make_priors_bottom(races_b)
dead_bottom_a = BS.detect_dead_bottom(races_a, priors_bottom, names=BS.NEW_SIGNALS_TRACK_A)
dead_bottom_b = BS.detect_dead_bottom(races_b, priors_bottom, names=BS.NEW_SIGNALS_ALL)
log(f"新規シグナル死亡チェック(Track A対象): {dead_bottom_a}")
log(f"新規シグナル死亡チェック(Track B対象): {dead_bottom_b}")

POOL_2A = alive_base + [n for n in BS.NEW_SIGNALS_TRACK_A if n not in dead_bottom_a]
POOL_2B = alive_base + [n for n in BS.NEW_SIGNALS_ALL if n not in dead_bottom_b]
POOL_3A = POOL_2A + ["ninki"]
POOL_3B = POOL_2B + ["ninki"]
log(f"\n系統2(市場を使わない、既存+新規)Track A: {len(POOL_2A)}本 {POOL_2A}")
log(f"系統2(市場を使わない、既存+新規)Track B: {len(POOL_2B)}本 {POOL_2B}")
log(f"系統3(系統2+ninki) Track A: {len(POOL_3A)}本")
log(f"系統3(系統2+ninki) Track B: {len(POOL_3B)}本")


def build_matrices(entries: list, names: list) -> list:
    """既存(反転流用)+新規候補+ninki を1つの(S, A)行列に統合する。
    符号規約は「高いほど4着以内で終わりやすい」。下位予測ではscore_picks_bottomが
    昇順argsortでこれを正しく反転する(ここでは反転しない)。"""
    mats = []
    for e in entries:
        current_class = NS.class_ordinal(e["race_name"])
        base_sig = NS.build_signals(e["df"], current_class, priors_all)
        new_sig = BS.build_bottom_signals(e["df"], current_class, priors_bottom,
                                          track_b=e.get("track_b", False))
        ninki = pd.to_numeric(e["df"]["bias_ninki"], errors="coerce")
        merged = {**base_sig, **new_sig, "ninki": NS._minmax(-ninki)}
        cols = [merged[n].to_numpy(dtype=float) for n in names]
        M = np.column_stack(cols)
        A = (~np.isnan(M)).astype(float)
        S = np.nan_to_num(M, nan=0.0)
        mats.append({"S": S, "A": A})
    return mats


# --- 符号規約チェック: label_bottomとの相関(高いほどgoodの規約なら負の相関のはず)
mats_3b_check = build_matrices(races_b, POOL_3B)
sign_check = BE.signal_label_correlation(races_b, mats_3b_check, POOL_3B)
log("\n符号規約チェック(spearman相関、label_bottom=1[5位以下]との相関。"
    "規約が正しければ負のはず。正の値は符号が逆の疑い):")
flipped = []
for n, rho in sign_check.items():
    flag = "  ← 要確認(正相関)" if (rho is not None and rho > 0.02) else ""
    if flag:
        flipped.append(n)
    log(f"  {n}: {rho:+.3f}" if rho is not None else f"  {n}: N/A(標本不足){flag}")
if flipped:
    log(f"  符号が疑わしいシグナル: {flipped}")
else:
    log("  全シグナルで符号規約と整合(有意な正相関なし)")

# --- 重複列チェック(実測、2026-08-24確認済みの再現)
sample_df = races_b[0]["df"] if races_b else None
if sample_df is not None and "data_cushion_slot1_win_rate" in sample_df.columns:
    dup_cols = ["data_cushion_slot1_win_rate", "data_cushion_slot2_win_rate",
                "data_baba_water_slot1_win_rate", "data_distance_slot5_win_rate",
                "data_course_slot4_win_rate"]
    present = [c for c in dup_cols if c in sample_df.columns]
    log(f"\n重複列チェック(全て「全成績」への構造的フォールバックのため探索対象から除外済み): {present}")

# --- provenance(市場ベースラインの取得タイミング非対称性)
try:
    prov = VP.build_provenance_table()
    prov_race_ids = set(prov["race_id"])
    verdict_map = dict(zip(prov["race_id"], prov["verdict"]))
    counts = {"pre_race": 0, "post_race": 0, "unrecorded": 0}
    for r in races_a:
        v = verdict_map.get(r["race_id"], "unrecorded")
        if v not in counts:
            v = "unrecorded"
        counts[v] += 1
    log(f"\nprovenance内訳(母集団N={len(races_a)}): {counts}")
    pre_race_ids_in_pop = {r["race_id"] for r in races_a if verdict_map.get(r["race_id"]) == "pre_race"}
    log(f"  pre_race(発走前取得)レース数: {len(pre_race_ids_in_pop)}"
        "  ※市場ベースラインの63%相当が発走後(確定人気)取得である既知の限定事項"
        "(project_nar_data_provenance_caveat)。今回はこの非対称性の解消は図らず、"
        "pre_raceサブ集団での市場基準値を参考値として後段で併記する。")
except Exception as exc:  # noqa: BLE001
    pre_race_ids_in_pop = set()
    log(f"\nprovenance確認をスキップ(理由: {exc})")


def random_baseline(entries: list, box_n: int) -> float:
    """ランダムにbox_n頭選んだ場合の期待precision(解析的、実サンプリング不要)。"""
    stake_total, hit_total = 0.0, 0.0
    for e in entries:
        n = len(e["df"])
        k = min(box_n, n)
        pos = e["df"]["label_bottom"].sum()
        stake_total += k
        hit_total += k * pos / n if n else 0.0
    return hit_total / stake_total * 100 if stake_total else 0.0


log("\n[事前確率チェック] Track A、box_n=3/4/5でのランダム/市場/既存プール等重み反転precision:")
for bn in (3, 4, 5):
    ev_check = BE.Evaluator(races_a, bn)
    mkt_check = ev_check.evaluate(ev_check.mkt_picks)
    mats_check = build_matrices(races_a, POOL_2A)
    eq_picks_check = BE.score_picks_bottom(mats_check, np.array([1.0 / len(POOL_2A)] * len(POOL_2A)), bn)
    eq_check = ev_check.evaluate(eq_picks_check)
    rnd = random_baseline(races_a, bn)
    log(f"  box_n={bn}: ランダム={rnd:.2f}%  市場={mkt_check['model']:.2f}%  "
        f"既存{len(POOL_2A)}本等重み反転={eq_check['model']:.2f}%(市場差={eq_check['excess']:+.2f}pt)")


# =====================================================================
# 本探索: 系統2/系統3 × Track A/B × box_n
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


def run_system(label: str, entries: list, pool: list, box_n: int, seed: int,
              primary: bool) -> dict:
    log(f"\n--- {label}(box_n={box_n}, プール{len(pool)}本, primary={primary}) ---")
    mats = build_matrices(entries, pool)
    ev = BE.Evaluator(entries, box_n)
    mkt = ev.evaluate(ev.mkt_picks)
    log(f"市場(人気最下位box_n頭): precision={mkt['model']:.2f}%")

    eq_picks = BE.score_picks_bottom(mats, equal_w(pool), box_n)
    r_eq = ev.evaluate(eq_picks)
    log(f"等重み: precision={r_eq['model']:.2f}%  市場差={r_eq['excess']:+.2f}pt")

    W = make_weight_patterns(pool, seed)
    all_picks = [BE.score_picks_bottom(mats, W[:, j], box_n) for j in range(N_PATTERNS)]
    all_st, all_rt = [], []
    for p in all_picks:
        s, r = ev.hits_for(p)
        all_st.append(s)
        all_rt.append(r)
    full_vals = np.array([ev.precision(all_st[j], all_rt[j]) for j in range(N_PATTERNS)])
    best_full = int(np.argmax(full_vals))
    top_w = {n: float(w) for n, w in zip(pool, W[:, best_full]) if w > 0.005}
    log(f"[in-sample最良](参考、統計的検定には使わない) pattern#{best_full}  "
        f"precision={full_vals[best_full]:.2f}%(市場差={full_vals[best_full] - mkt['model']:+.2f}pt)")
    log(f"  重み内訳(0.5%以上): {json.dumps(top_w, ensure_ascii=False)}")

    result = {
        "pool": pool, "market": mkt["model"],
        "equal_weight": {"model": r_eq["model"], "excess": r_eq["excess"]},
        "best_full_population": {
            "pattern_index": best_full, "model": float(full_vals[best_full]),
            "excess": float(full_vals[best_full] - mkt["model"]), "weights": top_w,
        },
    }
    if not primary:
        return result

    def fit_fn(train_idx, all_st=all_st, all_rt=all_rt):
        vals = np.array([ev.precision(all_st[j], all_rt[j], idx=train_idx) for j in range(N_PATTERNS)])
        chosen = int(np.argmax(vals))
        return W[:, chosen], chosen

    nested_oof = ev.lobo_oof(fit_fn, mats)
    log(f"[Nested LOBO OOF] {len(ev.block_ids)}ブロック: precision={nested_oof['model']:.2f}%  "
        f"市場差={nested_oof['excess']:+.2f}pt  "
        f"fold別argmaxユニーク数={nested_oof['fold_argmax_unique']}/{len(ev.block_ids)}")
    trust_oof = nested_oof["fold_argmax_unique"] >= 3
    if not trust_oof:
        log("  ※ユニーク数が2以下のため、Nested LOBO OOFはin-sample評価と実質同値。統計的検定には使わない。")

    opt = BE.selection_optimism(ev, mats, W, n_rep=200, seed=seed + 7)
    log(f"[選択バイアス診断] 選抜側(見た側)={opt['selected_side']:.2f}%  "
        f"未使用側(選抜重み)={opt['unseen_side']:.2f}%  未使用側{N_PATTERNS}パターン平均={opt['unseen_all_mean']:.2f}%")
    log(f"  楽観バイアス={opt['optimism_pt']:+.2f}pt  選ぶことの真の価値(true_edge_pt)={opt['true_edge_pt']:+.2f}pt"
        f"(sd{opt['true_edge_sd']:.2f})  未使用側で平均を上回る確率={opt['win_rate'] * 100:.0f}%")

    boot_vs_market = ev.paired_block_bootstrap(nested_oof["picks"], ev.mkt_picks, seed=seed + 11)
    log(f"[Nested LOBO OOF picks の市場比ペア差分ブートストラップCI, n=2000] "
        f"95%CI[{boot_vs_market['lo']:+.2f}, {boot_vs_market['hi']:+.2f}]pt")

    recall = ev.recall_and_clean(nested_oof["picks"])
    log(f"  参考指標: recall={recall['recall_pct']:.1f}%  clean_box_rate={recall['clean_box_rate_pct']:.1f}%")

    gate = trust_oof and boot_vs_market["lo"] > 0
    log(f"  事前登録ゲート(fold argmaxユニーク数>=3 かつ 市場比ペア差分CI下限>0): "
        f"{'YES(採用検討の1条件を満たす)' if gate else 'NO'}")

    result.update({
        "nested_lobo_oof": {"model": nested_oof["model"], "excess": nested_oof["excess"],
                            "fold_argmax_unique": nested_oof["fold_argmax_unique"],
                            "n_blocks": len(ev.block_ids), "trust_oof": trust_oof},
        "selection_optimism": opt,
        "bootstrap_vs_market_paired": boot_vs_market,
        "recall_and_clean": recall,
        "gate_partial": bool(gate),
        "_nested_oof_picks": nested_oof["picks"],
    })
    return result


results = {}
TRACKS = [("Track A(全期間)", races_a, POOL_2A, POOL_3A, "A"),
          ("Track B(2026-07-25以降)", races_b, POOL_2B, POOL_3B, "B")]

for track_label, entries, pool2, pool3, track_key in TRACKS:
    log("\n" + "=" * 72)
    log(f"{track_label}  (N={len(entries)})")
    log("=" * 72)
    results[track_key] = {"n_races": len(entries), "by_box_n": {}}

    # --- box_n=4(主軸、フル統計検定)
    r2 = run_system(f"{track_label} 系統2(新規/既存データのみ、oddsなし)", entries, pool2,
                    BOX_N_PRIMARY, SEED + (0 if track_key == "A" else 100), primary=True)
    r3 = run_system(f"{track_label} 系統3(系統2+ninki統合)", entries, pool3,
                    BOX_N_PRIMARY, SEED + (1 if track_key == "A" else 101), primary=True)
    results[track_key]["by_box_n"][BOX_N_PRIMARY] = {"system2": r2, "system3": r3}

    # --- box_n=3/5(参考値のみ)
    for bn in BOX_NS_REFERENCE:
        r2_ref = run_system(f"{track_label} 系統2 参考値", entries, pool2, bn,
                            SEED + 200 + bn, primary=False)
        r3_ref = run_system(f"{track_label} 系統3 参考値", entries, pool3, bn,
                            SEED + 300 + bn, primary=False)
        results[track_key]["by_box_n"][bn] = {"system2": r2_ref, "system3": r3_ref}

    # --- walk-forward(box_n=4、系統3のみ、簡易前半/後半)
    train_idx, test_idx, mid_date = BE.walk_forward_split(entries)
    if len(train_idx) >= 10 and len(test_idx) >= 10:
        ev_wf = BE.Evaluator(entries, BOX_N_PRIMARY)
        mats_wf = build_matrices(entries, pool3)
        W_wf = make_weight_patterns(pool3, SEED + 500 + (0 if track_key == "A" else 1))
        picks_wf = [BE.score_picks_bottom(mats_wf, W_wf[:, j], BOX_N_PRIMARY) for j in range(N_PATTERNS)]
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
        log(f"\n[walk-forward](分割日={mid_date}、系統3、box_n={BOX_N_PRIMARY}) "
            f"前半で選んだパターンの市場差: train={excess_train:+.2f}pt → test={excess_test:+.2f}pt  "
            f"符号一致={'YES' if (excess_train > 0) == (excess_test > 0) else 'NO'}")
        results[track_key]["walk_forward"] = {
            "mid_date": mid_date, "excess_train": float(excess_train), "excess_test": float(excess_test),
            "sign_agree": bool((excess_train > 0) == (excess_test > 0)),
        }
    else:
        log("\n[walk-forward] train/testいずれかが小さすぎるためスキップ")
        results[track_key]["walk_forward"] = None

    # --- 馬ID group-split感度分析(box_n=4、系統3のみ)
    idx_a, idx_b = BE.horse_group_split(entries, seed=SEED + 600)
    ev_gs = BE.Evaluator(entries, BOX_N_PRIMARY)
    mats_gs = build_matrices(entries, pool3)
    W_gs = make_weight_patterns(pool3, SEED + 700 + (0 if track_key == "A" else 1))
    picks_gs = [BE.score_picks_bottom(mats_gs, W_gs[:, j], BOX_N_PRIMARY) for j in range(N_PATTERNS)]
    st_gs, rt_gs = [], []
    for p in picks_gs:
        s, r = ev_gs.hits_for(p)
        st_gs.append(s)
        rt_gs.append(r)
    vals_a = np.array([ev_gs.precision(st_gs[j], rt_gs[j], idx=idx_a) for j in range(N_PATTERNS)])
    best_a = int(np.argmax(vals_a))
    val_b_of_best_a = ev_gs.precision(st_gs[best_a], rt_gs[best_a], idx=idx_b)
    mean_b = float(np.mean([ev_gs.precision(st_gs[j], rt_gs[j], idx=idx_b) for j in range(N_PATTERNS)]))
    log(f"\n[馬ID group-split感度分析](系統3、box_n={BOX_N_PRIMARY}) "
        f"グループAで選んだパターンのグループBでの成績={val_b_of_best_a:.2f}%  "
        f"グループB全パターン平均={mean_b:.2f}%  差={val_b_of_best_a - mean_b:+.2f}pt")
    results[track_key]["horse_group_split"] = {
        "selected_on_a_eval_on_b": val_b_of_best_a, "unseen_b_mean": mean_b,
        "true_edge_pt": val_b_of_best_a - mean_b,
    }

    # --- 天井効果の副分析(box_n=4のみ、field_size>=8)
    entries_ceiling = [e for e in entries if e["field_size"] >= BOX_N_PRIMARY + 4]
    if len(entries_ceiling) >= 20:
        ev_c = BE.Evaluator(entries_ceiling, BOX_N_PRIMARY)
        mkt_c = ev_c.evaluate(ev_c.mkt_picks)
        mats_c = build_matrices(entries_ceiling, pool3)
        eq_c = ev_c.evaluate(BE.score_picks_bottom(mats_c, equal_w(pool3), BOX_N_PRIMARY))
        log(f"\n[天井効果の副分析](field_size>=8、N={len(entries_ceiling)}/{len(entries)}) "
            f"市場={mkt_c['model']:.2f}%  系統3等重み={eq_c['model']:.2f}%(市場差={eq_c['excess']:+.2f}pt)")
        results[track_key]["ceiling_subpop"] = {
            "n_races": len(entries_ceiling), "market": mkt_c["model"],
            "system3_equal_weight_excess": eq_c["excess"],
        }
    else:
        results[track_key]["ceiling_subpop"] = None

    # --- pre_race限定サブ集団での市場基準値(参考、provenance非対称性の定量化)
    entries_pre = [e for e in entries if e["race_id"] in pre_race_ids_in_pop]
    if len(entries_pre) >= 20:
        ev_pre = BE.Evaluator(entries_pre, BOX_N_PRIMARY)
        mkt_pre = ev_pre.evaluate(ev_pre.mkt_picks)
        log(f"\n[pre_race限定サブ集団](N={len(entries_pre)}) 市場precision={mkt_pre['model']:.2f}%"
            f"  (全体の市場precisionと比較することでprovenance非対称性の影響を確認)")
        results[track_key]["pre_race_subpop_market"] = {
            "n_races": len(entries_pre), "market": mkt_pre["model"],
        }
    else:
        results[track_key]["pre_race_subpop_market"] = None


# =====================================================================
# まとめ
# =====================================================================
log("\n" + "=" * 72)
log("まとめ")
log("=" * 72)
for track_key, track_label in (("A", "Track A(全期間)"), ("B", "Track B(2026-07-25以降)")):
    r4 = results[track_key]["by_box_n"][BOX_N_PRIMARY]
    for sys_key, sys_label in (("system2", "系統2(oddsなし)"), ("system3", "系統3(odds込み)")):
        s = r4[sys_key]
        boot = s.get("bootstrap_vs_market_paired")
        gate = s.get("gate_partial")
        log(f"{track_label} / {sys_label}: 市場={s['market']:.2f}%  "
            f"Nested LOBO OOF市場差={s['nested_lobo_oof']['excess']:+.2f}pt  "
            f"選ぶことの真の価値={s['selection_optimism']['true_edge_pt']:+.2f}pt  "
            f"市場比ペア差分CI=[{boot['lo']:+.2f}, {boot['hi']:+.2f}]pt  "
            f"部分ゲート={'YES' if gate else 'NO'}")
    wf = results[track_key].get("walk_forward")
    if wf:
        log(f"  walk-forward符号一致: {'YES' if wf['sign_agree'] else 'NO'}"
            f"(train={wf['excess_train']:+.2f}pt→test={wf['excess_test']:+.2f}pt)")

log("\n最終判定(事前登録ルール: 系統2または系統3で 部分ゲートYES かつ walk-forward符号一致 の場合のみ「採用検討」):")
for track_key, track_label in (("A", "Track A(全期間)"), ("B", "Track B(2026-07-25以降)")):
    r4 = results[track_key]["by_box_n"][BOX_N_PRIMARY]
    wf = results[track_key].get("walk_forward")
    wf_ok = bool(wf and wf["sign_agree"]) if wf else False
    any_gate = any(r4[k]["gate_partial"] for k in ("system2", "system3"))
    verdict = "採用検討" if (any_gate and wf_ok) else "不採用"
    log(f"  {track_label}: {verdict}")

# strip helper-only keys (picks arrays) before JSON dump
for track_key in results:
    for bn, cell in results[track_key]["by_box_n"].items():
        for sys_key in ("system2", "system3"):
            cell[sys_key].pop("_nested_oof_picks", None)

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps({
    "n_races_track_a": len(races_a), "n_races_track_b": len(races_b),
    "dead_signals": dead, "pool_2a": POOL_2A, "pool_2b": POOL_2B,
    "n_patterns": N_PATTERNS, "seed": SEED, "weight_tiers": WEIGHT_TIERS,
    "sign_check": sign_check, "results": results,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
