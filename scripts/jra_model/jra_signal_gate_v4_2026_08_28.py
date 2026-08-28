# -*- coding: utf-8 -*-
"""Phase0診断 + Phase1個別シグナルゲート(2026-08-28、JRA Stage2: 全ファクター統合・
高確信度選抜計画のJRA移植)。

scripts/nar_model/nar_signal_gate_v5_2026_08_27.pyのJRA移植版。既存 jra_signals.py /
jra_eval.py / jra_dataset.py はすべて無改造で参照するのみ(V4シグナル追加はjra_signals.py
本体に2026-08-28実施済み、既存シグナル・既存メソッドの挙動には触れていない)。

対象: CANDIDATE_SIGNALS_V4(予想印4本+展開7本=11本、jra_signals.py参照)。

NAR版との差分:
  * POOL_TRUE_PROD はJRAの実際の本番重み(winner_v3.json/winner_box4.json/winner_box3.json、
    2026-08-28実データ確認: いずれもLEGACY_SIGNALS10本のみ)に合わせてLEGACY_SIGNALS。
    NAR本番は17本(LEGACY+NEW+V2の一部)だがJRAは過去の候補シグナル探索(V1〜V3)がすべて
    不採用のまま現在に至るため、10本が正しい「現行本番と同一構成」。
  * ゲート設計はNAR版と同一: baseline(10本等重み) vs candidate(baseline+候補1本、11本等重み)を
    paired_block_bootstrap(同一レース上の対比較)のCI下限>0で判定するG1(単一の事前指定した
    比較、複数パターンから選ぶ探索ではないためselection_optimismは不要というNAR版と同じ判断)。
    box_n=4を主指標、5/3で一貫性確認。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_dataset
import jra_eval as JE
import jra_signals as JS

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "jra_pipeline"
OUT_JSON = DATA_DIR / "jra_signal_gate_v4_2026_08_28_result.json"
OUT_TXT = DATA_DIR / "jra_signal_gate_v4_2026_08_28_report.txt"

BOX_NS = [4, 5, 3]
CORR_DROP_THRESHOLD = 0.70

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


# ============================================================ データロード
data = jra_dataset.load(rebuild=True)
races, actual = data["races"], data["actual"]
priors_all = JS.make_priors([r["df"] for r in races])
n_blocks = len({f"{r['kaisai_date']}_{r['racecourse']}" for r in races})
log(f"レース数: {len(races)}  日付: {data['dates'][0]}〜{data['dates'][-1]}({len(data['dates'])}日)"
    f"  頭数: {sum(len(r['df']) for r in races)}  ブロック数: {n_blocks}")

# ============================================================ Phase0: V4充足率診断
log("\n" + "=" * 72)
log("Phase0: V4(予想印・展開)候補の充足率診断")
log("=" * 72)
mats_v4_all = JS.signal_matrices(races, priors_all, JS.CANDIDATE_SIGNALS_V4, JS.CLASS_ORDINAL)
A_v4 = np.vstack([m["A"] for m in mats_v4_all])
fill_rates = {}
for j, n in enumerate(JS.CANDIDATE_SIGNALS_V4):
    fill = float(A_v4[:, j].mean())
    fill_rates[n] = fill
    log(f"  {n}: 充足率{fill:.1%}")

log("\n--- hi==lo NaN安全性の実データ確認(mark_honshi欠測レース) ---")
missing_mark_races = []
for i, r in enumerate(races):
    col = r["df"]["mark_honshi"] if "mark_honshi" in r["df"].columns else None
    if col is None or (col.isna() | (col.astype(str) == "")).all():
        missing_mark_races.append(i)
missing_check = []
mh_idx = JS.CANDIDATE_SIGNALS_V4.index("mark_honshi_score")
for i in missing_mark_races[:3]:
    all_nan = bool(np.isnan(mats_v4_all[i]["S"][:, mh_idx]).all() and
                   (mats_v4_all[i]["A"][:, mh_idx] == 0).all())
    missing_check.append({"race_id": races[i]["race_id"], "confirmed_all_nan": all_nan})
    log(f"  race_id={races[i]['race_id']}: mark_honshi欠測 -> mark_honshi_score全NaN化: {all_nan}")
log(f"  mark_honshi欠測レース総数: {len(missing_mark_races)}/{len(races)}")

# ============================================================ Phase0: V4相関スクリーニング
log("\n" + "=" * 72)
log("Phase0: V4候補間のSpearman相関スクリーニング(多重比較対策)")
log("=" * 72)
S_v4 = np.vstack([m["S"] for m in mats_v4_all])
S_v4_masked = np.where(A_v4 > 0, S_v4, np.nan)
df_v4 = pd.DataFrame(S_v4_masked, columns=JS.CANDIDATE_SIGNALS_V4)
corr_v4 = df_v4.corr(method="spearman")
log(corr_v4.round(3).to_string())

active_v4 = list(JS.CANDIDATE_SIGNALS_V4)
dropped_v4 = []
pairs = []
for i, a in enumerate(JS.CANDIDATE_SIGNALS_V4):
    for b in JS.CANDIDATE_SIGNALS_V4[i + 1:]:
        pairs.append((abs(corr_v4.loc[a, b]), a, b))
pairs.sort(key=lambda x: -x[0])
for rho, a, b in pairs:
    if not (rho >= CORR_DROP_THRESHOLD) or a not in active_v4 or b not in active_v4:
        continue
    if fill_rates[a] != fill_rates[b]:
        drop = a if fill_rates[a] < fill_rates[b] else b
    else:
        drop = a if a.endswith("_gap") else b
    active_v4.remove(drop)
    dropped_v4.append(drop)
    log(f"  |rho|={rho:.3f} {a} vs {b} -> 除外: {drop}")
log(f"\nスクリーニング後候補({len(active_v4)}本): {active_v4}")
log(f"除外({len(dropped_v4)}本、高相関のため): {dropped_v4}")

# ============================================================ Phase0: 死にシグナル確認
log("\n" + "=" * 72)
log("Phase0: V4(スクリーニング後)の死にシグナル確認")
log("=" * 72)


def detect_dead_jra(names) -> list:
    alive = {n: 0 for n in names}
    for r in races:
        current_class = JS._class_ordinal(r["race_name"], JS.CLASS_ORDINAL)
        sig = JS.compute_signals(r["df"], current_class, priors_all, JS.CLASS_ORDINAL)
        for n in names:
            if sig[n].notna().any():
                alive[n] += 1
    return [n for n in names if alive[n] == 0]


dead_v4 = detect_dead_jra(active_v4)
log(f"死にシグナル: {dead_v4}")
candidates = [n for n in active_v4 if n not in dead_v4]
log(f"生存候補({len(candidates)}本): {candidates}")

# ============================================================ Phase1: 個別シグナルゲート
log("\n" + "=" * 72)
log("Phase1: 個別シグナルゲート(baseline=POOL_TRUE_PROD 10本 vs baseline+候補1本)")
log("=" * 72)

POOL_TRUE_PROD = list(JS.LEGACY_SIGNALS)
log(f"POOL_TRUE_PROD(現行本番と同一構成、winner_v3/box4/box3.json実データ確認済み、"
    f"{len(POOL_TRUE_PROD)}本): {POOL_TRUE_PROD}")

NAMES = JS.ALL_SIGNALS_V4
mats_all = JS.signal_matrices(races, priors_all, NAMES, JS.CLASS_ORDINAL)


def equal_w(names_subset) -> np.ndarray:
    d = {n: 1.0 / len(names_subset) for n in names_subset}
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


gate_results = {}
for box_n in BOX_NS:
    log(f"\n--- box_n={box_n} ---")
    ev = JE.Evaluator(races, actual, box_n=box_n)
    w_base = equal_w(POOL_TRUE_PROD)
    picks_base = JE.score_picks(mats_all, w_base, box_n)
    eval_base = ev.evaluate(picks_base)
    log(f"  baseline({len(POOL_TRUE_PROD)}本等重み): model={eval_base['model']:.2f}% "
        f"market={eval_base['market']:.2f}% excess={eval_base['excess']:+.2f}pt")
    for c in candidates:
        w_cand = equal_w(POOL_TRUE_PROD + [c])
        picks_cand = JE.score_picks(mats_all, w_cand, box_n)
        eval_cand = ev.evaluate(picks_cand)
        diff = ev.block_bootstrap_diff(picks_cand, picks_base, seed=41)
        gate_pass = diff["lo"] > 0
        excess_diff_pt = eval_cand["excess"] - eval_base["excess"]
        log(f"  [{c}] excess={eval_cand['excess']:+.2f}pt (基準比{excess_diff_pt:+.2f}pt) "
            f"ペア差分95%CI=[{diff['lo']:+.2f},{diff['hi']:+.2f}] G1={'PASS' if gate_pass else 'NO'}")
        gate_results.setdefault(c, {})[f"box{box_n}"] = {
            "excess_pt": eval_cand["excess"], "baseline_excess_pt": eval_base["excess"],
            "excess_diff_pt": excess_diff_pt, "paired_ci_lo": diff["lo"], "paired_ci_hi": diff["hi"],
            "paired_ci_mean": diff["mean"], "gate_pass": gate_pass,
        }

log("\n" + "=" * 72)
log("Phase1 まとめ(box_n=4主指標、G1=ペア差分ブートストラップ95%CI下限>0)")
log("=" * 72)
for c in candidates:
    g4 = gate_results[c]["box4"]
    consistent = all(gate_results[c][f"box{b}"]["gate_pass"] == g4["gate_pass"] for b in BOX_NS)
    log(f"  {c}: box4基準比{g4['excess_diff_pt']:+.2f}pt "
        f"G1={'PASS' if g4['gate_pass'] else 'NO'} (box5/3一貫: {consistent})")

# ============================================================ 保存
DATA_DIR.mkdir(parents=True, exist_ok=True)
result = {
    "n_races": len(races), "dates": data["dates"],
    "fill_rates_v4": fill_rates,
    "missing_mark_nan_check": missing_check,
    "corr_v4": corr_v4.round(4).to_dict(),
    "dropped_v4_by_correlation": dropped_v4,
    "dead_v4": dead_v4,
    "candidates": candidates,
    "pool_true_prod": POOL_TRUE_PROD,
    "gate_results": gate_results,
}
OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
log(f"\n保存: {OUT_JSON}")
log(f"保存: {OUT_TXT}")
