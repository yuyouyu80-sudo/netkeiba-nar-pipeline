# -*- coding: utf-8 -*-
"""「33ラップ理論」シグナル v2(lap33_fit_v2)の個別ゲート検証。

v1のゲート(`jra_lap33_signal_gate_2026_08_28.py`、別セッション作成)はG1=NO(box4/5/3
すべて不採用)という結果が既に出ている(2026-08-28、claude-8eより報告)。本スクリプトは
v2(動的ペース補正+芝/ダート・距離帯別層別型スコア)が v1 と比べて改善するかを、同じ設計
(baseline=LEGACY_SIGNALS10本 vs candidate=baseline+lap33_fit_v2、paired block bootstrap、
CI下限>0でG1判定、box4主指標・5/3で一貫性確認)で検証する。v1本体・jra_signals.py本体は
無改造で参照するのみ。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_dataset
import jra_eval as JE
import jra_lap33_signals as L33
import jra_lap33_signals_v2 as L33V2
import jra_signals as JS

OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\904b9395-7511-4618-878e-3d211a238f9f\scratchpad"
)
OUT_JSON = OUT_DIR / "jra_lap33_signal_gate_v2_2026_08_28_result.json"
OUT_TXT = OUT_DIR / "jra_lap33_signal_gate_v2_2026_08_28_report.txt"

BOX_NS = [4, 5, 3]
POOL_TRUE_PROD = list(JS.LEGACY_SIGNALS)
NAMES = POOL_TRUE_PROD + ["lap33_fit_v2"]

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


# ============================================================ データロード
log("データロード中...")
data = jra_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
priors_all = JS.make_priors([r["df"] for r in races])
log(f"レース数: {len(races)}  日付: {data['dates'][0]}〜{data['dates'][-1]}({len(data['dates'])}日)")

lap33_lookup = L33.load_lap33_lookup()
race_meta = L33.load_race_surface_distance()
history_index = L33.build_history_index()
baseline_ratio = L33V2.population_front_ratio(races)
fit = L33V2.lap33_fit_v2_matrix(races, history_index, lap33_lookup, race_meta, baseline_ratio)

n_ref = sum(1 for v in fit.values() if v["course_ref"] is not None)
n_horses = sum(len(v["type_score"]) for v in fit.values())
n_type_known = sum(int((v["type_score"] != 0.0).sum()) for v in fit.values())
log(f"lap33_fit_v2: 参照値(理論表)が引けたレース {n_ref}/{len(races)}  "
    f"型判定できた馬(層別+フォールバック込み) {n_type_known}/{n_horses}"
    f"({n_type_known / n_horses * 100:.1f}%)")

# ============================================================ 行列構築
mats_all = []
for r in races:
    current_class = JS._class_ordinal(r["race_name"], JS.CLASS_ORDINAL)
    sig = JS.compute_signals(r["df"], current_class, priors_all)
    sig["lap33_fit_v2"] = fit[r["race_id"]]["lap33_fit_v2"]
    S = np.column_stack([sig[n].fillna(0.0).to_numpy(dtype=float) for n in NAMES])
    A = np.column_stack([sig[n].notna().to_numpy(dtype=float) for n in NAMES])
    mats_all.append({"S": S, "A": A})


def equal_w(names_subset) -> np.ndarray:
    d = {n: 1.0 / len(names_subset) for n in names_subset}
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


w_base = equal_w(POOL_TRUE_PROD)
w_cand = equal_w(NAMES)

gate_results = {}
for box_n in BOX_NS:
    log(f"\n--- box_n={box_n} ---")
    ev = JE.Evaluator(races, actual, box_n=box_n)
    picks_base = JE.score_picks(mats_all, w_base, box_n)
    picks_cand = JE.score_picks(mats_all, w_cand, box_n)
    eval_base = ev.evaluate(picks_base)
    eval_cand = ev.evaluate(picks_cand)
    diff = ev.block_bootstrap_diff(picks_cand, picks_base)
    gate_pass = bool(diff["lo"] > 0)
    log(f"  baseline({len(POOL_TRUE_PROD)}本等重み): model={eval_base['model']:.2f}% "
        f"market={eval_base['market']:.2f}% excess={eval_base['excess']:+.2f}pt")
    log(f"  candidate(+lap33_fit_v2、{len(NAMES)}本等重み): model={eval_cand['model']:.2f}% "
        f"excess={eval_cand['excess']:+.2f}pt  (baseline比{eval_cand['excess'] - eval_base['excess']:+.2f}pt)")
    log(f"  ペア差分ブートストラップ95%CI=[{diff['lo']:+.2f},{diff['hi']:+.2f}]pt  "
        f"G1={'PASS' if gate_pass else 'NO'}")
    gate_results[f"box{box_n}"] = {
        "baseline_excess_pt": eval_base["excess"], "candidate_excess_pt": eval_cand["excess"],
        "excess_diff_pt": eval_cand["excess"] - eval_base["excess"],
        "paired_ci_lo": diff["lo"], "paired_ci_hi": diff["hi"], "paired_ci_mean": diff["mean"],
        "gate_pass": gate_pass,
    }

log("\n" + "=" * 72)
log("まとめ(box4主指標、G1=ペア差分ブートストラップ95%CI下限>0)")
log("=" * 72)
g4 = gate_results["box4"]
consistent = all(gate_results[f"box{b}"]["gate_pass"] == g4["gate_pass"] for b in BOX_NS)
verdict = "採用検討" if g4["gate_pass"] else "不採用"
log(f"box4基準比{g4['excess_diff_pt']:+.2f}pt  95%CI=[{g4['paired_ci_lo']:+.2f},"
    f"{g4['paired_ci_hi']:+.2f}]  G1={'PASS' if g4['gate_pass'] else 'NO'}  "
    f"(box5/3一貫: {consistent})")
log(f"判定: {verdict}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps({
    "n_races": len(races), "pool_true_prod": POOL_TRUE_PROD,
    "n_ref_available": n_ref, "n_horses": n_horses, "n_type_known": n_type_known,
    "gate_results": gate_results, "verdict": verdict, "box5_3_consistent": consistent,
}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
