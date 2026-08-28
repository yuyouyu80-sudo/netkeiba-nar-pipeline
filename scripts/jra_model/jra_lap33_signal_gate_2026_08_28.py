# -*- coding: utf-8 -*-
"""「33ラップ理論」シグナル(lap33_fit)の個別ゲート検証(Part 2)。

設計はnar_signal_gate_v5_2026_08_27.py/jra_signal_gate_v4_2026_08_28.pyと同一の理由による:
baseline(POOL_TRUE_PROD=LEGACY_SIGNALS10本、等重み) vs candidate(baseline+lap33_fit、
11本等重み)という単一の事前指定した比較(多数の重みパターンから最良を選ぶ探索ではない)
なので、selection_optimism/group_kfold_oofは本質的に不要と判断し、
block_bootstrap_diff(同一レース上の対比較)のCI下限>0をゲートG1とする。box_n=4を主指標、
box5/3で一貫性確認。

前提: jra_lap33_theory_2026_08_28.py を先に実行し、jra_lap33_by_race_2026_08_28.csv
(race_id->33ラップ実測値)が出力済みであること。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_dataset
import jra_eval as JE
import jra_lap33_signals as L33
import jra_signals as JS

OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
OUT_JSON = OUT_DIR / "jra_lap33_signal_gate_2026_08_28_result.json"
OUT_TXT = OUT_DIR / "jra_lap33_signal_gate_2026_08_28_report.txt"

BOX_NS = [4, 5, 3]
POOL_TRUE_PROD = list(JS.LEGACY_SIGNALS)  # 現行本番(winner_v3/box4/box3.json実データ確認済み)
NAMES = POOL_TRUE_PROD + ["lap33_fit"]

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
fit = L33.lap33_fit_matrix(races, history_index, lap33_lookup, race_meta)

n_ref = sum(1 for v in fit.values() if v["course_ref"] is not None)
n_horses = sum(len(v["type_score"]) for v in fit.values())
n_type_known = sum(int((v["type_score"] != 0.0).sum()) for v in fit.values())
log(f"lap33_fit: 参照値(理論表)が引けたレース {n_ref}/{len(races)}  "
    f"型判定できた馬 {n_type_known}/{n_horses}({n_type_known / n_horses * 100:.1f}%)")

# ============================================================ 行列構築(既存signal_matricesと
# 同じ(S,A)形式に、外部計算のlap33_fitを1列追加する)
mats_all = []
for r in races:
    current_class = JS._class_ordinal(r["race_name"], JS.CLASS_ORDINAL)
    sig = JS.compute_signals(r["df"], current_class, priors_all)
    sig["lap33_fit"] = fit[r["race_id"]]["lap33_fit"]
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
    log(f"  candidate(+lap33_fit、{len(NAMES)}本等重み): model={eval_cand['model']:.2f}% "
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
