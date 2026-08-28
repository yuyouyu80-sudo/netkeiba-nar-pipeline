# -*- coding: utf-8 -*-
"""JRA Stage2 Phase J4: 多重比較補正(2026-08-28)。NAR Stage1のnar_stage1_rigor_check_
2026_08_28.pyのJRA移植版。Phase J1(peerセッション、7候補×box4)・Phase J2(自セッション、
Dirichlet/Ridge/Lasso×3box_n=9構成、現行本番[全246レース直接評価]比)・Phase J3(自セッション、
2頭軸流し・フォーメーション14構成、held-out限定)の全検定にBonferroni・Holm・BH(q=0.05)を
適用する。

既存の各結果JSON(jra_signal_gate_v4_2026_08_28_result.json / jra_search_combined_2026_08_28_
result.json / jra_new_bettypes_search_2026_08_28_result.json)を読むだけ、無改造。
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "jra_pipeline"
OUT_JSON = DATA_DIR / "jra_stage2_rigor_check_2026_08_28_result.json"
OUT_TXT = DATA_DIR / "jra_stage2_rigor_check_2026_08_28_report.txt"

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def normal_ci_to_p(mean: float, lo: float, hi: float) -> tuple:
    se = (hi - lo) / (2 * 1.959963984540054)
    z = mean / se if se > 0 else np.inf
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return se, z, p


def multi_comparison_table(name: str, items: list) -> dict:
    n = len(items)
    rows = []
    for it in items:
        se, z, p = normal_ci_to_p(it["mean"], it["lo"], it["hi"])
        rows.append({**it, "se": se, "z": z, "p": p})
    rows.sort(key=lambda r: r["p"])
    bonf_alpha = 0.05 / n
    holm_pass = True
    for k, r in enumerate(rows, start=1):
        thresh = 0.05 / (n - k + 1)
        r["holm_threshold"] = thresh
        r["holm_pass"] = holm_pass and (r["p"] <= thresh)
        if not r["holm_pass"]:
            holm_pass = False
    bh_q = 0.05
    passing_k = [k for k in range(1, n + 1) if rows[k - 1]["p"] <= (k / n) * bh_q]
    bh_cutoff_k = max(passing_k) if passing_k else 0
    for k, r in enumerate(rows, start=1):
        r["bh_threshold"] = (k / n) * bh_q
        r["bh_pass"] = k <= bh_cutoff_k
        r["bonferroni_threshold"] = bonf_alpha
        r["bonferroni_pass"] = r["p"] <= bonf_alpha

    log(f"\n=== {name}({n}検定) ===")
    log(f"  Bonferroni閾値: {bonf_alpha:.6f}")
    for r in rows:
        log(f"  {r['label']:36s} mean={r['mean']:+7.2f}pt p={r['p']:.5f}  "
            f"Bonf={'PASS' if r['bonferroni_pass'] else 'fail'}  "
            f"Holm={'PASS' if r['holm_pass'] else 'fail'}  "
            f"BH={'PASS' if r['bh_pass'] else 'fail'}")
    p_at_least_one = float(1 - (1 - 0.05) ** n)
    log(f"  参考: 独立にn={n}回検定した場合、名目α=0.05で1件以上偶然PASSする確率="
        f"{p_at_least_one*100:.1f}%")
    return {"n": n, "rows": rows, "p_at_least_one_false_positive": p_at_least_one}


# ============================================================ Phase J1: 7候補 box4(peerセッション)
gate = json.loads((DATA_DIR / "jra_signal_gate_v4_2026_08_28_result.json").read_text(encoding="utf-8"))
phase1_items = []
for c in gate["candidates"]:
    g = gate["gate_results"][c]["box4"]
    phase1_items.append({"label": c, "mean": g["excess_diff_pt"], "lo": g["paired_ci_lo"],
                          "hi": g["paired_ci_hi"]})
phase1_result = multi_comparison_table("Phase J1(box4、予想印・展開7候補、peerセッション実施)",
                                       phase1_items)

# ============================================================ Phase J2: 9構成(3手法×3box_n) vs 現行本番
combined = json.loads((DATA_DIR / "jra_search_combined_2026_08_28_result.json").read_text(encoding="utf-8"))
recheck = json.loads((DATA_DIR / "jra_search500_2026_08_28_groupkfold_recheck_result.json").read_text(encoding="utf-8"))
phase2_items = []
for box_n_str, d in combined["results_by_box"].items():
    for method, cmp_ in d["vs_current_model"].items():
        phase2_items.append({"label": f"box{box_n_str}_{method}", "mean": cmp_["mean"],
                              "lo": cmp_["lo"], "hi": cmp_["hi"]})
    dref = recheck["results_by_box"][box_n_str]["bootstrap_gkf_vs_current"]
    phase2_items.append({"label": f"box{box_n_str}_dirichlet", "mean": dref["mean"],
                         "lo": dref["lo"], "hi": dref["hi"]})
phase2_result = multi_comparison_table("Phase J2(box買い、3手法×3box_n、現行本番[全246レース]比)",
                                       phase2_items)

# ============================================================ Phase J3: 14構成(2軸流し・フォーメーション、held-out限定)
newbet = json.loads((DATA_DIR / "jra_new_bettypes_search_2026_08_28_result.json").read_text(encoding="utf-8"))
phase3_items = []
for r in newbet["results"]:
    ci = r["held_out_ci"]
    phase3_items.append({"label": f"{r['config']}_{r['bet_type']}", "mean": ci["mean"],
                         "lo": ci["lo"], "hi": ci["hi"]})
phase3_result = multi_comparison_table("Phase J3(2頭軸流し・フォーメーション14構成、held-out限定)",
                                       phase3_items)

# ============================================================ 保存
OUT_JSON.write_text(json.dumps({
    "phase_j1_box4": phase1_result, "phase_j2_vs_current": phase2_result,
    "phase_j3_new_bettypes": phase3_result,
}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
log(f"\n保存: {OUT_JSON}")
log(f"保存: {OUT_TXT}")
