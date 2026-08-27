# -*- coding: utf-8 -*-
"""Phase4レビュー2対応: 多重比較補正とDirichlet重み配分の帰無分布シミュレーションを、
レビュー2が本文中で示した数値を独立に再現・検証するために実施する。

レビュー2の指摘: 「明記する方針」だけでは不十分で、実際の数値(Bonferroni/Holm/BH閾値、
p値、偶然通過する確率、Dirichletの重み順位の帰無分布での位置)がないとレポート読者は
補正後も結果が残ると誤読する。ここで計算した数値をそのままHTMLレポート本文に転記する。

既存スクリプト(nar_signal_gate_v5_2026_08_27.py / nar_search_combined_2026_08_28.py /
nar_box3_coefficients_2026_08_28.py)は無改造、出力済みJSONを読むだけ。
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
GATE_JSON = OUT_DIR / "nar_signal_gate_v5_2026_08_27_result.json"
COMBINED_JSON = OUT_DIR / "nar_search_combined_2026_08_28_result.json"
OUT_JSON = OUT_DIR / "nar_stage1_rigor_check_2026_08_28_result.json"
OUT_TXT = OUT_DIR / "nar_stage1_rigor_check_2026_08_28_report.txt"

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def normal_ci_to_p(mean: float, lo: float, hi: float) -> tuple:
    """95%CI(正規近似)からSE・z・両側pを逆算する。"""
    se = (hi - lo) / (2 * 1.959963984540054)
    z = mean / se if se > 0 else np.inf
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return se, z, p


def multi_comparison_table(name: str, items: list) -> dict:
    """items: [{"label":..., "mean":..., "lo":..., "hi":...}, ...]
    Bonferroni/Holm/BH(q=0.05)を適用して表を返す。"""
    n = len(items)
    rows = []
    for it in items:
        se, z, p = normal_ci_to_p(it["mean"], it["lo"], it["hi"])
        rows.append({**it, "se": se, "z": z, "p": p})
    rows.sort(key=lambda r: r["p"])
    bonf_alpha = 0.05 / n
    # Holm: 昇順p値に対し alpha/(n-k+1)(k=1始まり)、途中で不通過なら以降全て不通過
    holm_pass = True
    for k, r in enumerate(rows, start=1):
        thresh = 0.05 / (n - k + 1)
        r["holm_threshold"] = thresh
        r["holm_pass"] = holm_pass and (r["p"] <= thresh)
        if not r["holm_pass"]:
            holm_pass = False
    # BH(q=0.05): 昇順p値に対し (k/n)*q、最大のkで通過なら それ以下すべて通過
    bh_q = 0.05
    bh_thresh = [(k, (k / n) * bh_q) for k in range(1, n + 1)]
    passing_k = [k for k, (kk, th) in zip(range(1, n + 1), bh_thresh) if rows[k - 1]["p"] <= th]
    bh_cutoff_k = max(passing_k) if passing_k else 0
    for k, r in enumerate(rows, start=1):
        r["bh_threshold"] = (k / n) * bh_q
        r["bh_pass"] = k <= bh_cutoff_k
        r["bonferroni_threshold"] = bonf_alpha
        r["bonferroni_pass"] = r["p"] <= bonf_alpha

    log(f"\n=== {name}({n}検定) ===")
    log(f"  Bonferroni閾値: {bonf_alpha:.6f}")
    for r in rows:
        log(f"  {r['label']:32s} mean={r['mean']:+.2f}pt p={r['p']:.5f}  "
            f"Bonf={'PASS' if r['bonferroni_pass'] else 'fail'}  "
            f"Holm(thr={r['holm_threshold']:.5f})={'PASS' if r['holm_pass'] else 'fail'}  "
            f"BH(thr={r['bh_threshold']:.5f})={'PASS' if r['bh_pass'] else 'fail'}")
    # 「n検定中1件以上が偶然でPASSする確率」(名目p<0.05の未補正検定をn回行った場合)
    p_at_least_one_false_positive = float(1 - (1 - 0.05) ** n)
    log(f"  参考: 独立にn={n}回検定した場合、名目α=0.05で1件以上偶然PASSする確率="
        f"{p_at_least_one_false_positive*100:.1f}%")
    return {"n": n, "rows": rows, "p_at_least_one_false_positive": p_at_least_one_false_positive}


# ============================================================ Phase1: 11候補 box4
gate = json.loads(GATE_JSON.read_text(encoding="utf-8"))
track_a, track_b = gate["track_a"], gate["track_b"]
gr = gate["gate_results"]
phase1_items = []
for c in track_a + track_b:
    g = gr[c]["box4"]
    phase1_items.append({"label": c, "mean": g["paired_ci_mean"], "lo": g["paired_ci_lo"],
                          "hi": g["paired_ci_hi"]})
phase1_result = multi_comparison_table("Phase1(box4、現行基準比ペア差分)", phase1_items)

# ============================================================ Phase2: 9構成(3手法×3box_n) vs 現行本番
combined = json.loads(COMBINED_JSON.read_text(encoding="utf-8"))
phase2_items = []
for box_n_str, d in combined["results_by_box"].items():
    for method, cmp_ in d["vs_current_prod"].items():
        phase2_items.append({"label": f"box{box_n_str}_{method}", "mean": cmp_["mean"],
                              "lo": cmp_["lo"], "hi": cmp_["hi"]})
phase2_result = multi_comparison_table("Phase2(3手法×3box_n、現行本番比)", phase2_items)

# ============================================================ Dirichlet重み配分の帰無分布シミュレーション
# nar_box3_coefficients_2026_08_28.py のDIRICHLET_POOL(18本、POOL_TRUE_PROD17+corner4_position)
# と同一次元のフラットDirichlet(alpha=1^18)から20万回抽出し、1位・2位の重みの順序統計量の
# 分布を作る。観測値(パターン#564、全データin-sample最良): 1位distance=0.1892、
# 2位corner4_position=0.1636(nar_box3_coefficients_2026_08_28_report.txt参照)。
N_DIM = 18
N_SIM = 200_000
SEED = 2026_08_28

log(f"\n=== Dirichlet(alpha=1^{N_DIM})の順序統計量、帰無分布シミュレーション(n={N_SIM}) ===")
rng = np.random.default_rng(SEED)
draws = rng.dirichlet(np.ones(N_DIM), size=N_SIM)
sorted_draws = np.sort(draws, axis=1)[:, ::-1]  # 降順
rank1 = sorted_draws[:, 0]
rank2 = sorted_draws[:, 1]

OBS_RANK1 = 0.1892  # distance (nar_box3_coefficients_2026_08_28_report.txt L80)
OBS_RANK2 = 0.1636  # corner4_position (同 L81)

p_rank1_ge_obs = float((rank1 >= OBS_RANK1).mean())
p_rank2_ge_obs = float((rank2 >= OBS_RANK2).mean())

log(f"  1位の重み: 期待値={rank1.mean():.4f}  観測値(distance)={OBS_RANK1}  "
    f"P(帰無分布で観測以上)={p_rank1_ge_obs*100:.1f}%")
log(f"  2位の重み: 期待値={rank2.mean():.4f}  観測値(corner4_position)={OBS_RANK2}  "
    f"P(帰無分布で観測以上)={p_rank2_ge_obs*100:.1f}%")
log("  解釈: 上記は「700本のうちin-sample argmaxで1本選んだ」状況の重み配分そのものが、")
log("  完全にランダムなDirichlet drawの順序統計量と比べて特に極端でないことを示す")
log("  (観測1位はランダムdrawの期待値を下回る)。Dirichletパターン探索の重み配分は")
log("  「どの信号が効くか」の証拠として提示すべきではない。")

dirichlet_null = {
    "n_dim": N_DIM, "n_sim": N_SIM,
    "rank1_expected": float(rank1.mean()), "rank1_observed": OBS_RANK1,
    "rank1_p_ge_observed": p_rank1_ge_obs,
    "rank2_expected": float(rank2.mean()), "rank2_observed": OBS_RANK2,
    "rank2_p_ge_observed": p_rank2_ge_obs,
}

# ============================================================ 保存
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps({
    "phase1_box4": phase1_result, "phase2_vs_current_prod": phase2_result,
    "dirichlet_weight_null_distribution": dirichlet_null,
}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
log(f"\n保存: {OUT_JSON}")
log(f"保存: {OUT_TXT}")
