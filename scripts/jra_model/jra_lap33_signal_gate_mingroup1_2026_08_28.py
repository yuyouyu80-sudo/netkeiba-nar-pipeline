# -*- coding: utf-8 -*-
"""「33ラップ理論」lap33_fitのゲート検証、MIN_GROUP_N=1版(ユーザー依頼「採用できる
状況を考えて」への実証追補)。

jra_lap33_signals.py本体は無改造。型判定カバレッジの感度試算で、n_lookback(20→40→9999)を
増やしても型判定できた馬の割合は64.2%のまま変化しない一方、MIN_GROUP_N(好走/凡走群それぞれに
必要な最小走数)を2→1に緩めると64.2%→85.2%まで上がることが分かった。「カバレッジ不足が
効果を薄めていただけではないか」という仮説を直接検証するため、L33.MIN_GROUP_Nを1に
モンキーパッチした状態でjra_lap33_signal_gate_2026_08_28.pyと全く同じゲート手順を再実行する。

トレードオフの注記: MIN_GROUP_N=1は「1走だけの好走(または凡走)」を型スコアの片側に
そのまま採用するため、標本内バラつきを均せず外れ値1本の影響を強く受ける。カバレッジは
上がるがシグナル自体のノイズは増える可能性がある——ここではその純粋なトレードオフの
損益を実測する。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_dataset
import jra_eval as JE
import jra_lap33_signals as L33
import jra_signals as JS

L33.MIN_GROUP_N = 1  # 本検証のための一時変更(モジュールファイル自体は無改造)

OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
OUT_TXT = OUT_DIR / "jra_lap33_signal_gate_mingroup1_2026_08_28_report.txt"

BOX_NS = [4, 5, 3]
POOL_TRUE_PROD = list(JS.LEGACY_SIGNALS)
NAMES = POOL_TRUE_PROD + ["lap33_fit"]

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


log(f"MIN_GROUP_N={L33.MIN_GROUP_N}(既定2から変更)で再検証")
log("データロード中...")
data = jra_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
priors_all = JS.make_priors([r["df"] for r in races])

lap33_lookup = L33.load_lap33_lookup()
race_meta = L33.load_race_surface_distance()
history_index = L33.build_history_index()
fit = L33.lap33_fit_matrix(races, history_index, lap33_lookup, race_meta)

n_horses = sum(len(v["type_score"]) for v in fit.values())
n_type_known = sum(int((v["type_score"] != 0.0).sum()) for v in fit.values())
log(f"型判定できた馬: {n_type_known}/{n_horses}({n_type_known / n_horses * 100:.1f}%)"
    f"  [MIN_GROUP_N=2では64.2%だった]")

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
    log(f"  baseline: excess={eval_base['excess']:+.2f}pt  candidate: excess={eval_cand['excess']:+.2f}pt"
        f"  (差{eval_cand['excess'] - eval_base['excess']:+.2f}pt)")
    log(f"  ペア差分95%CI=[{diff['lo']:+.2f},{diff['hi']:+.2f}]pt  G1={'PASS' if gate_pass else 'NO'}")
    gate_results[f"box{box_n}"] = {
        "excess_diff_pt": eval_cand["excess"] - eval_base["excess"],
        "paired_ci_lo": diff["lo"], "paired_ci_hi": diff["hi"], "gate_pass": gate_pass,
    }

g4 = gate_results["box4"]
consistent_sign = len({np.sign(gate_results[f"box{b}"]["excess_diff_pt"]) for b in BOX_NS}) == 1
verdict = "採用検討" if g4["gate_pass"] else "不採用"
log(f"\nbox4基準比{g4['excess_diff_pt']:+.2f}pt  G1={'PASS' if g4['gate_pass'] else 'NO'}"
    f"  box4/5/3の符号一致: {consistent_sign}")
log(f"判定(MIN_GROUP_N=1版): {verdict}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
log(f"\nwrote {OUT_TXT.name}")
