# -*- coding: utf-8 -*-
"""凡走予測(flop)第一弾4候補のPhase0診断 + G1ゲート(2026-09-11)。

計画: `C:\\Users\\yuyou\\.claude\\plans\\valiant-cuddling-aho.md`
(「JRA凡走予測モデル(flop)を『参考表示専用』から『本番予想に資する検証済みシグナル』へ」)

`jra_signal_gate_v4_2026_08_28.py`と同型の1ファイル構成(Phase0診断 + G1ゲートを続けて
実施する既存の慣習を踏襲)。既存 `jra_signals.py` / `jra_eval.py` / `jra_dataset.py` /
`jra_candidate_factors_v2.py` / `v3.py` はすべて無改造で参照するのみ。

評価設計(シニアエンジニアレビュー反映済み、jra_flop_eval.py参照): 「BOX5上位5頭から
flop_scoreが最も高い1頭を機械的に除外する」二値除外方式ではなく、
`adjusted_score = score - alpha*flop_composite` による連続スコア調整で常にbox_n頭ちょうどを
選ぶ方式を採用する。baseline(alpha=0、現行と同一) vs candidate(alpha>0)の
ペア差分95%CI下限>0でPASS。

対象母集団: `jra_dataset.load()`(通常戦246レース、jra_signal_gate_v4と同一母集団。
priorsとの整合性・過去の探索との比較可能性を優先)。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_candidate_factors_v2 as CF2
import jra_candidate_factors_v3 as CF3
import jra_dataset
import jra_eval as JE
import jra_flop_eval as JFE
import jra_flop_signals as JFS
import jra_history as JH
import jra_lap33_signals as L33
import jra_model_scoring as MS
import jra_signals as JS

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "jra_pipeline"
OUT_JSON = DATA_DIR / "jra_flop_gate1_2026_09_11_result.json"
OUT_TXT = DATA_DIR / "jra_flop_gate1_2026_09_11_report.txt"

BOX_NS = [4, 5, 3]
ALPHAS = [0.05, 0.1, 0.2]
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

log("\ncf2/cf3計算用の参照データ(race_results全件・騎手/調教師/馬主プロフィール)をロード中...")
results_all = JH.load_results()
race_meta = L33.load_race_surface_distance()
cf3_ctx = CF3.Context(results_all)

cf2_by_race, cf3_by_race = [], []
for r in races:
    df = r["df"]
    meta = race_meta.get(r["race_id"])
    racecourse, surface, distance_m = meta if meta else (r["racecourse"], None, None)
    cf2 = CF2.compute_layoff_factors(df, r["kaisai_date"])
    cf3 = CF3.compute_for_race(df, r["race_id"], r["kaisai_date"], racecourse, surface,
                                cf3_ctx, distance_m=distance_m)
    cf2_by_race.append(cf2)
    cf3_by_race.append(cf3)

# ============================================================ Phase0: 充足率診断
log("\n" + "=" * 72)
log("Phase0: flop第一弾4候補の充足率診断")
log("=" * 72)
mats_flop_all = JFS.flop_signal_matrices(cf2_by_race, cf3_by_race, JFS.FLOP_SIGNALS_V1)
A_flop = np.vstack([m["A"] for m in mats_flop_all])
S_flop = np.vstack([m["S"] for m in mats_flop_all])
fill_rates = {}
for j, n in enumerate(JFS.FLOP_SIGNALS_V1):
    fill = float(A_flop[:, j].mean())
    fill_rates[n] = fill
    log(f"  {n}: 充足率{fill:.1%}")

# ============================================================ Phase0: 死にシグナル確認
dead = [n for j, n in enumerate(JFS.FLOP_SIGNALS_V1) if A_flop[:, j].sum() == 0]
log(f"\n死にシグナル: {dead}")

# ============================================================ Phase0: 極性・分布の目視確認
log("\n" + "=" * 72)
log("Phase0: 極性・分布の目視確認(危険方向に0..1で分布しているか)")
log("=" * 72)
S_flop_masked = np.where(A_flop > 0, S_flop, np.nan)
df_flop = pd.DataFrame(S_flop_masked, columns=JFS.FLOP_SIGNALS_V1)
log(df_flop.describe().round(3).to_string())

# ============================================================ Phase0: 候補間相関
log("\n" + "=" * 72)
log("Phase0: flop候補間のSpearman相関")
log("=" * 72)
corr_internal = df_flop.corr(method="spearman")
log(corr_internal.round(3).to_string())

# ============================================================ Phase0: 既存シグナルとの相関(重複検出)
log("\n" + "=" * 72)
log("Phase0: 既存ALL_SIGNALS_V5との相関スクリーニング(|rho|>=%.2f で警告)" % CORR_DROP_THRESHOLD)
log("=" * 72)
mats_existing = JS.signal_matrices(races, priors_all, JS.ALL_SIGNALS_V5, JS.CLASS_ORDINAL)
A_ex = np.vstack([m["A"] for m in mats_existing])
S_ex = np.vstack([m["S"] for m in mats_existing])
S_ex_masked = np.where(A_ex > 0, S_ex, np.nan)
df_ex = pd.DataFrame(S_ex_masked, columns=JS.ALL_SIGNALS_V5)

high_corr_warnings = []
for fname in JFS.FLOP_SIGNALS_V1:
    corrs = {}
    for ename in JS.ALL_SIGNALS_V5:
        combined = pd.concat([df_flop[fname], df_ex[ename]], axis=1).dropna()
        if len(combined) < 10:
            continue
        rho = combined.iloc[:, 0].corr(combined.iloc[:, 1], method="spearman")
        corrs[ename] = rho
    if not corrs:
        continue
    top = sorted(corrs.items(), key=lambda kv: -abs(kv[1]))[:3]
    log(f"  {fname}: 最大相関 " + ", ".join(f"{e}={r:+.3f}" for e, r in top))
    for ename, rho in top:
        if abs(rho) >= CORR_DROP_THRESHOLD:
            high_corr_warnings.append({"flop_signal": fname, "existing_signal": ename, "rho": rho})

if high_corr_warnings:
    log(f"\n高相関警告(|rho|>={CORR_DROP_THRESHOLD}): {high_corr_warnings}")
else:
    log(f"\n高相関警告: なし(既存シグナルとの重複懸念は実データ上確認されなかった)")

# ============================================================ Phase0: 粗いリフト診断
log("\n" + "=" * 72)
log("Phase0: flop_composite(等重み)の粗いリフト診断(pred_rank1-5位の母集団内)")
log("=" * 72)
weights_normal = MS.load_weights("normal")
score_priors_normal = MS.compute_priors_for_population({"normal": [r["df"] for r in races]})["normal"]

# race_id -> {umaban(int): finish_num(float)} のlookup(results_allを1回だけ走査)
finish_lookup: dict = {}
if {"race_id", "umaban", "finish_num"}.issubset(results_all.columns):
    for rid, g in results_all.groupby("race_id"):
        m = {}
        for uma, fin in zip(pd.to_numeric(g["umaban"], errors="coerce"), g["finish_num"]):
            if pd.notna(uma):
                m[int(uma)] = fin
        finish_lookup[rid] = m

flagged_mask_by_race, population_mask_by_race, finish_by_race = [], [], []
for r, cf2, cf3 in zip(races, cf2_by_race, cf3_by_race):
    df = r["df"]
    sig = JFS.compute_flop_signals(cf2, cf3)
    composite = JFS.compute_flop_composite(sig)
    score = MS.score_race(df, r["race_name"], "normal", score_priors_normal, weights_normal)
    pred_rank = MS.pred_rank_of(score)
    pop = (pred_rank <= 5).to_numpy()
    # composite上位40%を「flopフラグ」とする簡易しきい値(統計的採否には使わない、粗い診断専用)
    comp_vals = composite.to_numpy(dtype=float)
    thresh = np.nanpercentile(comp_vals, 60) if composite.notna().any() else np.nan
    flag = (comp_vals >= thresh) if not np.isnan(thresh) else np.zeros(len(df), dtype=bool)

    rid_lookup = finish_lookup.get(r["race_id"], {})
    uma_series = JS._num(JS._col(df, "umaban"))
    finish = np.array([
        pd.to_numeric(rid_lookup.get(int(u)), errors="coerce") if pd.notna(u) else np.nan
        for u in uma_series
    ], dtype=float)

    flagged_mask_by_race.append(flag)
    population_mask_by_race.append(pop)
    finish_by_race.append(finish)

precision = JFS.flop_precision_table(flagged_mask_by_race, population_mask_by_race, finish_by_race)
log(f"  母集団: pred_rank1-5位  flopフラグ(composite上位40%)有無別の複勝圏外率")
log(f"  flopフラグ有り: {precision['flagged_outside_rate']} (n={precision['n_flagged']})")
log(f"  flopフラグ無し: {precision['baseline_outside_rate']} (n={precision['n_baseline']})")
log(f"  リフト: {precision['lift_pt']}pt(統計的採否には使わない、粗い事前スクリーニング専用)")

# ============================================================ Phase1: G1ゲート
log("\n" + "=" * 72)
log("Phase1: G1ゲート(baseline: alpha=0 vs candidate: alpha>0, box_n別)")
log("=" * 72)

NAMES_SCORE = JS.LEGACY_SIGNALS  # 現行本番と同一構成(winner_v3/box4/box3.jsonはLEGACY_SIGNALSのみ)
mats_score_all = JS.signal_matrices(races, priors_all, NAMES_SCORE, JS.CLASS_ORDINAL)
w_score_equal = np.array([1.0] * len(NAMES_SCORE))  # LEGACY_SIGNALSは元々等重みではないが、
# G1比較の目的は「flop加味の有無」の差分なので、baseline側の重み自体はwinner_v3.json実値を使う
w_score_prod = {}
for rt in ("normal", "normal_box4", "normal_box3"):
    w = MS.load_weights(rt)
    w_score_prod[rt] = np.array([float(w.get(n, 0.0)) for n in NAMES_SCORE])

w_flop_equal = np.array([1.0] * len(JFS.FLOP_SIGNALS_V1))

box_to_weight_key = {5: "normal", 4: "normal_box4", 3: "normal_box3"}
gate_results = {}
for box_n in BOX_NS:
    log(f"\n--- box_n={box_n}(重み: {box_to_weight_key[box_n]}) ---")
    ev = JE.Evaluator(races, actual, box_n=box_n)
    w_score = w_score_prod[box_to_weight_key[box_n]]
    picks_base = JFE.adjusted_picks(mats_score_all, w_score, mats_flop_all, w_flop_equal,
                                    alpha=0.0, box_n=box_n)
    eval_base = ev.evaluate(picks_base)
    log(f"  baseline(alpha=0、現行と同一): model={eval_base['model']:.2f}% "
        f"market={eval_base['market']:.2f}% excess={eval_base['excess']:+.2f}pt")

    for alpha in ALPHAS:
        picks_cand = JFE.adjusted_picks(mats_score_all, w_score, mats_flop_all, w_flop_equal,
                                        alpha=alpha, box_n=box_n)
        eval_cand = ev.evaluate(picks_cand)
        diff = ev.block_bootstrap_diff(picks_cand, picks_base, seed=41)
        gate_pass = diff["lo"] > 0
        excess_diff_pt = eval_cand["excess"] - eval_base["excess"]
        log(f"  [alpha={alpha}] excess={eval_cand['excess']:+.2f}pt (基準比{excess_diff_pt:+.2f}pt) "
            f"ペア差分95%CI=[{diff['lo']:+.2f},{diff['hi']:+.2f}] G1={'PASS' if gate_pass else 'NO'}")
        gate_results.setdefault(f"alpha_{alpha}", {})[f"box{box_n}"] = {
            "excess_pt": eval_cand["excess"], "baseline_excess_pt": eval_base["excess"],
            "excess_diff_pt": excess_diff_pt, "paired_ci_lo": diff["lo"], "paired_ci_hi": diff["hi"],
            "paired_ci_mean": diff["mean"], "gate_pass": gate_pass,
        }

log("\n" + "=" * 72)
log("Phase1 まとめ(box_n=4主指標、G1=ペア差分ブートストラップ95%CI下限>0)")
log("=" * 72)
any_pass = False
for alpha in ALPHAS:
    g4 = gate_results[f"alpha_{alpha}"]["box4"]
    consistent = all(gate_results[f"alpha_{alpha}"][f"box{b}"]["gate_pass"] == g4["gate_pass"]
                     for b in BOX_NS)
    log(f"  alpha={alpha}: box4基準比{g4['excess_diff_pt']:+.2f}pt "
        f"G1={'PASS' if g4['gate_pass'] else 'NO'} (box5/3一貫: {consistent})")
    any_pass = any_pass or g4["gate_pass"]

log(f"\n総合判定: {'いずれかのalphaでPASS' if any_pass else '全alphaでREJECT'}")

# ============================================================ 保存
DATA_DIR.mkdir(parents=True, exist_ok=True)
result = {
    "n_races": len(races), "n_blocks": n_blocks, "dates": data["dates"],
    "candidates": JFS.FLOP_SIGNALS_V1,
    "fill_rates": fill_rates,
    "dead_signals": dead,
    "corr_internal": corr_internal.round(4).to_dict(),
    "high_corr_warnings_vs_existing": high_corr_warnings,
    "precision_diag": precision,
    "alphas_tested": ALPHAS,
    "gate_results": gate_results,
    "any_alpha_pass": any_pass,
}
OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
log(f"\n保存: {OUT_JSON}")
log(f"保存: {OUT_TXT}")
