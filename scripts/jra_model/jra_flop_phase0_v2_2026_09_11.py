# -*- coding: utf-8 -*-
"""凡走予測(flop) FLOP_SIGNALS_V2(30候補)のPhase0診断(2026-09-11)。

計画: `C:\\Users\\yuyou\\.claude\\plans\\valiant-cuddling-aho.md`
(「JRA凡走シグナル(flop) 全候補(候補200由来 約40本)の網羅探索」)

`jra_flop_gate1_2026_09_11.py`(FLOP_SIGNALS_V1、4候補版)のPhase0診断を一般化したもの。
既存 `jra_signals.py` / `jra_eval.py` / `jra_dataset.py` / `jra_candidate_factors_v1/v2/v3.py`
はすべて無改造で参照するのみ。

スクリーニング手順:
  1. 充足率・死にシグナルチェック(全30候補)
  2. 候補間相関(|rho|>=0.70のペアは充足率が低い方を実際にドロップする。V1 Gate1は
     警告ログのみだった点を是正)
  3. 本番シグナルとの相関スクリーニングは`JS.ALL_SIGNALS_V4 + JS.CANDIDATE_SIGNALS_V5`
     (mark/corner系V4を含む完全な45信号プール、box3テンプレートと同じ構成)に対して行う
     (V1 Gate1は`ALL_SIGNALS_V5`だけを見ておりV4のmark系を含んでいなかった点を是正)
  4. 粗い事前リフト診断(統計的採否には使わない)
  5. 1-3でドロップされなかった候補を「V2ショートリスト」として確定・保存
     (Phase1探索スクリプトはこのショートリストを読み込んで使う)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_candidate_factors_v1 as CF1
import jra_candidate_factors_v2 as CF2
import jra_candidate_factors_v3 as CF3
import jra_dataset
import jra_flop_signals as JFS
import jra_history as JH
import jra_lap33_signals as L33
import jra_model_scoring as MS
import jra_signals as JS

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "jra_pipeline"
OUT_JSON = DATA_DIR / "jra_flop_phase0_v2_2026_09_11_result.json"
OUT_TXT = DATA_DIR / "jra_flop_phase0_v2_2026_09_11_report.txt"
SHORTLIST_JSON = DATA_DIR / "jra_flop_v2_shortlist_2026_09_11.json"

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

log("\ncf1/cf2/cf3計算用の参照データ(race_results全件・騎手/調教師/馬主プロフィール)をロード中...")
results_all = JH.load_results()
race_meta = L33.load_race_surface_distance()
hist_idx = CF1.build_history_index(results_all)
cf3_ctx = CF3.Context(results_all)

cf1_by_race, cf2_by_race, cf3_by_race = [], [], []
for r in races:
    df = r["df"]
    meta = race_meta.get(r["race_id"])
    racecourse, surface, distance_m = meta if meta else (r["racecourse"], None, None)
    cf1 = CF1.compute_for_race(df, racecourse, distance_m, surface, r["kaisai_date"], hist_idx)
    cf2 = CF2.compute_for_race(df, r["race_name"], r["kaisai_date"], hist_idx)
    cf3 = CF3.compute_for_race(df, r["race_id"], r["kaisai_date"], racecourse, surface,
                                cf3_ctx, distance_m=distance_m)
    cf1_by_race.append(cf1)
    cf2_by_race.append(cf2)
    cf3_by_race.append(cf3)

# ============================================================ Phase0: 充足率診断
log("\n" + "=" * 72)
log(f"Phase0: FLOP_SIGNALS_V2({len(JFS.FLOP_SIGNALS_V2)}候補)の充足率診断")
log("=" * 72)
mats_v2_all = JFS.flop_signal_matrices_v2(cf1_by_race, cf2_by_race, cf3_by_race, JFS.FLOP_SIGNALS_V2)
A_v2 = np.vstack([m["A"] for m in mats_v2_all])
S_v2 = np.vstack([m["S"] for m in mats_v2_all])
fill_rates = {}
for j, n in enumerate(JFS.FLOP_SIGNALS_V2):
    fill = float(A_v2[:, j].mean())
    fill_rates[n] = fill
    log(f"  {n}: 充足率{fill:.1%}")

dead = [n for j, n in enumerate(JFS.FLOP_SIGNALS_V2) if A_v2[:, j].sum() == 0]
log(f"\n死にシグナル: {dead}")

S_v2_masked = np.where(A_v2 > 0, S_v2, np.nan)
df_v2 = pd.DataFrame(S_v2_masked, columns=JFS.FLOP_SIGNALS_V2)

# ============================================================ Phase0: 候補間相関(実際にドロップ)
log("\n" + "=" * 72)
log(f"Phase0: 候補間Spearman相関(|rho|>={CORR_DROP_THRESHOLD}のペアは充足率が低い方をドロップ)")
log("=" * 72)
corr_internal = df_v2.corr(method="spearman")
dropped_internal = []
survivors = [n for n in JFS.FLOP_SIGNALS_V2 if n not in dead]
names_check = list(survivors)
i = 0
while i < len(names_check):
    j = i + 1
    while j < len(names_check):
        a, b = names_check[i], names_check[j]
        rho = corr_internal.loc[a, b]
        if pd.notna(rho) and abs(rho) >= CORR_DROP_THRESHOLD:
            drop = a if fill_rates[a] < fill_rates[b] else b
            log(f"  {a} vs {b}: rho={rho:+.3f} -> {drop}をドロップ(充足率が低い方)")
            dropped_internal.append({"pair": [a, b], "rho": float(rho), "dropped": drop})
            names_check.remove(drop)
            if drop == a:
                j = i + 1
                continue
        j += 1
    i += 1
survivors_after_internal = names_check
log(f"\n候補間相関ドロップ後: {len(survivors_after_internal)}候補 "
    f"(死にシグナル{len(dead)}件 + 相関ドロップ{len(dropped_internal)}件を除外)")

# ============================================================ Phase0: 本番シグナルとの相関スクリーニング
log("\n" + "=" * 72)
log(f"Phase0: 本番シグナル(ALL_SIGNALS_V4+CANDIDATE_SIGNALS_V5、"
    f"{len(JS.ALL_SIGNALS_V4 + JS.CANDIDATE_SIGNALS_V5)}信号)との相関スクリーニング"
    f"(|rho|>={CORR_DROP_THRESHOLD}でドロップ)")
log("=" * 72)
PROD_POOL = JS.ALL_SIGNALS_V4 + JS.CANDIDATE_SIGNALS_V5  # box3テンプレートと同じ構成(V1の
# screeningがALL_SIGNALS_V5(V4のmark/corner系を含まない)だけを見ていた点をここで是正する。
mats_prod = JS.signal_matrices(races, priors_all, PROD_POOL, JS.CLASS_ORDINAL)
A_prod = np.vstack([m["A"] for m in mats_prod])
S_prod = np.vstack([m["S"] for m in mats_prod])
S_prod_masked = np.where(A_prod > 0, S_prod, np.nan)
df_prod = pd.DataFrame(S_prod_masked, columns=PROD_POOL)

high_corr_warnings = []
survivors_final = []
for fname in survivors_after_internal:
    corrs = {}
    for ename in PROD_POOL:
        combined = pd.concat([df_v2[fname], df_prod[ename]], axis=1).dropna()
        if len(combined) < 10:
            continue
        rho = combined.iloc[:, 0].corr(combined.iloc[:, 1], method="spearman")
        corrs[ename] = rho
    top = sorted(corrs.items(), key=lambda kv: -abs(kv[1]))[:3] if corrs else []
    max_rho = max((abs(r) for _, r in corrs.items()), default=0.0)
    log(f"  {fname}: 最大相関 " + (", ".join(f"{e}={r:+.3f}" for e, r in top) if top else "(算出不可)"))
    if max_rho >= CORR_DROP_THRESHOLD:
        worst = max(corrs.items(), key=lambda kv: abs(kv[1]))
        high_corr_warnings.append({"flop_signal": fname, "existing_signal": worst[0], "rho": worst[1]})
    else:
        survivors_final.append(fname)

log(f"\n本番シグナルとの高相関でドロップ: {[w['flop_signal'] for w in high_corr_warnings]}")
log(f"\nV2ショートリスト確定: {len(survivors_final)}候補")
log(f"  {survivors_final}")

# ============================================================ Phase0: 粗いリフト診断(ショートリスト等重み合成)
log("\n" + "=" * 72)
log("Phase0: V2ショートリスト等重み合成の粗いリフト診断(pred_rank1-5位の母集団内)")
log("=" * 72)
weights_normal = MS.load_weights("normal")
score_priors_normal = MS.compute_priors_for_population({"normal": [r["df"] for r in races]})["normal"]

finish_lookup: dict = {}
if {"race_id", "umaban", "finish_num"}.issubset(results_all.columns):
    for rid, g in results_all.groupby("race_id"):
        m = {}
        for uma, fin in zip(pd.to_numeric(g["umaban"], errors="coerce"), g["finish_num"]):
            if pd.notna(uma):
                m[int(uma)] = fin
        finish_lookup[rid] = m

flagged_mask_by_race, population_mask_by_race, finish_by_race = [], [], []
for r, cf1, cf2, cf3 in zip(races, cf1_by_race, cf2_by_race, cf3_by_race):
    df = r["df"]
    sig = JFS.compute_flop_signals_v2(cf1, cf2, cf3)
    composite = JS.combine_signals(sig, {n: 1.0 for n in survivors_final})
    score = MS.score_race(df, r["race_name"], "normal", score_priors_normal, weights_normal)
    pred_rank = MS.pred_rank_of(score)
    pop = (pred_rank <= 5).to_numpy()
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

# ============================================================ 保存
DATA_DIR.mkdir(parents=True, exist_ok=True)
result = {
    "n_races": len(races), "n_blocks": n_blocks, "dates": data["dates"],
    "candidates": JFS.FLOP_SIGNALS_V2,
    "fill_rates": fill_rates,
    "dead_signals": dead,
    "dropped_internal_corr": dropped_internal,
    "high_corr_warnings_vs_prod": high_corr_warnings,
    "shortlist": survivors_final,
    "precision_diag": precision,
}
OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
SHORTLIST_JSON.write_text(json.dumps({"shortlist": survivors_final}, ensure_ascii=False, indent=2),
                           encoding="utf-8")
log(f"\n保存: {OUT_JSON}")
log(f"保存: {OUT_TXT}")
log(f"保存: {SHORTLIST_JSON}")
