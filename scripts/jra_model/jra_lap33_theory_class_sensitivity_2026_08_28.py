# -*- coding: utf-8 -*-
"""Part1(記述的検証)の新馬・未勝利戦を含む/含まないケースの感度分析(ユーザー依頼)。

jra_lap33_signal_gate_2026_08_28.py(Part2)の母集団はjra_dataset.py全体の設計により
新馬・未勝利戦を除外している(このプロジェクトの予想モデル基盤全体の前提であり、今回の
33ラップ検証のために新設したものではない: predict_pattern29.py/confidence_sweep_v2.pyと
同一の"新馬|未勝利"正規表現)。一方、jra_lap33_theory_2026_08_28.py(Part1)はrace_name
によるクラスフィルタを一切行っておらず、実際に最終分析対象6,460件中2,847件(44.1%)が
新馬・未勝利戦だった(race_results全体の構成比44.8%とほぼ同じ=除外なし)。

「新馬・未勝利を含めたことで実態より良い一致度が出ているのではないか」という疑問に答える
ため、(a)新馬・未勝利のみ、(b)条件戦以上のみ、(c)全体(Part1本体と同じ)の3群で、
コース別平均33ラップと理論PDF値との一致度(符号一致率・Spearman順位相関・MAE)を
別々に計算し比較する。Part1本体のcompute_lap33/除外ロジックはそのまま踏襲し、
race_nameによる分割を追加しただけ。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jra_lap33_theory_table import THEORY_TABLE, THEORY_TABLE_AMBIGUOUS_AVG, AMBIGUOUS_CELLS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LAP_DIR = PROJECT_ROOT / "data" / "lap_times"
RESULTS_DIR = PROJECT_ROOT / "data" / "race_results"
OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
OUT_TXT = OUT_DIR / "jra_lap33_theory_class_sensitivity_2026_08_28_report.txt"

MIN_N_TRUST = 20
lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def load_lap_times() -> pd.DataFrame:
    frames = []
    for year_dir in sorted(LAP_DIR.glob("[0-9][0-9][0-9][0-9]")):
        for p in sorted(year_dir.glob("*.csv")):
            df = pd.read_csv(p, dtype=str)
            if len(df):
                frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["segment_index"] = pd.to_numeric(out["segment_index"], errors="coerce")
    out["lap_time_sec"] = pd.to_numeric(out["lap_time_sec"], errors="coerce")
    return out


def load_race_meta() -> pd.DataFrame:
    frames = []
    for year_dir in sorted(RESULTS_DIR.glob("[0-9][0-9][0-9][0-9]")):
        for p in sorted(year_dir.glob("*.csv")):
            df = pd.read_csv(
                p, dtype=str,
                usecols=["race_id", "surface", "distance_m", "racecourse", "race_name"])
            frames.append(df)
    out = pd.concat(frames, ignore_index=True).drop_duplicates("race_id")
    out["distance_m"] = pd.to_numeric(out["distance_m"], errors="coerce")
    return out


def compute_lap33(lap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for race_id, g in lap.groupby("race_id"):
        g = g.dropna(subset=["segment_index", "lap_time_sec"]).sort_values("segment_index")
        n = len(g)
        if n < 6:
            continue
        vals = g["lap_time_sec"].to_numpy()
        final_3f = float(vals[-3:].sum())
        preceding_3f = float(vals[-6:-3].sum())
        rows.append({"race_id": race_id, "n_segments": n,
                     "final_3f": final_3f, "preceding_3f": preceding_3f,
                     "lap33": preceding_3f - final_3f})
    return pd.DataFrame(rows)


def summarize(merged: pd.DataFrame, label: str) -> dict:
    """merged(1レース1行、lap33/racecourse/surface/distance_m列を持つ)から
    コース別平均33ラップを集計し、理論値との一致度指標を返す。"""
    grp = merged.groupby(["racecourse", "surface", "distance_m"])["lap33"].agg(["mean", "std", "count"])
    rows_out = []
    for (course, surf, dist), r in grp.iterrows():
        key = (course, surf, int(dist))
        theory_val = THEORY_TABLE.get(key)
        ambiguous = key in AMBIGUOUS_CELLS
        if theory_val is None and ambiguous:
            theory_val = THEORY_TABLE_AMBIGUOUS_AVG[key]
        if theory_val is None:
            continue
        rows_out.append({
            "racecourse": course, "surface": surf, "distance_m": int(dist),
            "n_races": int(r["count"]), "actual_mean": float(r["mean"]),
            "theory_value": float(theory_val), "diff": float(r["mean"] - theory_val),
            "low_n": bool(r["count"] < MIN_N_TRUST),
        })
    df = pd.DataFrame(rows_out)
    if df.empty:
        log(f"[{label}] 比較可能セル無し(該当データ不足)")
        return {"label": label, "n_input_races": len(merged), "n_cells": 0}

    n_cells = len(df)
    n_trusted = int((~df["low_n"]).sum())
    sign_agree = float((np.sign(df["actual_mean"]) == np.sign(df["theory_value"])).mean())
    trusted_df = df[~df["low_n"]]
    sign_agree_trusted = (
        float((np.sign(trusted_df["actual_mean"]) == np.sign(trusted_df["theory_value"])).mean())
        if len(trusted_df) >= 2 else float("nan"))
    rho, pval = stats.spearmanr(df["actual_mean"], df["theory_value"])
    if len(trusted_df) >= 2:
        rho_t, pval_t = stats.spearmanr(trusted_df["actual_mean"], trusted_df["theory_value"])
    else:
        rho_t, pval_t = float("nan"), float("nan")
    mae = float(df["diff"].abs().mean())
    mae_t = float(trusted_df["diff"].abs().mean()) if len(trusted_df) else float("nan")

    log(f"\n[{label}] 入力レース数={len(merged)}  比較セル数={n_cells}(N>={MIN_N_TRUST}: {n_trusted})")
    log(f"  符号一致率: 全{sign_agree * 100:.1f}%  N>=20のみ{sign_agree_trusted * 100:.1f}%")
    log(f"  Spearman rho: 全{rho:+.3f}(p={pval:.4f})  N>=20のみ{rho_t:+.3f}(p={pval_t:.4f})")
    log(f"  MAE: 全{mae:.2f}pt  N>=20のみ{mae_t:.2f}pt")
    return {
        "label": label, "n_input_races": len(merged), "n_cells": n_cells, "n_cells_trusted": n_trusted,
        "sign_agreement_all": sign_agree, "sign_agreement_trusted": sign_agree_trusted,
        "spearman_rho_all": float(rho), "spearman_p_all": float(pval),
        "spearman_rho_trusted": float(rho_t), "spearman_p_trusted": float(pval_t),
        "mae_all": mae, "mae_trusted": mae_t,
    }


log("ロード中...")
lap = load_lap_times()
meta = load_race_meta()
lap33 = compute_lap33(lap)
merged = lap33.merge(meta, on="race_id", how="left")
merged = merged[merged["surface"].isin(["芝", "ダ"]) & merged["distance_m"].notna()]
contaminated = (merged["n_segments"] == 6) & (merged["distance_m"] % 200 != 0)
merged = merged[~contaminated]
merged["is_maiden_or_mishoori"] = merged["race_name"].str.contains("新馬|未勝利", regex=True, na=False)
log(f"Part1本体と同じ前処理後: {len(merged)}件"
    f"(新馬/未勝利={int(merged['is_maiden_or_mishoori'].sum())}件"
    f"={merged['is_maiden_or_mishoori'].mean() * 100:.1f}%, "
    f"条件戦以上={int((~merged['is_maiden_or_mishoori']).sum())}件)")

log("\n" + "=" * 72)
log("新馬・未勝利を含む/含まないでコース別33ラップ vs 理論の一致度がどう変わるか")
log("=" * 72)

results = []
results.append(summarize(merged, "全体(Part1本体と同一)"))
results.append(summarize(merged[merged["is_maiden_or_mishoori"]], "新馬・未勝利のみ"))
results.append(summarize(merged[~merged["is_maiden_or_mishoori"]], "条件戦以上のみ(Part2母集団と同じクラス範囲)"))

log("\n" + "=" * 72)
log("結論")
log("=" * 72)
log("Part1(コース別平均33ラップの記述的検証)はrace_nameによるクラスフィルタを行っておらず、"
    "新馬・未勝利戦(44.1%)を除外していない。3群の一致度指標が近ければ、新馬・未勝利を"
    "含めたことによる水増しは無いと判断できる。")

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

import json  # noqa: E402
(OUT_DIR / "jra_lap33_theory_class_sensitivity_2026_08_28_result.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
log(f"\nwrote jra_lap33_theory_class_sensitivity_2026_08_28_report.txt / _result.json")
