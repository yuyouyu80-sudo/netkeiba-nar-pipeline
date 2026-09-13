# -*- coding: utf-8 -*-
"""血統理論(インブリード係数・ニックス)の記述的検証、2年分データでの再検証。

jra_pedigree_theory_2026_08_29.py(2026年7-8月JRA・3,359件)と全く同じ手法・同じ除外条件
(新馬・未勝利を除く通常戦、完走のみ)を、2024年〜2026年8月JRA・16,117頭分の血統データ
(2026-08-29〜30にバックフィル済み)に対して再実行する。シグナル化・採否判断はしない
(記述的検証のみ)。
"""
import glob
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.netkeiba_pipeline.storage.paths import load_all_pedigree  # noqa: E402

OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\1ee9e967-1d8b-4f9a-9e82-dc5eee4b8f13\scratchpad"
)
RESULTS_DIR = PROJECT_ROOT / "data" / "race_results"

lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    lines.append(str(s))


def inbreeding_coefficient(row: pd.Series) -> float:
    by_ancestor: dict[str, list[str]] = {}
    for col in row.index:
        if col.endswith("_horse_id") and col.startswith("ped_"):
            hid = row[col]
            if pd.isna(hid) or hid == "":
                continue
            path = col[len("ped_") : -len("_horse_id")]
            by_ancestor.setdefault(hid, []).append(path)

    f_total = 0.0
    for _ancestor_id, paths in by_ancestor.items():
        if len(paths) < 2:
            continue
        s_paths = [p for p in paths if p[0] == "S"]
        d_paths = [p for p in paths if p[0] == "D"]
        for sp, dp in product(s_paths, d_paths):
            n1, n2 = len(sp), len(dp)
            f_total += (0.5) ** (n1 + n2 + 1)
    return f_total


log("血統データロード中(2024-2026年8月、JRA)...")
ped = load_all_pedigree()
log(f"pedigreeデータ: {len(ped)}頭")

ped["inbreeding_f"] = ped.apply(inbreeding_coefficient, axis=1)
ped["nicks_key"] = list(zip(ped["ped_S_horse_id"], ped["ped_DS_horse_id"]))

log("\n" + "=" * 72)
log("インブリード係数F(X)の分布(5代打ち切り簡易版、2024-2026年8月・全頭)")
log("=" * 72)
log(ped["inbreeding_f"].describe().to_string())
log(f"F=0の馬: {(ped['inbreeding_f'] == 0).sum()}/{len(ped)} "
    f"({(ped['inbreeding_f'] == 0).mean()*100:.1f}%)")

log("\n血統データを2024年〜2026年8月JRA通常戦の実績と結合中(新馬・未勝利除く、完走のみ)...")
result_frames = []
for year in ("2024", "2025", "2026"):
    for p in sorted(glob.glob(str(RESULTS_DIR / year / "*.csv"))):
        df = pd.read_csv(p, dtype=str, encoding="utf-8")
        result_frames.append(df)
results = pd.concat(result_frames, ignore_index=True)
log(f"race_results総行数: {len(results)}")
results = results[~results["race_name"].str.contains("新馬|未勝利", regex=True, na=False)]
results["finish_pos_num"] = pd.to_numeric(results["finish_pos"], errors="coerce")
results = results.dropna(subset=["finish_pos_num"])
results["popularity_num"] = pd.to_numeric(results["popularity"], errors="coerce")

field_size = results.groupby("race_id")["horse_id"].transform("count")
results["field_size"] = field_size
results["finish_pct"] = (results["finish_pos_num"] - 1) / (results["field_size"] - 1).clip(lower=1)
results["is_win"] = (results["finish_pos_num"] == 1).astype(int)

merged = results.merge(ped[["horse_id", "inbreeding_f", "nicks_key"]], on="horse_id", how="inner")
log(f"結合成功: {len(merged)}件(通常戦・完走・2024-2026年8月JRA、pedigree取得済み馬のみ)")

log("\n" + "=" * 72)
log("インブリード係数F と 着順(finish_pct=0が1着、1が最下位相当) の記述的相関")
log("=" * 72)
rho, pval = stats.spearmanr(merged["inbreeding_f"], merged["finish_pct"])
log(f"Spearman順位相関: rho={rho:+.4f} p={pval:.4f} (N={len(merged)})")

f_pos = merged[merged["inbreeding_f"] > 0]
f_zero = merged[merged["inbreeding_f"] == 0]
log(f"\nF=0群: N={len(f_zero)}  平均着順割合={f_zero['finish_pct'].mean():.4f}  勝率={f_zero['is_win'].mean()*100:.2f}%")
log(f"F>0群: N={len(f_pos)}  平均着順割合={f_pos['finish_pct'].mean():.4f}  勝率={f_pos['is_win'].mean()*100:.2f}%")
u, up = stats.mannwhitneyu(f_zero["finish_pct"], f_pos["finish_pct"], alternative="two-sided")
log(f"Mann-Whitney U検定(着順割合、F=0群 vs F>0群): p={up:.4f}")

# 連続値としてのF自体も四分位群で見る(非線形の可能性の記述的確認)
merged["f_quartile"] = pd.qcut(merged["inbreeding_f"], 4, duplicates="drop")
log("\nFの四分位群ごとの勝率・平均着順割合:")
log(merged.groupby("f_quartile", observed=True).agg(
    n=("is_win", "size"), win_rate=("is_win", "mean"), mean_finish_pct=("finish_pct", "mean")
).to_string())

log("\n" + "=" * 72)
log("ニックス(個体種牡馬×個体ブルードメアサイアの組み合わせ)の記述的検証")
log("=" * 72)
combo_counts = merged.groupby("nicks_key").size().sort_values(ascending=False)
log(f"出現する組み合わせ数: {len(combo_counts)}  対象レコード数: {len(merged)}")
for thresh in (1, 3, 5, 10, 20, 30):
    n_thresh = (combo_counts >= thresh).sum() if thresh > 1 else (combo_counts == 1).sum()
    label = f"N=={thresh}" if thresh == 1 else f"N>={thresh}"
    log(f"{label}の組み合わせ: {n_thresh} ({n_thresh/len(combo_counts)*100:.1f}%)")

baseline_win_rate = merged["is_win"].mean()
log(f"\n母集団全体の勝率(ベースライン): {baseline_win_rate*100:.2f}% (N={len(merged)})")

MIN_N = 20
eligible = combo_counts[combo_counts >= MIN_N].index
log(f"\nN>={MIN_N}の組み合わせ({len(eligible)}件)の勝率一覧(降順、上位15件のみ表示。"
    "**候補シグナルとしての採用可否判断ではなく頻度分布の記述目的**):")
combo_stats = (
    merged[merged["nicks_key"].isin(eligible)]
    .groupby("nicks_key")
    .agg(n=("is_win", "size"), win_rate=("is_win", "mean"), mean_finish_pct=("finish_pct", "mean"))
    .sort_values("win_rate", ascending=False)
)
log(combo_stats.head(15).to_string())
log(f"\n参考: 全体平均を大きく下回る組み合わせ(下位5件):")
log(combo_stats.tail(5).to_string())

# 二項検定でベースラインからの乖離が統計的に有意かどうかも見る(多重検定は未補正、参考値)
log(f"\nN>={MIN_N}の組み合わせのうち、二項検定(片側、ベースライン{baseline_win_rate*100:.2f}%との比較、"
    "有意水準p<0.05、多重検定未補正)で有意に勝率が高い/低いもの:")
sig_rows = []
for key, row in combo_stats.iterrows():
    n, wins = int(row["n"]), int(round(row["n"] * row["win_rate"]))
    p_binom = stats.binomtest(wins, n, baseline_win_rate, alternative="two-sided").pvalue
    if p_binom < 0.05:
        sig_rows.append({"nicks_key": key, "n": n, "win_rate": row["win_rate"], "p": p_binom})
sig_df = pd.DataFrame(sig_rows).sort_values("p") if sig_rows else pd.DataFrame()
log(sig_df.to_string() if len(sig_df) else "(該当なし)")
log(f"該当件数: {len(sig_df)}/{len(eligible)}(多重検定未補正なのでp<0.05が"
    f"{len(eligible)}件中{int(len(eligible)*0.05)}件程度は偶然でも出る点に注意)")

log("\n" + "=" * 72)
log("まとめ(記述的検証のみ、シグナル化・採否判断は行わない)")
log("=" * 72)
log(f"1. インブリード係数: N={len(merged)}(2026-08-29時点の3,359件から拡大)でも"
    f"Spearman rho={rho:+.4f}(p={pval:.4f})。")
log(f"2. ニックス: N>={MIN_N}を満たす組み合わせは{len(eligible)}/{len(combo_counts)}件"
    f"({len(eligible)/len(combo_counts)*100:.1f}%)。多重検定未補正のp<0.05該当は{len(sig_df)}件。")

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_txt = OUT_DIR / "jra_pedigree_theory_2years_2026_08_30_report.txt"
out_txt.write_text("\n".join(lines), encoding="utf-8")
merged.to_csv(OUT_DIR / "jra_pedigree_theory_2years_2026_08_30_merged.csv", index=False, encoding="utf-8")
log(f"\nwrote {out_txt.name}")
