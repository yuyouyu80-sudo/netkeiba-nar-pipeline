# -*- coding: utf-8 -*-
"""血統理論(インブリード係数・ニックス)の記述的検証(Part 1)。33ラップ理論
(jra_lap33_theory_2026_08_28.py)と同じ方針: シグナル化・採否判断は行わず、まず実データで
相関・符号・サンプルサイズを記述的に確認するだけに留める。

対象母集団: 2026年7-8月のJRA通常戦(新馬・未勝利を除く、jra_dataset.pyと同じ除外条件)で、
data/pedigree/{horse_id}.csv が取得済みの馬。

## インブリード係数(Wright式、5代打ち切りの簡易版)
data/pedigree/{horse_id}.csv の62祖先スロット(ped_{path}_horse_id、path=S/Dのビット列)から、
同一horse_idが「父方(Sから始まる経路)」と「母方(Dから始まる経路)」の両方に出現するペアだけを
対象に F(X) = Σ (1/2)^(n1+n2+1) を計算する(n1,n2は各経路の長さ=世代数、共通祖先自身の
F_Aは0と近似)。同じ側(父方同士・母方同士)にしか出現しない重複は対象馬自身のインブリードには
寄与しない(その重複は対象馬の親のインブリードを示すだけ)ため除外する。5代打ち切りのため、
6代目以降にしか出ない共通祖先は捕捉できず、真のFを構造的に過小評価する(専門家指摘、
メモリproject_pedigree_fetch_design_2026_08_29参照)。

## ニックス(父×母父の組み合わせ)
netkeiba上に系統マスタが無いため、実施可能な最も細かい粒度(個体種牡馬×個体ブルードメアサイア、
ped_S_horse_id × ped_DS_horse_id)で組み合わせ頻度・勝率を集計する。note.com記事(73,119頭でも
「ニックスはサンプル不足で結論困難」)と同じ疎行列問題が、本プロジェクトの数百レース規模では
どの程度深刻かを記述するのが目的で、「勝率が高かった組み合わせ」を候補シグナルとして
採用するための探索ではない(組み合わせ数に対しN数が少ない中で偶然良く見えるセルを拾うことの
危険性は過去のNAR/JRA両方の自由探索の教訓と同じ)。
"""
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
RESULTS_DIR = PROJECT_ROOT / "data" / "race_results" / "2026"

lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    lines.append(str(s))


# ============================================================= インブリード係数
def inbreeding_coefficient(row: pd.Series) -> float:
    """rowはload_all_pedigree()の1行(pd.Series、文字列dtype)。"""
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


log("血統データロード中...")
ped = load_all_pedigree()
log(f"pedigreeデータ: {len(ped)}頭")

ped["inbreeding_f"] = ped.apply(inbreeding_coefficient, axis=1)
ped["nicks_key"] = list(zip(ped["ped_S_horse_id"], ped["ped_DS_horse_id"]))

log("\n" + "=" * 72)
log("インブリード係数F(X)の分布(5代打ち切り簡易版)")
log("=" * 72)
log(ped["inbreeding_f"].describe().to_string())
log(f"F=0(5代以内に父方/母方共通祖先なし)の馬: {(ped['inbreeding_f'] == 0).sum()}/{len(ped)} "
    f"({(ped['inbreeding_f'] == 0).mean()*100:.1f}%)")
log(f"F>0の馬: {(ped['inbreeding_f'] > 0).sum()}頭")
if (ped["inbreeding_f"] > 0).any():
    log("F>0馬の分布:\n" + ped.loc[ped["inbreeding_f"] > 0, "inbreeding_f"].describe().to_string())

# ============================================================= レース結果と結合
log("\n血統データを2026年7-8月JRA通常戦の実績と結合中...")

result_frames = []
for pattern in ("202607*.csv", "202608*.csv"):
    for p in sorted(RESULTS_DIR.glob(pattern)):
        df = pd.read_csv(p, dtype=str, encoding="utf-8")
        result_frames.append(df)
results = pd.concat(result_frames, ignore_index=True)
results = results[~results["race_name"].str.contains("新馬|未勝利", regex=True, na=False)]
results["finish_pos_num"] = pd.to_numeric(results["finish_pos"], errors="coerce")
results = results.dropna(subset=["finish_pos_num"])
results["popularity_num"] = pd.to_numeric(results["popularity"], errors="coerce")

field_size = results.groupby("race_id")["horse_id"].transform("count")
results["field_size"] = field_size
results["finish_pct"] = (results["finish_pos_num"] - 1) / (results["field_size"] - 1).clip(lower=1)
results["is_win"] = (results["finish_pos_num"] == 1).astype(int)

merged = results.merge(ped[["horse_id", "inbreeding_f", "nicks_key"]], on="horse_id", how="inner")
log(f"結合成功: {len(merged)}件(通常戦・完走・7-8月JRA、pedigree取得済み馬のみ)")

# ============================================================= インブリードF と着順の相関
log("\n" + "=" * 72)
log("インブリード係数F と 着順(finish_pct=0が1着、1が最下位相当) の記述的相関")
log("=" * 72)
rho, pval = stats.spearmanr(merged["inbreeding_f"], merged["finish_pct"])
log(f"Spearman順位相関: rho={rho:+.4f} p={pval:.4f} (N={len(merged)})")
log("(注: F<0が有利という理論なら rho は負を予想。5代打ち切りのためFは大半0近傍に集中)")

f_pos = merged[merged["inbreeding_f"] > 0]
f_zero = merged[merged["inbreeding_f"] == 0]
log(f"\nF=0群: N={len(f_zero)}  平均着順割合={f_zero['finish_pct'].mean():.4f}  勝率={f_zero['is_win'].mean()*100:.2f}%")
log(f"F>0群: N={len(f_pos)}  平均着順割合={f_pos['finish_pct'].mean():.4f}  勝率={f_pos['is_win'].mean()*100:.2f}%")
if len(f_pos) >= 5:
    u, up = stats.mannwhitneyu(f_zero["finish_pct"], f_pos["finish_pct"], alternative="two-sided")
    log(f"Mann-Whitney U検定(着順割合、F=0群 vs F>0群): p={up:.4f}")

# ============================================================= ニックス(個体種牡馬×個体BMS)
log("\n" + "=" * 72)
log("ニックス(個体種牡馬×個体ブルードメアサイアの組み合わせ)の記述的検証")
log("=" * 72)

combo_counts = merged.groupby("nicks_key").size().sort_values(ascending=False)
log(f"出現する組み合わせ数: {len(combo_counts)}  対象レコード数: {len(merged)}")
log(f"N=1の組み合わせ: {(combo_counts == 1).sum()} ({(combo_counts == 1).mean()*100:.1f}%)")
log(f"N>=3の組み合わせ: {(combo_counts >= 3).sum()}")
log(f"N>=5の組み合わせ: {(combo_counts >= 5).sum()}")
log(f"N>=10の組み合わせ: {(combo_counts >= 10).sum()}")

baseline_win_rate = merged["is_win"].mean()
log(f"\n母集団全体の勝率(ベースライン): {baseline_win_rate*100:.2f}% (N={len(merged)})")

MIN_N = 5
eligible = combo_counts[combo_counts >= MIN_N].index
log(f"\nN>={MIN_N}の組み合わせ({len(eligible)}件)の勝率一覧(降順、上位10件のみ表示。"
    "**候補シグナルとしての採用可否判断ではなく頻度分布の記述目的**):")
combo_stats = (
    merged[merged["nicks_key"].isin(eligible)]
    .groupby("nicks_key")
    .agg(n=("is_win", "size"), win_rate=("is_win", "mean"), mean_finish_pct=("finish_pct", "mean"))
    .sort_values("win_rate", ascending=False)
)
log(combo_stats.head(10).to_string())

log("\n" + "=" * 72)
log("まとめ(このスクリプトは記述的検証のみ、シグナル化・採否判断は行わない)")
log("=" * 72)
log(f"1. インブリード係数: 5代打ち切りのため{(ped['inbreeding_f']==0).mean()*100:.1f}%の馬がF=0"
    "(構造的な過小評価、6代目以降の共通祖先を捕捉できないため)。"
    f"F>0群と着順の相関はSpearman rho={rho:+.4f}(p={pval:.4f})。")
log(f"2. ニックス: {len(eligible)}/{len(combo_counts)}件のみがN>={MIN_N}を満たし、"
    f"{(combo_counts == 1).mean()*100:.1f}%はN=1(note.com記事が73,119頭規模でも指摘した"
    "疎行列問題が、本プロジェクトの規模ではさらに深刻に再現している)。")

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_txt = OUT_DIR / "jra_pedigree_theory_2026_08_29_report.txt"
out_txt.write_text("\n".join(lines), encoding="utf-8")
merged.to_csv(OUT_DIR / "jra_pedigree_theory_2026_08_29_merged.csv", index=False, encoding="utf-8")
log(f"\nwrote {out_txt.name}")
