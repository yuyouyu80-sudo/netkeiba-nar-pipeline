# -*- coding: utf-8 -*-
"""血統(種牡馬=父)ベースの適性プロファイル算出: 成熟年齢・推奨距離・適正馬場状態
(芝/ダート・馬場状態)・新馬戦適性・コース適性。

netkeibaが既に提供している`ca_sire_win_rate`等(course_data.py、そのレースのコース×距離
限定の父系成績)とは異なり、ここでは2024年〜2026年8月JRAの全レース結果(新馬戦も含む、
除外なし)と血統データ(data/pedigree/、父=ped_S_horse_id)を突き合わせ、**種牡馬ごとに
その産駒全体の実績を集計**して以下5種のプロファイルを導出する:

1. 推奨距離: 距離帯(短距離<=1400m/マイル1401-1800m/中距離1801-2200m/長距離>2200m)別の
   産駒勝率のうち最も高い帯を「推奨距離帯」とする(帯ごとにN>=MIN_SUBGROUP_Nを要求)。
2. 適正馬場状態(表面): 芝/ダートの産駒勝率を比較しどちらが得意かを見る。
3. 適正馬場状態(馬場状態): 良/稍重/重/不良4区分の産駒勝率を比較し最も高い区分を見る。
4. 新馬戦適性: race_nameに「新馬」を含むレースだけに絞った産駒勝率(早熟傾向の代理指標)。
5. コース適性: 競馬場別の産駒勝率のうち最も高い競馬場を見る。
6. (付随)成熟年齢: 産駒の「勝利時の平均年齢」と「全出走時の平均年齢」の差
   (mean_win_age - mean_all_age)。負に大きいほど若齢での勝利に偏る=早熟傾向、
   正に大きいほど高齢での勝利に偏る=晩成傾向、という記述的な代理指標。

**注意(このスクリプトの位置づけ)**: これは種牡馬の産駒全体を母集団にした記述的な集計であり、
個々の産駒(1頭)の将来成績を保証するものではない。的中率・回収率で検証されたシグナルでは
なく、あくまで血統データから機械的に導出した参考プロファイル。予想モデルへのシグナル
組み込みの可否判断はしていない(jra_pedigree_theory_*.pyと同じ立場)。
"""
import glob
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.netkeiba_pipeline.storage.paths import load_all_pedigree  # noqa: E402

OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\1ee9e967-1d8b-4f9a-9e82-dc5eee4b8f13\scratchpad"
)
RESULTS_DIR = PROJECT_ROOT / "data" / "race_results"

MIN_SIRE_N = 30       # このN未満の種牡馬はプロファイル対象外(信頼できない)
MIN_SUBGROUP_N = 10   # 距離帯/馬場/コース等の内訳セルがこのN未満なら「N不足」として推奨しない

DISTANCE_BUCKETS = [
    (0, 1400, "短距離(~1400m)"),
    (1401, 1800, "マイル(1401-1800m)"),
    (1801, 2200, "中距離(1801-2200m)"),
    (2201, 99999, "長距離(2201m~)"),
]


def distance_bucket(d: float) -> str:
    for lo, hi, label in DISTANCE_BUCKETS:
        if lo <= d <= hi:
            return label
    return "不明"


AGE_RE = re.compile(r"(\d+)")


def parse_age(sex_age: str) -> float:
    m = AGE_RE.search(str(sex_age))
    return float(m.group(1)) if m else float("nan")


print("血統データロード中...")
ped = load_all_pedigree()
sire_map = ped.set_index("horse_id")[["ped_S_horse_id", "ped_S_name_ja", "ped_S_name_en"]]


def sire_label(row) -> str:
    return row["ped_S_name_ja"] if pd.notna(row["ped_S_name_ja"]) and row["ped_S_name_ja"] else row["ped_S_name_en"]


sire_map = sire_map.assign(sire_label=sire_map.apply(sire_label, axis=1))
print(f"pedigreeデータ: {len(ped)}頭")

print("race_resultsロード中(2024-2026年8月、JRA、除外なし)...")
frames = []
for year in ("2024", "2025", "2026"):
    for p in sorted(glob.glob(str(RESULTS_DIR / year / "*.csv"))):
        frames.append(pd.read_csv(p, dtype=str, encoding="utf-8"))
results = pd.concat(frames, ignore_index=True)
print(f"race_results総行数: {len(results)}")

results["finish_pos_num"] = pd.to_numeric(results["finish_pos"], errors="coerce")
results = results.dropna(subset=["finish_pos_num"])
results["is_win"] = (results["finish_pos_num"] == 1).astype(int)
results["distance_m_num"] = pd.to_numeric(results["distance_m"], errors="coerce")
results["distance_bucket"] = results["distance_m_num"].map(distance_bucket)
results["age"] = results["sex_age"].map(parse_age)
results["is_debut"] = results["race_name"].str.contains("新馬", na=False)

merged = results.merge(
    sire_map.reset_index()[["horse_id", "ped_S_horse_id", "sire_label"]], on="horse_id", how="inner"
)
merged = merged.dropna(subset=["ped_S_horse_id"])
print(f"pedigree結合成功: {len(merged)}行 / {merged['ped_S_horse_id'].nunique()}種牡馬")

profiles = []
for sire_id, g in merged.groupby("ped_S_horse_id"):
    n = len(g)
    if n < MIN_SIRE_N:
        continue
    label = g["sire_label"].iloc[0]
    n_horses = g["horse_id"].nunique()
    overall_win_rate = g["is_win"].mean()

    # 1. 推奨距離
    dist_stats = g.groupby("distance_bucket").agg(n=("is_win", "size"), win_rate=("is_win", "mean"))
    dist_ok = dist_stats[dist_stats["n"] >= MIN_SUBGROUP_N]
    best_dist = dist_ok["win_rate"].idxmax() if len(dist_ok) else None
    best_dist_wr = dist_ok["win_rate"].max() if len(dist_ok) else None

    # 2. 芝/ダート適性
    surf_stats = g.groupby("surface").agg(n=("is_win", "size"), win_rate=("is_win", "mean"))
    surf_ok = surf_stats[surf_stats["n"] >= MIN_SUBGROUP_N]
    best_surf = surf_ok["win_rate"].idxmax() if len(surf_ok) else None
    best_surf_wr = surf_ok["win_rate"].max() if len(surf_ok) else None

    # 3. 馬場状態適性
    going_stats = g.groupby("going").agg(n=("is_win", "size"), win_rate=("is_win", "mean"))
    going_ok = going_stats[going_stats["n"] >= MIN_SUBGROUP_N]
    best_going = going_ok["win_rate"].idxmax() if len(going_ok) else None
    best_going_wr = going_ok["win_rate"].max() if len(going_ok) else None

    # 4. 新馬戦適性
    debut_g = g[g["is_debut"]]
    debut_n = len(debut_g)
    debut_win_rate = debut_g["is_win"].mean() if debut_n >= MIN_SUBGROUP_N else None

    # 5. コース適性
    course_stats = g.groupby("racecourse").agg(n=("is_win", "size"), win_rate=("is_win", "mean"))
    course_ok = course_stats[course_stats["n"] >= MIN_SUBGROUP_N]
    best_course = course_ok["win_rate"].idxmax() if len(course_ok) else None
    best_course_wr = course_ok["win_rate"].max() if len(course_ok) else None

    # 6. 成熟年齢の代理指標
    win_ages = g.loc[g["is_win"] == 1, "age"].dropna()
    all_ages = g["age"].dropna()
    age_skew = (win_ages.mean() - all_ages.mean()) if len(win_ages) >= 5 else None

    profiles.append({
        "sire_id": sire_id, "sire_name": label, "n_runs": n, "n_horses": n_horses,
        "overall_win_rate": overall_win_rate,
        "best_distance": best_dist, "best_distance_win_rate": best_dist_wr,
        "best_surface": best_surf, "best_surface_win_rate": best_surf_wr,
        "best_going": best_going, "best_going_win_rate": best_going_wr,
        "debut_n": debut_n, "debut_win_rate": debut_win_rate,
        "best_course": best_course, "best_course_win_rate": best_course_wr,
        "maturation_age_skew": age_skew,
    })

profile_df = pd.DataFrame(profiles).sort_values("n_runs", ascending=False)
print(f"\nプロファイル対象種牡馬数(N>={MIN_SIRE_N}): {len(profile_df)} "
      f"/ 全{merged['ped_S_horse_id'].nunique()}種牡馬")

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_csv = OUT_DIR / "jra_sire_aptitude_profile_2026_08_30.csv"
profile_df.to_csv(out_csv, index=False, encoding="utf-8")
print(f"wrote {out_csv}")

print("\n=== 産駒数上位20種牡馬のプロファイル(抜粋) ===")
cols = ["sire_name", "n_runs", "overall_win_rate", "best_distance", "best_surface",
        "best_going", "debut_win_rate", "best_course", "maturation_age_skew"]
print(profile_df[cols].head(20).to_string(index=False))

print("\n=== 早熟傾向トップ10(maturation_age_skewが小さい=マイナス方向、N>=30) ===")
early = profile_df.dropna(subset=["maturation_age_skew"]).sort_values("maturation_age_skew").head(10)
print(early[["sire_name", "n_runs", "maturation_age_skew", "debut_win_rate"]].to_string(index=False))

print("\n=== 晩成傾向トップ10(maturation_age_skewが大きい=プラス方向、N>=30) ===")
late = profile_df.dropna(subset=["maturation_age_skew"]).sort_values("maturation_age_skew", ascending=False).head(10)
print(late[["sire_name", "n_runs", "maturation_age_skew", "debut_win_rate"]].to_string(index=False))
