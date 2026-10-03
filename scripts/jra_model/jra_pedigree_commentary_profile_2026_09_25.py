# -*- coding: utf-8 -*-
"""血統(父=種牡馬・母父=BMS)ベースの、予想レポート向け定性コメント生成用プロファイル。

jra_sire_aptitude_profile_2026_08_30.py(2024-2026年、父のみ、距離/芝ダート/馬場/新馬/コース/
成熟年齢の6種、scratchpad出力=非永続)を土台に、(1)母集団を2016-2026年全期間へ拡張、
(2)母父(BMS)も同一ロジックで併走、(3)回り方向・坂(高低差)・コース規模(周長)・洋芝・
重賞実績率・脚質(4角先頭率)・季節・休み明け実績の8種を追加、(4)出力をdata/jra_pipeline配下
(git管理下・恒久)に変更、した拡張版。

## 位置づけ(重要)
[[project_jra_pedigree_theory_verification_2026_08_29]]で統計的な予想シグナルとしての価値は
既に検証済み(インブリード係数rho≈0、種牡馬適性4シグナルはtrue_edge/sd全て不採用=-0.23〜-0.42)。
本スクリプトはそれとは別の目的で、「種牡馬・母父の産駒全体の実績を要約した記述統計」を予想
レポートに添える**定性コメント(血統評論スタイル)**として使うためのものであり、的中率・回収率で
検証された予想シグナルではない(jra_sire_aptitude_profile_2026_08_30.pyと同じ立場、time-safe
なpoint-in-time統計でもない=全期間の産駒実績をまとめて見せる記述的プロファイル)。

気性の激しさ・集中力・精神的タフさ・操縦性・揉まれ弱さ・ゲートセンス・海外遠征適応力・
地方ダート(深い砂)適性は、血統表(祖先の名前・生年のみ)からは直接算出できないため扱わない
(2026-09-25にユーザーへ提示した血統ファクター30項目のうち、この8項目はコメント生成対象外と
した上で本スクリプトを設計している)。

出力: data/jra_pipeline/pedigree_commentary_profile.csv
1行 = (ancestor_role: sire/bms, ancestor_id) のプロファイル。

## role="dam"の追加(2026-09-27、ユーザー指摘「母・母母・父母も含めてほしい」対応)
父・母父と全く同じ集計方式で、対象馬自身の直接の母(ped_D_horse_id)を軸に産駒成績を集計する
role="dam"を追加。母馬は生涯の産駒数が少ない(既存調査: 全18,016頭中央値2頭、5頭以上は
3,837頭のみ)ため、MIN_ANCESTOR_N=30を満たす母馬はごく一部に限られる(既知の統計的制約、
ユーザーへ明示の上で「可能性を全て入れる」方針により追加)。この母馬プロファイルを、
対象馬の母(D)だけでなく母母(DD)・父母(SD)のIDでも同じrole="dam"としてルックアップすることで、
「その祖先が自身の産駒/係累を持っていれば」条件付きで第4〜6の評価軸として使う。
"""
import glob
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.netkeiba_pipeline.storage.paths import load_all_pedigree  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "data" / "race_results"
MIN_RESULTS_YEAR = 2016
COURSE_MASTER_PATH = PROJECT_ROOT / "data" / "jra_course_master.csv"
OUT_PATH = PROJECT_ROOT / "data" / "jra_pipeline" / "pedigree_commentary_profile.csv"
BREAKDOWN_OUT_PATH = PROJECT_ROOT / "data" / "jra_pipeline" / "pedigree_bucket_breakdown.csv"

# category_breakdown()が「最高勝率の1区分」だけを返すのに対し、こちらは同じ集計から
# 全区分のwin_rate/Nを保持する(2026-09-26、レース別の実際の条件と突き合わせるため新設)。
BREAKDOWN_DIMENSIONS = [
    ("distance", "distance_bucket", False),
    ("surface", "surface", False),
    ("going", "going", False),
    ("course", "racecourse", False),
    ("turn", "turn_direction", False),
    ("elevation", "elevation_tier", False),
    ("circumference", "circumference_tier", False),
    ("season", "season", False),
    ("rest", "rest_bucket", False),
    ("turf_type", "turf_type_local", True),  # True = 芝のみの部分集合(turf_g)で集計
]

MIN_ANCESTOR_N = 30   # このN未満の父/母父はプロファイル対象外(信頼できない)
MIN_SUBGROUP_N = 10   # 内訳セルがこのN未満なら「N不足」として言及しない

DISTANCE_BUCKETS = [
    (0, 1400, "短距離(~1400m)"),
    (1401, 1800, "マイル(1401-1800m)"),
    (1801, 2200, "中距離(1801-2200m)"),
    (2201, 99999, "長距離(2201m~)"),
]

SEASON_MAP = {
    3: "春", 4: "春", 5: "春",
    6: "夏", 7: "夏", 8: "夏",
    9: "秋", 10: "秋", 11: "秋",
    12: "冬", 1: "冬", 2: "冬",
}

REST_BUCKETS = [
    (0, 21, "連闘・中1-2週(21日以下)"),
    (22, 89, "中間隔(22-89日)"),
    (90, 99999, "休み明け(90日以上)"),
]


def distance_bucket(d: float) -> str:
    for lo, hi, label in DISTANCE_BUCKETS:
        if lo <= d <= hi:
            return label
    return "不明"


def rest_bucket(days: float) -> str:
    if pd.isna(days):
        return None
    for lo, hi, label in REST_BUCKETS:
        if lo <= days <= hi:
            return label
    return None


def category_breakdown(g: pd.DataFrame, col: str, min_n: int = MIN_SUBGROUP_N):
    """gのcol列で勝率が最も高い区分(N>=min_nのものに限る)を返す。
    戻り値: (best_label, best_win_rate, best_n) または対象なしなら(None, None, None)。"""
    sub = g.dropna(subset=[col])
    if len(sub) == 0:
        return None, None, None
    stats = sub.groupby(col).agg(n=("is_win", "size"), win_rate=("is_win", "mean"))
    ok = stats[stats["n"] >= min_n]
    if len(ok) == 0:
        return None, None, None
    best = ok["win_rate"].idxmax()
    return best, float(ok.loc[best, "win_rate"]), int(ok.loc[best, "n"])


def category_breakdown_full(g: pd.DataFrame, col: str, min_n: int = MIN_SUBGROUP_N) -> pd.DataFrame:
    """category_breakdown()と同じ集計だが、最高勝率の1区分だけでなく全区分(N>=min_n)の
    win_rate/Nを返す。列: bucket_label, n, win_rate。"""
    sub = g.dropna(subset=[col])
    if len(sub) == 0:
        return pd.DataFrame(columns=["bucket_label", "n", "win_rate"])
    stats = sub.groupby(col).agg(n=("is_win", "size"), win_rate=("is_win", "mean"))
    ok = stats[stats["n"] >= min_n].reset_index().rename(columns={col: "bucket_label"})
    return ok[["bucket_label", "n", "win_rate"]]


def subset_win_rate(g: pd.DataFrame, mask: pd.Series, min_n: int = MIN_SUBGROUP_N):
    sub = g[mask]
    if len(sub) < min_n:
        return len(sub), None
    return len(sub), float(sub["is_win"].mean())


def build_profile_for_role(
    merged: pd.DataFrame, id_col: str, name_col: str, role: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """merged: 1頭1レース1行、id_col=父/母父horse_id列、name_col=表示名列を持つ全出走行。
    ancestor_id(=id_col)ごとに集計してプロファイルDataFrameを返す。
    戻り値は(profile_df, breakdown_df)のタプル。breakdown_dfはBREAKDOWN_DIMENSIONS各項目の
    全区分内訳(long format、レース別の実際の条件と突き合わせる用途、2026-09-26新設)。"""
    profiles = []
    breakdown_rows = []
    for ancestor_id, g in merged.dropna(subset=[id_col]).groupby(id_col):
        n = len(g)
        if n < MIN_ANCESTOR_N:
            continue
        label = g[name_col].iloc[0]
        n_horses = g["horse_id"].nunique()
        overall_win_rate = g["is_win"].mean()

        best_distance, best_distance_wr, best_distance_n = category_breakdown(g, "distance_bucket")
        best_surface, best_surface_wr, best_surface_n = category_breakdown(g, "surface")
        best_going, best_going_wr, best_going_n = category_breakdown(g, "going")
        best_course, best_course_wr, best_course_n = category_breakdown(g, "racecourse")
        best_turn, best_turn_wr, best_turn_n = category_breakdown(g, "turn_direction")
        best_elevation, best_elevation_wr, best_elevation_n = category_breakdown(g, "elevation_tier")
        best_circumference, best_circumference_wr, best_circumference_n = category_breakdown(
            g, "circumference_tier"
        )
        best_season, best_season_wr, best_season_n = category_breakdown(g, "season")
        best_rest, best_rest_wr, best_rest_n = category_breakdown(g, "rest_bucket")

        turf_g = g[g["surface"] == "芝"]
        best_turf_type, best_turf_type_wr, best_turf_type_n = category_breakdown(turf_g, "turf_type_local")

        for dimension, col, turf_only in BREAKDOWN_DIMENSIONS:
            src_g = turf_g if turf_only else g
            bd = category_breakdown_full(src_g, col)
            for _, row in bd.iterrows():
                breakdown_rows.append({
                    "ancestor_role": role, "horse_id_ancestor": ancestor_id, "ancestor_name": label,
                    "dimension": dimension, "bucket_label": row["bucket_label"],
                    "win_rate": row["win_rate"], "n": int(row["n"]),
                })

        debut_n, debut_wr = subset_win_rate(g, g["is_debut"])
        graded_n, graded_wr = subset_win_rate(g, g["is_graded"])
        leader_n, leader_wr = subset_win_rate(g, g["corner4_is_leader"] == True)  # noqa: E712
        non_leader_n, non_leader_wr = subset_win_rate(g, g["corner4_is_leader"] == False)  # noqa: E712

        win_ages = g.loc[g["is_win"] == 1, "age"].dropna()
        all_ages = g["age"].dropna()
        age_skew = (win_ages.mean() - all_ages.mean()) if len(win_ages) >= 5 else None

        profiles.append({
            "ancestor_role": role, "ancestor_id": ancestor_id, "ancestor_name": label,
            "n_runs": n, "n_horses": n_horses, "overall_win_rate": overall_win_rate,
            "best_distance": best_distance, "best_distance_win_rate": best_distance_wr,
            "best_distance_n": best_distance_n,
            "best_surface": best_surface, "best_surface_win_rate": best_surface_wr,
            "best_surface_n": best_surface_n,
            "best_going": best_going, "best_going_win_rate": best_going_wr, "best_going_n": best_going_n,
            "best_course": best_course, "best_course_win_rate": best_course_wr, "best_course_n": best_course_n,
            "best_turn": best_turn, "best_turn_win_rate": best_turn_wr, "best_turn_n": best_turn_n,
            "best_elevation": best_elevation, "best_elevation_win_rate": best_elevation_wr,
            "best_elevation_n": best_elevation_n,
            "best_circumference": best_circumference, "best_circumference_win_rate": best_circumference_wr,
            "best_circumference_n": best_circumference_n,
            "best_turf_type": best_turf_type, "best_turf_type_win_rate": best_turf_type_wr,
            "best_turf_type_n": best_turf_type_n,
            "best_season": best_season, "best_season_win_rate": best_season_wr, "best_season_n": best_season_n,
            "best_rest": best_rest, "best_rest_win_rate": best_rest_wr, "best_rest_n": best_rest_n,
            "debut_n": debut_n, "debut_win_rate": debut_wr,
            "graded_n": graded_n, "graded_win_rate": graded_wr,
            "leader_n": leader_n, "leader_win_rate": leader_wr,
            "non_leader_n": non_leader_n, "non_leader_win_rate": non_leader_wr,
            "maturation_age_skew": age_skew,
        })
    return pd.DataFrame(profiles), pd.DataFrame(breakdown_rows)


def main() -> None:
    print("血統データロード中...")
    ped = load_all_pedigree()

    def label(row, id_col, name_ja_col, name_en_col):
        v = row[name_ja_col]
        return v if pd.notna(v) and v else row[name_en_col]

    sire_map = ped[["horse_id", "ped_S_horse_id", "ped_S_name_ja", "ped_S_name_en"]].copy()
    sire_map["sire_label"] = sire_map.apply(
        lambda r: label(r, "ped_S_horse_id", "ped_S_name_ja", "ped_S_name_en"), axis=1
    )
    bms_map = ped[["horse_id", "ped_DS_horse_id", "ped_DS_name_ja", "ped_DS_name_en"]].copy()
    bms_map["bms_label"] = bms_map.apply(
        lambda r: label(r, "ped_DS_horse_id", "ped_DS_name_ja", "ped_DS_name_en"), axis=1
    )
    dam_map = ped[["horse_id", "ped_D_horse_id", "ped_D_name_ja", "ped_D_name_en"]].copy()
    dam_map["dam_label"] = dam_map.apply(
        lambda r: label(r, "ped_D_horse_id", "ped_D_name_ja", "ped_D_name_en"), axis=1
    )
    print(f"pedigreeデータ: {len(ped)}頭")

    print("コースマスターロード中...")
    course_master = pd.read_csv(COURSE_MASTER_PATH, dtype=str, encoding="utf-8")
    course_master["turf_elevation_m_num"] = pd.to_numeric(course_master["turf_elevation_m"], errors="coerce")
    course_master["dirt_elevation_m_num"] = pd.to_numeric(course_master["dirt_elevation_m"], errors="coerce")
    course_master["turf_circumference_m_num"] = pd.to_numeric(
        course_master["turf_circumference_m"], errors="coerce"
    )
    course_master["dirt_circumference_m_num"] = pd.to_numeric(
        course_master["dirt_circumference_m"], errors="coerce"
    )
    turf_elev_median = course_master["turf_elevation_m_num"].median()
    dirt_elev_median = course_master["dirt_elevation_m_num"].median()
    turf_circ_median = course_master["turf_circumference_m_num"].median()
    dirt_circ_median = course_master["dirt_circumference_m_num"].median()
    print(
        f"  坂の高低差中央値: 芝{turf_elev_median:.1f}m / ダート{dirt_elev_median:.1f}m、"
        f"周長中央値: 芝{turf_circ_median:.0f}m / ダート{dirt_circ_median:.0f}m"
    )
    YOZHIBA_VENUES = {"札幌", "函館"}  # 洋芝(冷涼地系の芝)を使用する2場

    print("race_resultsロード中(2016-2026年、JRA全期間、除外なし)...")
    frames = []
    for year_dir in sorted(RESULTS_DIR.glob("*")):
        if not (year_dir.is_dir() and year_dir.name.isdigit() and int(year_dir.name) >= MIN_RESULTS_YEAR):
            continue  # 収集側の過去バックフィル(2011年〜)や nar/ を取り込まず、2016年以降に固定する
        for p in sorted(glob.glob(str(year_dir / "*.csv"))):
            frames.append(pd.read_csv(p, dtype=str, encoding="utf-8"))
    results = pd.concat(frames, ignore_index=True)
    print(f"race_results総行数: {len(results)}")

    results["finish_pos_num"] = pd.to_numeric(results["finish_pos"], errors="coerce")
    results = results.dropna(subset=["finish_pos_num"]).copy()
    results["is_win"] = (results["finish_pos_num"] == 1).astype(int)
    results["distance_m_num"] = pd.to_numeric(results["distance_m"], errors="coerce")
    results["distance_bucket"] = results["distance_m_num"].map(distance_bucket)
    results["age"] = pd.to_numeric(results["sex_age"].str.extract(r"(\d+)")[0], errors="coerce")
    results["is_debut"] = results["race_name"].str.contains("新馬", na=False)
    results["is_graded"] = results["race_name"].str.contains(r"\(G[I]{1,3}\)", regex=True, na=False)
    results["corner4_is_leader"] = results["corner4_is_leader"].map({"True": True, "False": False})
    results["race_date_dt"] = pd.to_datetime(results["race_date"], errors="coerce")
    results["season"] = results["race_date_dt"].dt.month.map(SEASON_MAP)

    # 回り方向・坂の高低差(サーフェスに応じて芝/ダート別の値・中央値を使う)・コース規模
    cm = course_master.set_index("venue")
    results = results.merge(
        cm[["turn_direction", "turf_elevation_m_num", "dirt_elevation_m_num",
            "turf_circumference_m_num", "dirt_circumference_m_num"]],
        left_on="racecourse", right_index=True, how="left",
    )

    def elevation_tier(row):
        # 2026-09-26修正: 実データのダート表記は"ダート"でなく"ダ"だったため、この分岐が
        # 一度もマッチせず坂適性がダート産駒で常にNone(NaN)になっていたバグを修正。
        if row["surface"] == "芝":
            v, med = row["turf_elevation_m_num"], turf_elev_median
        elif row["surface"] == "ダ":
            v, med = row["dirt_elevation_m_num"], dirt_elev_median
        else:
            return None
        if pd.isna(v):
            return None
        return "急坂型(高低差大)" if v >= med else "平坦型(高低差小)"

    def circumference_tier(row):
        # 2026-09-26修正: elevation_tierと同じ"ダート"→"ダ"表記バグ。
        if row["surface"] == "芝":
            v, med = row["turf_circumference_m_num"], turf_circ_median
        elif row["surface"] == "ダ":
            v, med = row["dirt_circumference_m_num"], dirt_circ_median
        else:
            return None
        if pd.isna(v):
            return None
        return "広いコース型(周長大)" if v >= med else "小回り型(周長小)"

    results["elevation_tier"] = results.apply(elevation_tier, axis=1)
    results["circumference_tier"] = results.apply(circumference_tier, axis=1)
    results["turf_type_local"] = results["racecourse"].map(
        lambda v: "洋芝" if v in YOZHIBA_VENUES else "野芝"
    )

    # 休養日数(JRA間隔のみ、NAR跨ぎ・海外跨ぎは考慮しない=それらは別途「参考値」扱い)
    results = results.sort_values(["horse_id", "race_date_dt"])
    results["prev_race_date"] = results.groupby("horse_id")["race_date_dt"].shift(1)
    results["rest_days"] = (results["race_date_dt"] - results["prev_race_date"]).dt.days
    results["rest_bucket"] = results["rest_days"].map(rest_bucket)

    sire_merged = results.merge(
        sire_map[["horse_id", "ped_S_horse_id", "sire_label"]], on="horse_id", how="inner"
    ).dropna(subset=["ped_S_horse_id"])
    bms_merged = results.merge(
        bms_map[["horse_id", "ped_DS_horse_id", "bms_label"]], on="horse_id", how="inner"
    ).dropna(subset=["ped_DS_horse_id"])
    dam_merged = results.merge(
        dam_map[["horse_id", "ped_D_horse_id", "dam_label"]], on="horse_id", how="inner"
    ).dropna(subset=["ped_D_horse_id"])
    print(
        f"pedigree結合成功: 父{len(sire_merged)}行/{sire_merged['ped_S_horse_id'].nunique()}種牡馬、"
        f"母父{len(bms_merged)}行/{bms_merged['ped_DS_horse_id'].nunique()}頭、"
        f"母{len(dam_merged)}行/{dam_merged['ped_D_horse_id'].nunique()}頭"
    )

    print("父(種牡馬)プロファイル集計中...")
    sire_profile, sire_breakdown = build_profile_for_role(sire_merged, "ped_S_horse_id", "sire_label", "sire")
    print(f"  対象種牡馬数(N>={MIN_ANCESTOR_N}): {len(sire_profile)}")

    print("母父(BMS)プロファイル集計中...")
    bms_profile, bms_breakdown = build_profile_for_role(bms_merged, "ped_DS_horse_id", "bms_label", "bms")
    print(f"  対象母父数(N>={MIN_ANCESTOR_N}): {len(bms_profile)}")

    print("母(繁殖牝馬)プロファイル集計中...")
    dam_profile, dam_breakdown = build_profile_for_role(dam_merged, "ped_D_horse_id", "dam_label", "dam")
    print(f"  対象母数(N>={MIN_ANCESTOR_N}): {len(dam_profile)}"
          f"(母集団{dam_merged['ped_D_horse_id'].nunique()}頭中、産駒数の少なさによりごく一部)")

    out = pd.concat([sire_profile, bms_profile, dam_profile], ignore_index=True)
    out = out.rename(columns={"ancestor_id": "horse_id_ancestor"})
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"\nwrote {OUT_PATH} ({len(out)}行)")

    breakdown = pd.concat([sire_breakdown, bms_breakdown, dam_breakdown], ignore_index=True)
    breakdown.to_csv(BREAKDOWN_OUT_PATH, index=False, encoding="utf-8")
    print(f"wrote {BREAKDOWN_OUT_PATH} ({len(breakdown)}行)")


if __name__ == "__main__":
    main()
