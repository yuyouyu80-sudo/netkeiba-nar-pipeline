# -*- coding: utf-8 -*-
"""今日(2026-09-26)の中山開催12レース全頭について、血統評論プロファイル
(data/jra_pipeline/pedigree_commentary_profile.csv)から14項目を抽出し、
HTMLレポート用のJSONを組み立てる。"""
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from src.netkeiba_pipeline.storage.paths import pedigree_csv_path  # noqa: E402

PROFILE_PATH = PROJECT_ROOT / "data" / "jra_pipeline" / "pedigree_commentary_profile.csv"
NEWSPAPER_DIR = PROJECT_ROOT / "data" / "newspaper"
SCRATCH = PROJECT_ROOT / "data" / "jra_pipeline" / "pedigree_reports"
SCRATCH.mkdir(parents=True, exist_ok=True)
OUT_PATH = SCRATCH / "nakayama_today_data.json"

RACE_IDS = [f"2026060408{n:02d}" for n in range(1, 13)]

FACTOR_DEFS = [
    ("surface", "芝・ダート適性", "best_surface", "best_surface_win_rate", "best_surface_n"),
    ("distance", "距離適性", "best_distance", "best_distance_win_rate", "best_distance_n"),
    ("going", "道悪(馬場状態)適性", "best_going", "best_going_win_rate", "best_going_n"),
    ("turf_type", "洋芝/野芝", "best_turf_type", "best_turf_type_win_rate", "best_turf_type_n"),
    ("course", "得意競馬場", "best_course", "best_course_win_rate", "best_course_n"),
    ("turn", "右回り/左回り", "best_turn", "best_turn_win_rate", "best_turn_n"),
    ("elevation", "坂適性(高低差)", "best_elevation", "best_elevation_win_rate", "best_elevation_n"),
    ("circumference", "コース規模", "best_circumference", "best_circumference_win_rate", "best_circumference_n"),
    ("season", "得意季節", "best_season", "best_season_win_rate", "best_season_n"),
    ("rest", "休み明け/連闘適性", "best_rest", "best_rest_win_rate", "best_rest_n"),
]


def _num(v):
    if pd.isna(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def role_factor_block(row):
    if row is None:
        return None
    block = {
        "n_runs": int(row["n_runs"]),
        "n_horses": int(row["n_horses"]),
        "overall_win_rate": _num(row["overall_win_rate"]),
        "factors": {},
    }
    for key, _label, val_col, wr_col, n_col in FACTOR_DEFS:
        val = row.get(val_col)
        if pd.isna(val):
            block["factors"][key] = None
        else:
            block["factors"][key] = {
                "value": val,
                "win_rate": _num(row[wr_col]),
                "n": int(row[n_col]),
            }
    block["debut"] = None
    if pd.notna(row.get("debut_win_rate")):
        block["debut"] = {"win_rate": _num(row["debut_win_rate"]), "n": int(row["debut_n"])}
    block["graded"] = None
    if pd.notna(row.get("graded_win_rate")):
        block["graded"] = {"win_rate": _num(row["graded_win_rate"]), "n": int(row["graded_n"])}
    block["running_style"] = None
    if pd.notna(row.get("leader_win_rate")) and pd.notna(row.get("non_leader_win_rate")):
        block["running_style"] = {
            "leader_win_rate": _num(row["leader_win_rate"]),
            "leader_n": int(row["leader_n"]),
            "non_leader_win_rate": _num(row["non_leader_win_rate"]),
            "non_leader_n": int(row["non_leader_n"]),
        }
    block["maturation_age_skew"] = _num(row.get("maturation_age_skew"))
    return block


def main():
    profile = pd.read_csv(PROFILE_PATH, dtype={"horse_id_ancestor": str}, encoding="utf-8")
    sire_profile = profile[profile["ancestor_role"] == "sire"].set_index("horse_id_ancestor")
    bms_profile = profile[profile["ancestor_role"] == "bms"].set_index("horse_id_ancestor")

    races_out = []
    for race_id in RACE_IDS:
        path = NEWSPAPER_DIR / f"{race_id}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
        df["_umaban_num"] = pd.to_numeric(df["umaban"], errors="coerce")
        df = df.sort_values("_umaban_num")

        horses_out = []
        for _, r in df.iterrows():
            horse_id = r["horse_id"]
            ped_path = pedigree_csv_path(horse_id)
            sire_id = bms_id = sire_name = bms_name = None
            sire_block = bms_block = None
            if ped_path.exists():
                ped = pd.read_csv(ped_path, dtype=str, encoding="utf-8").iloc[0]
                sire_id = ped.get("ped_S_horse_id")
                bms_id = ped.get("ped_DS_horse_id")
                sire_name = ped.get("ped_S_name_ja")
                bms_name = ped.get("ped_DS_name_ja")
                if pd.notna(sire_id) and sire_id in sire_profile.index:
                    sire_block = role_factor_block(sire_profile.loc[sire_id])
                if pd.notna(bms_id) and bms_id in bms_profile.index:
                    bms_block = role_factor_block(bms_profile.loc[bms_id])

            horses_out.append({
                "umaban": int(r["umaban"]),
                "waku": int(r["waku"]),
                "horse_id": horse_id,
                "horse_name": r["horse_name"],
                "sire_name": sire_name if pd.notna(sire_name) else None,
                "bms_name": bms_name if pd.notna(bms_name) else None,
                "sire": sire_block,
                "bms": bms_block,
            })

        races_out.append({
            "race_id": race_id,
            "race_number": int(race_id[-2:]),
            "horses": horses_out,
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"factor_defs": FACTOR_DEFS, "races": races_out}, f, ensure_ascii=False, indent=1)
    print(f"wrote {OUT_PATH}: {len(races_out)}races")
    total_horses = sum(len(r["horses"]) for r in races_out)
    print(f"total horses: {total_horses}")


if __name__ == "__main__":
    main()
