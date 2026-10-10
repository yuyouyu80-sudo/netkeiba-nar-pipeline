# -*- coding: utf-8 -*-
"""finish_v4 F1: 生年月日・生産者の表と、月齢を使ってよいかの基準(カバー率のゲート)(2026-10-11、プラン第3版 F1)。

- 入力: data/horse_profile/{horse_id}.csv(収集用の会話が取得中。ここでは読むだけ)。
- 出力: data/jra_pipeline/finish_v4/profile.parquet(Git管理外)と birth_coverage.json。
- 使う列: 生年月日・生産者。馬主は使わない(取得した時点の馬主で、過去のレースの時点の値ではないため)。
- 外国産: horse_id の5〜6桁目が "10" 以外(内国産は "10"。"11" などは外国産で、南半球産は7〜12月生まれがある)。
  月齢の効果は内国産に限るか、外国産の指示変数を付ける(F2-6)。
- **月齢を使う基準(プランで固定):** 2014〜2020年の2〜3歳の走で、生年月日がある走の割合が98%以上。
  あわせて、レース単位の完全ケース(出走馬全員の生年月日がある)の割合と、欠け方が着順・キャリア・生産者・生まれ年で
  偏っていないかを出す。欠けを示す指示変数や平均埋めは使わない(欠け方が取得の順序=将来の経歴と関係し、漏れになるため)。

使い方:
  python birth_data.py            # profile.parquet を作り直し、カバー率を出す
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pedigree_reports"))
import radar_v2_aptitude as A  # noqa: E402
import radar_v2_common as C  # noqa: E402

PROFILE_DIR = C.PROJECT_ROOT / "data" / "horse_profile"
OUT_DIR = C.PROJECT_ROOT / "data" / "jra_pipeline" / "finish_v4"
PROFILE_OUT = OUT_DIR / "profile.parquet"
COV_OUT = OUT_DIR / "birth_coverage.json"
GATE = 0.98
GATE_YEARS = (2014, 2020)


def build_profile() -> pd.DataFrame:
    rows = []
    for p in PROFILE_DIR.glob("*.csv"):
        with open(p, encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            hdr = next(r, None)
            row = next(r, None)
        if not hdr or not row:
            continue
        d = dict(zip(hdr, row))
        rows.append({"horse_id": d.get("horse_id") or p.stem, "birth_date": d.get("birth_date") or None,
                     "breeder": d.get("breeder") or None, "breeder_id": d.get("breeder_id") or None})
    P = pd.DataFrame(rows).drop_duplicates("horse_id")
    P["birth_date"] = pd.to_datetime(P["birth_date"], errors="coerce")
    P["foreign"] = P["horse_id"].str[4:6] != "10"
    P["birth_month"] = P["birth_date"].dt.month
    P["year_ok"] = P["birth_date"].dt.year.astype("Int64").astype(str) == P["horse_id"].str[:4]
    P.to_parquet(PROFILE_OUT, index=False)
    return P


def coverage(P: pd.DataFrame) -> dict:
    runs = pd.read_parquet(A.RUNS, columns=["race_id", "horse_id", "year", "age", "pos", "career", "date"])
    runs = runs[runs["year"].between(*GATE_YEARS)]
    assert runs["year"].max() <= 2020
    bd = P.set_index("horse_id")["birth_date"]
    br = P.set_index("horse_id")["breeder_id"]
    runs["has_file"] = runs["horse_id"].isin(P["horse_id"])
    runs["has_bd"] = runs["horse_id"].map(bd).notna()
    runs["has_breeder"] = runs["horse_id"].map(br).notna()
    y23 = runs[runs["age"].between(2, 3)]
    race_cc = y23.groupby("race_id")["has_bd"].all()

    def tab(df, col, bins=None, labels=None):
        key = pd.cut(df[col], bins, labels=labels).astype(str) if bins else df[col].astype(str)
        g = df.groupby(key)["has_bd"].agg(["mean", "size"])
        return {k: {"rate": round(float(v["mean"]), 4), "n": int(v["size"])} for k, v in g.iterrows()}
    out = {"made": time.strftime("%Y-%m-%d %H:%M"), "n_profiles": int(len(P)),
           "profiles_with_birth_date": int(P["birth_date"].notna().sum()),
           "birth_year_matches_id": round(float(P.loc[P["birth_date"].notna(), "year_ok"].mean()), 4),
           "foreign": {"n": int(P["foreign"].sum()),
                       "born_jul_dec": int((P["foreign"] & (P["birth_month"] >= 7)).sum()),
                       "domestic_born_jul_dec": int((~P["foreign"] & (P["birth_month"] >= 7)).sum())},
           "birth_month_domestic": {int(k): int(v) for k, v in P.loc[~P["foreign"], "birth_month"].value_counts().sort_index().items()},
           "gate": {"years": GATE_YEARS, "ages": "2〜3歳", "threshold": GATE,
                    "run_coverage": round(float(y23["has_bd"].mean()), 4), "n_runs": int(len(y23)),
                    "race_complete_case": round(float(race_cc.mean()), 4), "n_races": int(len(race_cc)),
                    "file_coverage": round(float(y23["has_file"].mean()), 4)},
           "all_ages_run_coverage": round(float(runs["has_bd"].mean()), 4),
           "breeder_run_coverage": round(float(runs["has_breeder"].mean()), 4),
           "by_year": tab(y23, "year"),
           "by_pos": tab(y23, "pos", [0, 3, 9, 30], ["1〜3着", "4〜9着", "10着以下"]),
           "by_career": tab(y23, "career", [-1, 0, 3, 9, 1000], ["初出走", "1〜3走", "4〜9走", "10走〜"]),
           "by_birth_year": tab(y23.assign(by=y23["horse_id"].str[:4]), "by")}
    out["gate"]["passed"] = bool(out["gate"]["run_coverage"] >= GATE)
    rates = [v["rate"] for v in out["by_pos"].values()]
    out["gate"]["pos_spread"] = round(float(max(rates) - min(rates)), 4)
    return out


def main():
    t0 = time.time()
    P = build_profile()
    cov = coverage(P)
    cov["elapsed_sec"] = round(time.time() - t0, 1)
    COV_OUT.write_text(json.dumps(cov, ensure_ascii=False, indent=1), encoding="utf-8")
    g = cov["gate"]
    print("profiles", cov["n_profiles"], "birth", cov["profiles_with_birth_date"], "run_cov", g["run_coverage"],
          "race_cc", g["race_complete_case"], "passed", g["passed"])


if __name__ == "__main__":
    main()
