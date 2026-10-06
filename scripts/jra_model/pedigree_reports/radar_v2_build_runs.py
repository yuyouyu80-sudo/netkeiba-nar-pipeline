# -*- coding: utf-8 -*-
"""血統レーダー v2 の R1: 2011〜2026年の全平地出走の表を作る(2026-10-06)。

事前登録 data/jra_pipeline/pedigree_reports/RADAR_V2_PREREG_2026_10_06.md §1 の規則(radar_v2_common.load_runs)に、
コース形態・内外回り(add_course_features)、ラップ由来のペース(lap_pace)、父・母父・父父のIDを付ける。
出力: radar_v2/runs_all.parquet(.gitignore対象)、radar_v2/runs_all_manifest.json(行数・年別件数・欠損率・入力ハッシュ)。
"""
import hashlib
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import radar_v2_common as C  # noqa: E402

OUT = C.OUT_DIR / "runs_all.parquet"
MANIFEST = C.OUT_DIR / "runs_all_manifest.json"


def input_hash() -> dict:
    files = sorted(glob.glob(str(C.RESULTS_DIR / "20*" / "*.csv")))
    h = hashlib.sha256()
    for p in files:
        h.update(Path(p).name.encode())
        h.update(hashlib.sha256(Path(p).read_bytes()).digest())
    return {"n_result_files": len(files), "results_sha256": h.hexdigest()}


def main():
    t0 = time.time()
    runs, meta = C.load_runs(max_year=None)
    runs = C.add_course_features(runs)
    pace = C.lap_pace(range(2011, 2027))
    runs = runs.merge(pace, on="race_id", how="left")
    ped = C.load_ped_ids(runs["horse_id"].unique().tolist(), C.OUT_DIR / "ped_ids_v2.parquet")
    runs = runs.merge(ped, on="horse_id", how="left")
    assert not runs.duplicated(["horse_id", "date"]).any()
    runs.to_parquet(OUT, index=False)

    man = {"built_at": time.strftime("%Y-%m-%d %H:%M"), **meta, **input_hash(), "n_runs": int(len(runs)),
           "n_races": int(runs["race_id"].nunique()), "max_date": str(runs["date"].max().date()),
           "runs_by_year": runs.groupby("year").size().to_dict(),
           "missing": {c: round(float(runs[c].isna().mean()), 4) for c in
                       ["front_raw", "kire_raw", "pace", "track_index_num", "going2", "rest_bucket", "class_ord",
                        "circ_tier", "ped_S_horse_id", "ped_DS_horse_id", "ped_SS_horse_id"]},
           "turn_type_counts": runs.drop_duplicates("race_id")["turn_type"].value_counts().to_dict(),
           "elapsed_sec": round(time.time() - t0, 1)}
    man["runs_by_year"] = {str(k): int(v) for k, v in man["runs_by_year"].items()}
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(man, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
