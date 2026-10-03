# -*- coding: utf-8 -*-
"""血統レポート(中山血統適性台帳・血統レーダーチャート中山8R)の全工程を順に再生成する。

前提: data/jra_pipeline/pedigree_commentary_profile.csv と pedigree_bucket_breakdown.csv が最新であること
(scripts/jra_model/jra_pedigree_commentary_profile_2026_09_25.py で再生成。父・母父・母の3role)。
入出力は data/jra_pipeline/pedigree_reports/ に置く(race_conditions_check.csvは取得に会員ログインが
必要なキャッシュのため入力としてGit管理、他の生成物は.gitignoreで除外)。
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    "build_bloodline_map",
    "build_race_fit_score",
    "build_race8_radar_data",
    "gen_race8_radar_report",
    "build_nakayama_today_table",
    "gen_nakayama_table_report",
]

if __name__ == "__main__":
    for step in STEPS:
        print(f"=== {step}", flush=True)
        rc = subprocess.run([sys.executable, "-u", str(HERE / f"{step}.py")]).returncode
        if rc:
            sys.exit(rc)
