# -*- coding: utf-8 -*-
"""既存日程の`race_json_display/`(公開ページ表示専用の派生コピー)を、現在の
horse_baseline.pyパラメータで再生成する。旧`step4_regenerate_display.py`をCLI引数化して
1本化したもの(2026-08-18、scripts/jra_race_sim/publish/へ永続化)。

`generate_race_json.py`(新規開催日を一から生成)とは別目的: こちらは**表示専用コピーの
差分再生成**であり、判断根拠として使う`race_json/`本体は一切変更しない(パラメータ較正の
再現性を保つための既存ルール)。パラメータ較正を行った後、公開ページに反映したい時に使う。
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

import course_geometry_registry as geo

ENGINE_DIR = str(Path(__file__).resolve().parent.parent)
REPO_ROOT = Path(r"c:\Users\yuyou\Desktop\新しい作業場所")
DEFAULT_NEWSPAPER_DIR = REPO_ROOT / "data" / "newspaper"
DEFAULT_RACE_NAMES_DIR = REPO_ROOT / "data" / "jra_race_sim"
DEFAULT_OUT_DIR = Path(ENGINE_DIR) / "race_json_display"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", default="20260801,20260802", help="カンマ区切りYYYYMMDD")
    ap.add_argument("--race-names-dir", default=str(DEFAULT_RACE_NAMES_DIR))
    ap.add_argument("--newspaper-dir", default=str(DEFAULT_NEWSPAPER_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    dates = args.dates.split(",")
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = []
    for date in dates:
        race_names_csv = os.path.join(args.race_names_dir, "race_names_%s.csv" % date)
        df = pd.read_csv(race_names_csv, dtype={"race_id": str})
        target = df[df["surface"].notna()].copy()
        print("=== %s: 対象レース数 %d / %d (除外: %d) ===" % (
            date, len(target), len(df), len(df) - len(target)))

        for _, row in target.iterrows():
            race_id = row["race_id"]
            race_name = row["race_name"]
            racecourse = row["racecourse"]
            surface = geo.SURFACE_MAP.get(row["surface"], row["surface"])
            distance = float(row["distance_m"])
            straight = geo.is_straight_course(row)

            np_path = os.path.join(args.newspaper_dir, "%s.csv" % race_id)
            try:
                npdf = pd.read_csv(np_path, dtype=str)
                umaban_list = sorted(set(int(x) for x in npdf["umaban"]))
            except Exception as e:
                all_results.append({"date": date, "race_id": race_id, "race_name": race_name,
                                     "ok": False, "error": "newspaper CSV読込失敗: %s" % e})
                continue

            out_json = os.path.join(args.out_dir, "%s.json" % race_id)
            cmd = [sys.executable, os.path.join(ENGINE_DIR, "simulate_one_race.py"),
                   "--race-id", race_id, "--race-name", race_name,
                   "--racecourse", racecourse, "--surface", surface, "--distance", str(distance),
                   "--umaban-list", ",".join(str(u) for u in umaban_list),
                   "--out-json", out_json]
            geometry_note = "straight"
            if straight:
                cmd.append("--is-straight-course")
            else:
                circ, home, geometry_note = geo.resolve_geometry(racecourse, surface, distance)
                if circ is None:
                    all_results.append({"date": date, "race_id": race_id, "race_name": race_name,
                                         "ok": False, "error": "未知の競馬場/馬場: %s/%s" % (racecourse, surface)})
                    continue
                cmd += ["--circumference-m", str(circ), "--home-stretch-m", str(home)]

            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            elapsed = time.time() - t0
            ok = proc.returncode == 0
            all_results.append({
                "date": date, "race_id": race_id, "race_name": race_name, "racecourse": racecourse,
                "ok": ok, "elapsed": elapsed, "geometry_note": geometry_note,
                "error": proc.stderr.strip()[-500:] if not ok else "",
            })
            status = "OK  " if ok else "FAIL"
            print("[%s] %s %s %s (%.1fs)" % (status, race_id, race_name, racecourse, elapsed))
            if not ok:
                print("       error: %s" % proc.stderr.strip().replace("\n", " / ")[-500:])

    ok_list = [r for r in all_results if r["ok"]]
    fail_list = [r for r in all_results if not r["ok"]]
    print("\n=== 全体サマリ: 成功 %d / %d ===" % (len(ok_list), len(all_results)))
    if fail_list:
        print("失敗レース:")
        for r in fail_list:
            print("  %s %s %s: %s" % (r["date"], r["race_id"], r["race_name"], r.get("error", "")))
    print("\n出力先: %s (race_json/本体は一切変更していません)" % args.out_dir)


if __name__ == "__main__":
    main()
