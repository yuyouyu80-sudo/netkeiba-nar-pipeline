# -*- coding: utf-8 -*-
"""新規開催日の`race_json/`(相互作用シミュレーション結果、コースアニメーション表示用)を
一から生成するオーケストレーター。旧`build_all_races_multi.py`をCLI引数化して1本化したもの
(2026-08-18、scripts/jra_race_sim/publish/へ永続化)。

生成済みの`race_json/{race_id}.json`は「較正の判断根拠として使った確定済みデータ」という
既存ルール(旧docstring: 8/2分は一切変更しない)を踏襲し、**このスクリプトの再実行では
既存日程を上書きしない**(--datesに新しい日程だけを指定すること。既存日程を意図的に
作り直したい場合のみ--forceを付ける)。

新パラメータでの見た目だけの再生成(既存日程の`race_json_display/`更新)は
`regenerate_race_json_display.py`を使う(このスクリプトとは別目的、両方必要)。
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
DEFAULT_OUT_DIR = Path(ENGINE_DIR) / "race_json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", required=True, help="カンマ区切りYYYYMMDD(複数可)")
    ap.add_argument("--race-names-dir", default=str(DEFAULT_RACE_NAMES_DIR),
                     help="race_names_{date}.csvがあるディレクトリ")
    ap.add_argument("--newspaper-dir", default=str(DEFAULT_NEWSPAPER_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--force", action="store_true",
                     help="出力先に既に{race_id}.jsonが存在するレースも再生成する(既定はスキップ)")
    args = ap.parse_args()

    dates = args.dates.split(",")
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = []
    for date in dates:
        race_names_csv = os.path.join(args.race_names_dir, "race_names_%s.csv" % date)
        df = pd.read_csv(race_names_csv, dtype={"race_id": str})
        target = df[df["surface"].notna()].copy()  # 障害レースを除外
        print("=== %s: 対象レース数 %d / %d (除外: %d) ===" % (
            date, len(target), len(df), len(df) - len(target)))

        for _, row in target.iterrows():
            race_id = row["race_id"]
            race_name = row["race_name"]
            racecourse = row["racecourse"]
            surface = geo.SURFACE_MAP.get(row["surface"], row["surface"])
            distance = float(row["distance_m"])
            straight = geo.is_straight_course(row)

            out_json = os.path.join(args.out_dir, "%s.json" % race_id)
            if os.path.exists(out_json) and not args.force:
                print("[SKIP] %s %s (既存、--forceで上書き)" % (race_id, race_name))
                continue

            np_path = os.path.join(args.newspaper_dir, "%s.csv" % race_id)
            try:
                npdf = pd.read_csv(np_path, dtype=str)
                umaban_list = sorted(set(int(x) for x in npdf["umaban"]))
            except Exception as e:
                all_results.append({"date": date, "race_id": race_id, "race_name": race_name,
                                     "ok": False, "error": "newspaper CSV読込失敗: %s" % e, "elapsed": 0.0})
                continue

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
                                         "ok": False, "error": "未知の競馬場/馬場の組み合わせ: %s/%s" % (racecourse, surface),
                                         "elapsed": 0.0})
                    continue
                cmd += ["--circumference-m", str(circ), "--home-stretch-m", str(home)]

            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            elapsed = time.time() - t0
            ok = proc.returncode == 0
            all_results.append({
                "date": date, "race_id": race_id, "race_name": race_name, "racecourse": racecourse,
                "surface": surface, "distance": distance, "n_horses": len(umaban_list),
                "geometry_note": geometry_note,
                "ok": ok, "elapsed": elapsed,
                "stdout": proc.stdout.strip(), "error": proc.stderr.strip() if not ok else "",
            })
            status = "OK  " if ok else "FAIL"
            print("[%s] %s %s %s (%.1fs) geo=%s" % (status, race_id, race_name, racecourse, elapsed, geometry_note))
            if not ok:
                print("       error: %s" % proc.stderr.strip().replace("\n", " / "))

    ok_list = [r for r in all_results if r["ok"]]
    fail_list = [r for r in all_results if not r["ok"]]
    print("\n=== 全体サマリ: 成功 %d / %d(スキップ除く) ===" % (len(ok_list), len(all_results)))
    if fail_list:
        print("失敗レース:")
        for r in fail_list:
            print("  %s %s %s: %s" % (r["date"], r["race_id"], r["race_name"], r.get("error", "")))
    if all_results:
        total_elapsed = sum(r["elapsed"] for r in all_results)
        print("合計所要時間: %.1f秒" % total_elapsed)


if __name__ == "__main__":
    main()
