# -*- coding: utf-8 -*-
"""venue_page_template.pyのアニメーションに埋め込むための実測データ(馬ごとのタイム・
着順・上がり3F・通過順位、および先頭馬の実測ラップテーブル)をrace_results/lap_timesの
CSVから抽出する。compare_sim_vs_actual_multi.pyと同じ突き合わせルール(実測ラップの
第1区間は distance % 200 の端数、無ければ200m)を踏襲する。"""
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(r"c:\Users\yuyou\Desktop\新しい作業場所")

_TIME_RE = re.compile(r"^(\d+):(\d+\.\d+)$")
_date_cache = {}


def time_to_seconds(t):
    if t is None or (isinstance(t, float) and pd.isna(t)) or t == "":
        return None
    m = _TIME_RE.match(str(t).strip())
    if not m:
        return None
    return int(m.group(1)) * 60 + float(m.group(2))


def _load_date(date):
    if date not in _date_cache:
        results_df = pd.read_csv(PROJECT_ROOT / "data" / "race_results" / "2026" / f"{date}.csv", dtype=str)
        laps_df = pd.read_csv(PROJECT_ROOT / "data" / "lap_times" / "2026" / f"{date}.csv", dtype=str)
        _date_cache[date] = (results_df, laps_df)
    return _date_cache[date]


def get_actual_for_race(race_id, date, distance_m):
    """戻り値: {"horses": {umaban(int): {...}}, "leaderLapTable": [...] or None}"""
    results_df, laps_df = _load_date(date)
    real = results_df[results_df["race_id"] == race_id]

    horses = {}
    for _, r in real.iterrows():
        try:
            umaban = int(r["umaban"])
        except (TypeError, ValueError):
            continue
        time_sec = time_to_seconds(r.get("time"))
        last3f_raw = str(r.get("last_3f", "") or "").strip()
        last3f = float(last3f_raw) if last3f_raw not in ("", "nan") else None
        finish_pos_raw = str(r.get("finish_pos", "") or "").strip()
        passing_order = str(r.get("passing_order", "") or "").strip()
        horses[umaban] = {
            "actualFinishPos": finish_pos_raw or None,
            "actualTime": str(r.get("time") or "").strip() or None,
            "actualTimeSec": time_sec,
            "actualLast3f": last3f,
            "actualPassingOrder": passing_order or None,
        }

    real_laps = laps_df[laps_df["race_id"] == race_id].copy()
    leader_lap_table = None
    if len(real_laps) > 0:
        real_laps["lap_time_sec"] = real_laps["lap_time_sec"].astype(float)
        real_laps["segment_index"] = real_laps["segment_index"].astype(int)
        real_laps = real_laps.sort_values("segment_index")
        splits = real_laps["lap_time_sec"].to_numpy()
        n_seg = len(splits)
        first_len = float(distance_m) % 200.0
        if first_len == 0:
            first_len = 200.0
        seg_lens = [first_len] + [200.0] * (n_seg - 1)
        marks = []
        acc = 0.0
        for length in seg_lens:
            acc += length
            marks.append(acc)
        cumulative = 0.0
        rows = []
        for d_mark, split in zip(marks, splits):
            cumulative += float(split)
            rows.append({"distance": round(d_mark), "cumulative": round(cumulative, 2), "split": round(float(split), 2)})
        leader_lap_table = rows

    return {"horses": horses, "leaderLapTable": leader_lap_table}
