# -*- coding: utf-8 -*-
"""実況動画フレーム読み取り(video_timeline.csv、data/video_positions/{race_id}.csv)ベースの
新指標m7。corner_passing_metrics.pyのrace_m6と同じ考え方(footrule_norm)だが、公式の1角〜4角
通過順位ではなく、動画から読み取った任意タイムスタンプでの順位と比較する点が異なる。
sim側の「時刻tでの順位」はstate.log(t, d_rail, ...)を線形補間して求める
(horse_pair_sim.time_at_distance()の逆関数に相当、DT=0.05s刻みのログを補間)。

2026-08-18、scripts/jra_race_sim/へ永続化(旧名stretch_m7_metric.py)。
"""
import bisect
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim_runner_lib import footrule_norm  # noqa: E402


def d_rail_at_time(state, t_query):
    """state.log([(t, d_rail, lane_m, speed, stamina, ground_distance), ...])から、
    時刻t_queryでのd_railを線形補間で返す。ログ範囲外はクランプ(開始前は0、完走後はログの
    末尾=最終d_railで頭打ち、time_at_distance()のフリーズ挙動と対称)。"""
    log = state.log
    times = [row[0] for row in log]
    if t_query <= times[0]:
        return log[0][1]
    if t_query >= times[-1]:
        return log[-1][1]
    i = bisect.bisect_right(times, t_query) - 1
    t0, d0 = log[i][0], log[i][1]
    t1, d1 = log[i + 1][0], log[i + 1][1]
    if t1 == t0:
        return d0
    f = (t_query - t0) / (t1 - t0)
    return d0 + (d1 - d0) * f


def sim_rank_at_time(states, t_query):
    """時刻t_queryでの全頭のd_rail順位(1=先頭)を返す。"""
    d_rail = {s.umaban: d_rail_at_time(s, t_query) for s in states}
    ranked = sorted(d_rail.items(), key=lambda kv: -kv[1])
    return {u: i + 1 for i, (u, _) in enumerate(ranked)}


def race_m7(states, video_df, t_min=None, t_max=None):
    """statesと video_positions/{race_id}.csv 相当のDataFrame
    (列: t_sec,checkpoint,umaban,rank_official)から、動画で読み取った各タイムスタンプでの
    footrule距離を計算する。
    t_min/t_max: 対象とするt_secの範囲(例: kick区間のみに絞りたい場合に指定。dash区間の
    行を除外する等)。Noneなら全行対象。
    戻り値: {"per_t": [{"t_sec":.., "footrule":.., "n_common":..}, ...],
             "mean_footrule": 平均値(Noneの点は除外), "n_t": 有効タイムスタンプ数}
    """
    df = video_df.copy()
    if t_min is not None:
        df = df[df["t_sec"] >= t_min]
    if t_max is not None:
        df = df[df["t_sec"] <= t_max]

    per_t = []
    for t_sec, grp in df.groupby("t_sec"):
        video_rank = {int(r["umaban"]): int(r["rank_official"]) for _, r in grp.iterrows()}
        if len(video_rank) < 2:
            continue
        sim_rank_all = sim_rank_at_time(states, float(t_sec))
        sim_rank = {u: sim_rank_all[u] for u in video_rank if u in sim_rank_all}
        fr = footrule_norm(sim_rank, video_rank)
        if fr is not None:
            per_t.append({"t_sec": float(t_sec), "footrule": fr, "n_common": len(sim_rank)})

    valid = [p["footrule"] for p in per_t]
    mean_footrule = float(pd.Series(valid).mean()) if valid else None
    return {"per_t": per_t, "mean_footrule": mean_footrule, "n_t": len(valid)}


def race_m7_from_csv(states, csv_path, t_min=None, t_max=None):
    """video_positions/{race_id}.csvファイルパスから直接race_m7を計算する簡易ラッパー。"""
    df = pd.read_csv(csv_path, encoding="utf-8")
    return race_m7(states, df, t_min=t_min, t_max=t_max)


if __name__ == "__main__":
    print("video_position_metric.py: import only, use race_m7() from another script", file=sys.stderr)
