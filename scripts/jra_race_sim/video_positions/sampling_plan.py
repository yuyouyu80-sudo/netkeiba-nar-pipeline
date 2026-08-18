# -*- coding: utf-8 -*-
"""動画フレームからの位置読み取り対象タイムスタンプ(kick_ts/dash_ts)を、レースメタ情報から
計算する。2026-08-15〜18のラストスパート較正タスクで対話内のみで実行されていたロジック
(旧`step0_kick_window.py`+ステップ1)を、17レース分の既存`step1_sampling_plan.json`から
逆算・再実装したもの(元の`.py`は保存されておらず現存しないため、これは移植ではなく再実装。
`--verify-against`オプションで既存出力との一致を検証できる)。

計算方法:
- `sim_runner_lib.run_race()`でそのレースを1回シミュレーションし、`horse_baseline._KICK_START_D`
  (キック開始距離)を読み取る(run_race()内部で`hb._init_kick_geometry()`が必ず呼ばれるため、
  呼び出し後にモジュール属性として取得できる)。
- 全頭中「最も早くキック開始距離に到達した馬」の到達時刻を`t_kick_earliest`とする。
- 全頭中「最も遅い完走時刻」を`t_goal`とする。
- `kick_ts`: `floor(t_kick_earliest) - 4`を起点に3秒間隔で、`t_goal`以上になる最初の値まで。
- `dash_ts`: `[5, 10, 15, 20, 25]`のうち`kick_ts[0]`未満のものだけを残す
  (キック区間サンプルと重複しないようにする)。
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import horse_baseline as hb  # noqa: E402
import sim_runner_lib as srl  # noqa: E402

DASH_CANDIDATES = [5, 10, 15, 20, 25]
KICK_STEP_SEC = 3
START_BUFFER_SEC = 4


def _time_at_distance_inv(state, d_target):
    """state.log([(t, d_rail, ...), ...])から、d_railが最初にd_targetへ達する時刻を線形補間で
    求める(video_position_metric.d_rail_at_time()の逆方向、こちらは距離→時刻)。"""
    log = state.log
    for i in range(1, len(log)):
        d0, d1 = log[i - 1][1], log[i][1]
        if d1 >= d_target:
            t0, t1 = log[i - 1][0], log[i][0]
            if d1 == d0:
                return t0
            f = (d_target - d0) / (d1 - d0)
            return t0 + (t1 - t0) * f
    return log[-1][0]


def compute_sampling_plan(race_id, race_name, racecourse, surface, distance, umaban_list,
                           is_straight, circumference_m=None, home_stretch_m=None):
    """1レース分のkick_ts/dash_tsを計算する。戻り値は
    {"race_id","name","distance","t_kick_earliest","t_goal","kick_start_d","kick_ts","dash_ts"}。"""
    states = srl.run_race(
        race_id=race_id, race_name=race_name, racecourse=racecourse, surface=surface,
        distance=distance, umaban_list=umaban_list, is_straight=is_straight,
        circumference_m=circumference_m, home_stretch_m=home_stretch_m,
    )
    kick_start_d = hb._KICK_START_D

    t_kick_earliest = min(_time_at_distance_inv(s, kick_start_d) for s in states)
    t_goal = max((s.finish_time if s.finish_time is not None else s.log[-1][0]) for s in states)

    kick_start = math.floor(t_kick_earliest) - START_BUFFER_SEC
    kick_ts = [kick_start]
    while kick_ts[-1] < t_goal:
        kick_ts.append(kick_ts[-1] + KICK_STEP_SEC)

    dash_ts = [t for t in DASH_CANDIDATES if t < kick_ts[0]]

    return {
        "race_id": race_id, "name": race_name, "distance": distance,
        "t_kick_earliest": round(t_kick_earliest, 1), "t_goal": round(t_goal, 1),
        "kick_start_d": round(kick_start_d, 1),
        "kick_ts": kick_ts, "dash_ts": dash_ts, "n_kick_samples": len(kick_ts),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--races-json", required=True,
                     help="レース一覧JSON(配列)。各要素にrace_id/race_name/racecourse/surface/"
                          "distance/umaban_list/is_straight/circumference_m/home_stretch_mを含む")
    ap.add_argument("--out", required=True, help="出力先JSON(race_id -> sampling plan の辞書)")
    ap.add_argument("--verify-against", default=None,
                     help="既存のsampling plan JSONと比較し、不一致があれば報告する(回帰検証用)")
    args = ap.parse_args()

    races = json.loads(Path(args.races_json).read_text(encoding="utf-8"))
    result = {}
    for r in races:
        plan = compute_sampling_plan(
            race_id=r["race_id"], race_name=r.get("race_name", r["race_id"]),
            racecourse=r["racecourse"], surface=r["surface"], distance=r["distance"],
            umaban_list=r["umaban_list"], is_straight=r["is_straight"],
            circumference_m=r.get("circumference_m"), home_stretch_m=r.get("home_stretch_m"),
        )
        result[r["race_id"]] = plan
        print(f"{r['race_id']}: kick_ts={len(plan['kick_ts'])}点({plan['kick_ts'][0]}-{plan['kick_ts'][-1]}) "
              f"dash_ts={plan['dash_ts']}")

    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"書き出し: {args.out}")

    if args.verify_against:
        ref = json.loads(Path(args.verify_against).read_text(encoding="utf-8"))
        mismatches = []
        for rid, plan in result.items():
            ref_plan = ref.get(rid)
            if ref_plan is None:
                continue
            if plan["kick_ts"] != ref_plan["kick_ts"] or plan["dash_ts"] != ref_plan["dash_ts"]:
                mismatches.append(rid)
        if mismatches:
            print(f"不一致: {mismatches}")
            sys.exit(1)
        print(f"検証OK: {len(result)}レース中、比較対象全件がkick_ts/dash_tsで一致")


if __name__ == "__main__":
    main()
