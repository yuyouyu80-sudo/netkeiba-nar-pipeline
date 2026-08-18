# -*- coding: utf-8 -*-
"""1レース分の相互作用シミュレーションを1プロセス内で完結させるCLIワーカー。

「1レース=1プロセス」というプロセス分離を設計の柱にすることで、horse_baseline.py の
_regression_cache・_TIME_AT_V1(SEG1_LEN依存の定数)・CSV_PATH や、sim_geometry.py の
CIRCUMFERENCE_M/CORNER_R_M/HOME_STRETCH_M/METERS_PER_UNIT といったモジュールレベルの
状態が複数レース間で汚染される問題を、コードを変更せず構造的に回避する
(setattrで1回だけレース固有の値に上書きしてから使う、という現行コードに近い形を維持)。
"""
import argparse
import sys

import race_potential
import horse_baseline as hb
import sim_geometry as sg
import horse_pair_sim as hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race-id", required=True)
    ap.add_argument("--race-name", default="")
    ap.add_argument("--racecourse", required=True)
    ap.add_argument("--surface", required=True, choices=["芝", "ダート"])
    ap.add_argument("--distance", type=float, required=True)
    ap.add_argument("--is-straight-course", action="store_true")
    ap.add_argument("--circumference-m", type=float, default=None)
    ap.add_argument("--home-stretch-m", type=float, default=None)
    ap.add_argument("--umaban-list", required=True, help="カンマ区切り(例: 1,2,3,...)")
    ap.add_argument("--newspaper-csv", default=None, help="省略時は data/newspaper/{race_id}.csv")
    ap.add_argument("--potential-csv", default=None, help="省略時は _workdir内に自動生成")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    if not args.is_straight_course and (args.circumference_m is None or args.home_stretch_m is None):
        ap.error("--is-straight-course を指定しない場合は --circumference-m と --home-stretch-m が必須です")

    newspaper_csv = args.newspaper_csv or (
        r"C:\Users\yuyou\Desktop\新しい作業場所\data\newspaper\%s.csv" % args.race_id
    )
    potential_csv = args.potential_csv or (
        r"C:\Users\yuyou\Desktop\新しい作業場所\scripts\jra_race_sim\_workdir"
        r"\race_potential_%s.csv" % args.race_id
    )

    # 1. このレース専用のポテンシャルCSVを生成
    race_potential.build_potential_csv(
        race_id=args.race_id, race_distance=args.distance, surface=args.surface,
        racecourse=args.racecourse, newspaper_csv_path=newspaper_csv, out_csv_path=potential_csv,
    )

    # 2. horse_baseline のモジュール状態をこのレース用に上書き(プロセス分離で汚染は起きない)
    hb.CSV_PATH = potential_csv
    hb.D_TOTAL = args.distance
    hb.SEG1_LEN = max(0.0, args.distance - 600.0)  # 上がり3F=600mは距離によらず固定
    hb._regression_cache = None  # 新しいCSV_PATHに対応する回帰係数を再計算させる
    hb.SURFACE = args.surface  # have=0フォールバック定数・キック最高速度の芝/ダート切替用
    hb.IS_STRAIGHT_COURSE = bool(args.is_straight_course)  # kick_start_distance()の直線コース分岐用

    # 3. ジオメトリをこのレース用に上書き(kick_start_distance()がHOME_STRETCH_M/CORNER_LEN_Mを
    #    参照するため、hb._init_kick_geometry()より必ず先に行う)
    if args.is_straight_course:
        # 直線コースはcircumference_mが物理的に存在しないため physics_geometry() は使わない。
        # HOME_STRETCH_M を距離以上にして、レース全体を「ホームストレッチ挙動」扱いにする
        # (horse_pair_sim.IS_STRAIGHT_COURSE=True と組み合わせて、コーナー関連の計算を
        # 一切スキップさせる)。
        sg.HOME_STRETCH_M = args.distance + 1.0
        extrap_note = "直線コース(コーナー計算なし)"
    else:
        geo = sg.apply_physics_geometry(args.circumference_m, args.home_stretch_m)
        extrap_note = "CORNER_R_M=%.1fm HOME_STRETCH_M=%.1fm" % (geo["corner_r_m"], geo["home_stretch_m"])

    hb._init_kick_geometry()  # kick_start_d・巡航区間所要時間(v_cruise=1相当)をこのレース用に再計算

    # 4. horse_pair_sim のモジュール状態をこのレース用に上書き
    hp.PAIR = tuple(int(x) for x in args.umaban_list.split(","))
    hp.IS_STRAIGHT_COURSE = bool(args.is_straight_course)

    # 5. シミュレーション実行
    states = hp.simulate()

    # 6. 検証アサーション(失敗したら非ゼロ終了・標準エラーへ詳細出力)
    problems = []
    for s in states:
        d_rails = [p[1] for p in s.log]
        lanes = [p[2] for p in s.log]
        monotone = all(d_rails[i] <= d_rails[i + 1] + 1e-6 for i in range(len(d_rails) - 1))
        in_bounds = all(hp.LANE_MIN_M - 1e-6 <= l <= hp.TRACK_WIDTH_M + 1e-6 for l in lanes)
        if not monotone:
            problems.append("umaban=%d d_rail not monotone" % s.umaban)
        if not in_bounds:
            problems.append("umaban=%d lane out of bounds" % s.umaban)
    if problems:
        print("ASSERTION FAILED race_id=%s: %s" % (args.race_id, "; ".join(problems)), file=sys.stderr)
        sys.exit(1)

    # 7. JSON書き出し(every=6 => dt=0.05s*6=0.3s間隔、ファイルサイズ抑制のため0.1s間隔より粗くする)
    hp.export_json(states, args.out_json, every=6)

    # 8. 実測ビン(300-1100m)を超える外挿量をログ(2026-08-09〜: shape_seg1はkick_start_dまで
    #    しか使わないため、SEG1_LENではなくkick_start_d基準で判定する)
    extrap_m = max(0.0, hb._KICK_START_D - 1100.0)
    flag = " [注意:大きい]" if extrap_m > 400.0 else ""
    print("OK race_id=%s race_name=%s racecourse=%s surface=%s distance=%.0fm n_horses=%d "
          "SEG1_LEN=%.0fm kick_start_d=%.0fm extrap=%.0fm%s geometry=[%s]" % (
              args.race_id, args.race_name, args.racecourse, args.surface, args.distance,
              len(states), hb.SEG1_LEN, hb._KICK_START_D, extrap_m, flag, extrap_note))


if __name__ == "__main__":
    main()
