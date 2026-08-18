# -*- coding: utf-8 -*-
"""simulate_one_race.py の処理本体を、サブプロセスを起動せずインポートして直接呼べる
関数として切り出したライブラリ。5パラメータの探索(calibrate_pass_params.py)は
1回の評価で数十レースを繰り返し実行するため、プロセス起動オーバーヘッドを避けたい。

simulate_one_race.py が「1レース=1プロセス」で構造的に回避しているモジュール
グローバルの汚染は、run_race() の冒頭で毎回明示的に全グローバルを上書きすることで
同様に回避する(汚染源は heuristic ではなく network of globals: horse_baseline.py の
CSV_PATH/D_TOTAL/SEG1_LEN/IS_STRAIGHT_COURSE/_regression_cache/SURFACE/
_KICK_START_D・_K_CRUISE_BY_STYLE(_init_kick_geometry()経由)、sim_geometry.py の
CIRCUMFERENCE_M/CORNER_R_M/HOME_STRETCH_M/CORNER_LEN_M/METERS_PER_UNIT、
horse_pair_sim.py の PAIR/IS_STRAIGHT_COURSE + 探索対象の5パラメータ、の
計14個。simulate_one_race.pyと1件ずつ突き合わせて漏れがないことを確認済み)。
"""
import re

import numpy as np
import pandas as pd

import race_potential
import horse_baseline as hb
import sim_geometry as sg
import horse_pair_sim as hp

SCRATCH = r"C:\Users\yuyou\Desktop\新しい作業場所\scripts\jra_race_sim\_workdir"

PASS_PARAM_NAMES = [
    "GAP_LONGITUDINAL_INSIDE_CUT_M", "GAP_LONGITUDINAL_M", "LATERAL_OVERLAP_M",
    "LATERAL_GAP_FOR_AVOID_M", "GAP_LONGITUDINAL_STRETCH_CLEAR_M",
]

# 現行値(horse_pair_sim.py記載のオリジナル値、探索のベースライン兼較正対象外パラメータの固定値)
BASELINE_PARAMS = {
    "GAP_LONGITUDINAL_INSIDE_CUT_M": 2.0,
    "GAP_LONGITUDINAL_M": 5.0,
    "LATERAL_OVERLAP_M": 2.5,
    "LATERAL_GAP_FOR_AVOID_M": 5.0,
    "GAP_LONGITUDINAL_STRETCH_CLEAR_M": 1.0,
}

# run_race()呼び出し側がk1/k2を指定しない場合のフォールバック。
# 2026-08-13エンジニアレビュー指摘: 以前はここに-0.35/-10.0をリテラルで複製していたが、
# horse_baseline.py側だけ更新してこちらの更新を忘れると次回較正がサイレントに古い値へ
# 固定される「単一の真実源」違反だった。import直後のhb.RUNNING_STYLE_K1/K2(=hb.py側の
# 現在のモジュールデフォルト)をそのままキャプチャする形にし、複製をやめる。
_DEFAULT_RUNNING_STYLE_K1 = hb.RUNNING_STYLE_K1
_DEFAULT_RUNNING_STYLE_K2 = hb.RUNNING_STYLE_K2

# 2026-08-15追加: ラストスパート/スタートダッシュ系5パラメータの較正用(実況動画フレーム
# 読み取りベースのm7較正)。上と同じ「import直後にhb側のデフォルトをキャプチャし複製しない」
# パターンを踏襲する(2026-08-13エンジニアレビュー指摘のsingle-source-of-truth方針)。
_DEFAULT_KICK_START_MIN_M = hb.KICK_START_MIN_M
_DEFAULT_R_MIN = hb.R_MIN
_DEFAULT_R_MAX = hb.R_MAX
_DEFAULT_KICK_EFFORT_EXPONENT = hb.KICK_EFFORT_EXPONENT
_DEFAULT_DASH_PEAK_DIST_M = hb.DASH_PEAK_DIST_M


def run_race(race_id, race_name, racecourse, surface, distance, umaban_list,
             is_straight, circumference_m=None, home_stretch_m=None, pass_params=None,
             running_style_k1=None, running_style_k2=None,
             kick_start_min_m=None, r_min=None, r_max=None,
             kick_effort_exponent=None, dash_peak_dist_m=None):
    """simulate_one_race.py のmain()相当。JSONは書かず states を直接返す。
    running_style_k1/k2: 指定時はhb.RUNNING_STYLE_K1/K2を上書きする
    (2026-08-13、m6較正での軽量グリッド評価用に追加。Noneならhorse_baseline.pyの
    現行デフォルト値のまま)。
    kick_start_min_m/r_min/r_max/kick_effort_exponent/dash_peak_dist_m: 2026-08-15追加、
    ラストスパート/スタートダッシュ系パラメータの較正用(m7較正)。指定時はhb側の同名定数を
    上書きする。dash_peak_dist_mを指定した場合はhb.MULTIPLIER_ONSET_DIST_M(=DASH_PEAK_DIST_M
    を参照するモジュールロード時定数、自動追随しない)も明示的に追随させる。いずれもNoneなら
    horse_baseline.pyの現行デフォルト値のまま。"""
    newspaper_csv = r"C:\Users\yuyou\Desktop\新しい作業場所\data\newspaper\%s.csv" % race_id
    potential_csv = SCRATCH + r"\_search_potential_%s.csv" % race_id

    race_potential.build_potential_csv(
        race_id=race_id, race_distance=distance, surface=surface,
        racecourse=racecourse, newspaper_csv_path=newspaper_csv, out_csv_path=potential_csv,
    )

    hb.CSV_PATH = potential_csv
    hb.D_TOTAL = distance
    hb.SEG1_LEN = max(0.0, distance - 600.0)
    hb._regression_cache = None
    hb.SURFACE = surface
    hb.IS_STRAIGHT_COURSE = bool(is_straight)

    if is_straight:
        sg.HOME_STRETCH_M = distance + 1.0
    else:
        sg.apply_physics_geometry(circumference_m, home_stretch_m)

    # モジュールグローバルの汚染防止(このファイル冒頭のdocstring方針と同じ): Noneでも
    # 必ず明示的に既定値へ戻す。呼び出し側ループで一部だけk1/k2を指定した場合に、
    # 前回呼び出しの値が意図せず残らないようにする。
    hb.RUNNING_STYLE_K1 = running_style_k1 if running_style_k1 is not None else _DEFAULT_RUNNING_STYLE_K1
    hb.RUNNING_STYLE_K2 = running_style_k2 if running_style_k2 is not None else _DEFAULT_RUNNING_STYLE_K2
    # 2026-08-14追加: モンテカルロ・アンサンブル(monte_carlo_ensemble.py)がSTAMINA_OFFSET_OVERRIDEを
    # 設定した場合に前回呼び出しの値が残らないよう、通常呼び出し(このrun_race())では必ず空にする
    # (このファイル冒頭のグローバル汚染防止方針と同じパターン)。
    hb.STAMINA_OFFSET_OVERRIDE = {}

    # 2026-08-15追加: ラストスパート/スタートダッシュ系5パラメータの上書き(未指定分は明示的に
    # デフォルトへ戻す、既存のk1/k2と同じ汚染防止パターン)。kick_start_distance()が
    # KICK_START_MIN_Mを参照するため、必ずhb._init_kick_geometry()より前に設定する。
    hb.KICK_START_MIN_M = kick_start_min_m if kick_start_min_m is not None else _DEFAULT_KICK_START_MIN_M
    hb.R_MIN = r_min if r_min is not None else _DEFAULT_R_MIN
    hb.R_MAX = r_max if r_max is not None else _DEFAULT_R_MAX
    hb.KICK_EFFORT_EXPONENT = (kick_effort_exponent if kick_effort_exponent is not None
                                else _DEFAULT_KICK_EFFORT_EXPONENT)
    hb.DASH_PEAK_DIST_M = dash_peak_dist_m if dash_peak_dist_m is not None else _DEFAULT_DASH_PEAK_DIST_M
    # MULTIPLIER_ONSET_DIST_M = DASH_PEAK_DIST_M はhb.py内でモジュールロード時に一度だけ
    # 評価される定数であり、DASH_PEAK_DIST_Mを書き換えても自動追随しない。明示的に追随させる
    # (このプランのエンジニアレビューで指摘された派生グローバル)。
    hb.MULTIPLIER_ONSET_DIST_M = hb.DASH_PEAK_DIST_M

    hb._init_kick_geometry()  # ジオメトリ設定後に呼ぶ(kick_start_distance()が参照するため)

    hp.PAIR = tuple(int(x) for x in umaban_list)
    hp.IS_STRAIGHT_COURSE = bool(is_straight)

    params = dict(BASELINE_PARAMS)
    if pass_params:
        params.update(pass_params)
    for name, val in params.items():
        setattr(hp, name, val)

    return hp.simulate()


_TIME_RE = re.compile(r"^(\d+):(\d+\.\d+)$")


def time_to_seconds(t):
    if t is None or (isinstance(t, float) and pd.isna(t)) or t == "":
        return None
    m = _TIME_RE.match(str(t).strip())
    if not m:
        return None
    return int(m.group(1)) * 60 + float(m.group(2))


def to_rank(values: dict) -> dict:
    items = sorted(values.items(), key=lambda kv: kv[1])
    return {k: i + 1 for i, (k, _) in enumerate(items)}


def footrule_norm(rank_a: dict, rank_b: dict):
    common = sorted(set(rank_a) & set(rank_b))
    n = len(common)
    if n < 2:
        return None
    raw = sum(abs(rank_a[k] - rank_b[k]) for k in common)
    max_possible = (n * n) // 2
    return raw / max_possible if max_possible > 0 else None


def race_metrics(states, real_df, home_stretch_m, distance_m, is_straight):
    """1レース分のm1(footrule_norm)・m2(time MAE)・m5(stretch footrule_norm)を返す。
    real_df: そのrace_idの実測行(finish_pos_num列必須、umaban列はint化済み想定)。"""
    real = real_df.copy()
    real["finish_pos_num"] = pd.to_numeric(real["finish_pos"], errors="coerce")
    finished = real[real["finish_pos_num"].notna()].copy()
    finished_by_umaban = {int(r["umaban"]): r for _, r in finished.iterrows()}
    real_finish_matched_all = {int(u): int(finished_by_umaban[int(u)]["finish_pos_num"]) for u in finished_by_umaban}

    sim_time = {}
    sim_stretch_time = {}
    d_stretch_entry = (distance_m - home_stretch_m) if (home_stretch_m is not None and not is_straight) else None
    for s in states:
        ft = s.finish_time if s.finish_time is not None else s.log[-1][0]
        sim_time[s.umaban] = ft
        if d_stretch_entry is not None:
            sim_stretch_time[s.umaban] = hp.time_at_distance(s, max(0.0, d_stretch_entry))

    matched = sorted(set(sim_time) & set(real_finish_matched_all))
    if len(matched) < 2:
        return None

    # m1: 着順footrule
    sim_rank1 = to_rank({u: sim_time[u] for u in matched})
    real_rank1 = to_rank({u: real_finish_matched_all[u] for u in matched})
    m1 = footrule_norm(sim_rank1, real_rank1)

    # m2: 走破タイムMAE(符号付き誤差の絶対値平均)
    m2_errs = []
    for u in matched:
        r = finished_by_umaban[u]
        real_time_s = time_to_seconds(r["time"])
        if real_time_s is not None:
            m2_errs.append(abs(sim_time[u] - real_time_s))
    m2_mae = float(np.mean(m2_errs)) if m2_errs else None

    # m5: 直線入り順位footrule(参考指標)
    m5 = None
    if d_stretch_entry is not None and d_stretch_entry > 0:
        real_stretch = {}
        for u in matched:
            po = str(finished_by_umaban[u].get("passing_order", "") or "")
            if po.strip():
                try:
                    real_stretch[u] = int(po.split("-")[-1])
                except ValueError:
                    pass
        common5 = sorted(set(sim_stretch_time) & set(real_stretch))
        if len(common5) >= 2:
            sim_rank5 = to_rank({u: sim_stretch_time[u] for u in common5})
            real_rank5 = to_rank({u: real_stretch[u] for u in common5})
            m5 = footrule_norm(sim_rank5, real_rank5)

    return {"m1_footrule_norm": m1, "m2_mae": m2_mae, "m5_footrule_norm": m5, "n_matched": len(matched)}
