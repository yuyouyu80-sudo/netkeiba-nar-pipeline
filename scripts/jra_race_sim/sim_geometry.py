# -*- coding: utf-8 -*-
"""horse_run_all.html の trackGeometry()/pointAtLapT() をPythonに移植した、
相互作用シミュレーション用の唯一の物理定数源。

JS側(SVG描画座標系)とPython側(相互作用シミュレーション、メートル座標系)で
数値がズレると、過去のbuild_artifact.pyと同種の「幾何バグ」を再発するため、
モジュール読み込み時に自己チェックのassertを実行する。
"""
import math

# --- horse_run_all.html の trackGeometry() と同一の定数(SVG単位) ---
VIEW_W, VIEW_H, PAD = 380.0, 220.0, 46.0
HOME_STRETCH_SVG_INPUT = 353.9  # JS側の "homeStretch" 変数(straightFracの算出にのみ使う入力値)
CIRCUMFERENCE_M = 1472.0        # 実コース周長(メートル)


def track_geometry():
    """trackGeometry()のPython移植。戻り値は全てSVG単位(メートルではない)。"""
    straight_frac = min(0.45, max(0.15, HOME_STRETCH_SVG_INPUT / CIRCUMFERENCE_M))
    half_w = (VIEW_W - 2 * PAD) / 2
    ry = (VIEW_H - 2 * PAD) / 2
    rx = ry * 0.85
    straight_half_len = min(half_w - rx - 8, max(24, half_w * (straight_frac / 0.3)))
    perim = 2 * (2 * straight_half_len) + 2 * (math.pi * ry)
    finish_lap_t = (2 * straight_half_len) / perim
    # 単一の直線・単一のコーナーそれぞれが lapT のうち占める割合
    str_frac_single = (2 * straight_half_len) / perim
    turn_frac_single = (math.pi * ry) / perim
    return {
        "ry": ry, "rx": rx, "straight_half_len": straight_half_len,
        "perim": perim, "finish_lap_t": finish_lap_t,
        "str_frac_single": str_frac_single, "turn_frac_single": turn_frac_single,
    }


G = track_geometry()

# --- メートル換算(物理シミュレーションは以降すべてこの単位系で行う) ---
METERS_PER_UNIT = CIRCUMFERENCE_M / G["perim"]
CORNER_R_M = G["ry"] * METERS_PER_UNIT
# 単一直線の実距離(メートル)。JS入力値の353.9mではなく、aspect比によるクランプ後の
# 実ジオメトリから逆算した値を使う(pointAtLapTの実際のセグメント境界と一致させるため。
# straightHalfLenがhalf_w-rx-8でクランプされ、straightFrac由来の値より短くなっている)。
HOME_STRETCH_M = G["str_frac_single"] * CIRCUMFERENCE_M
CORNER_LEN_M = G["turn_frac_single"] * CIRCUMFERENCE_M

# --- 自己チェック: JS側(horse_run_all.html)で実際に使われている数値と一致するか ---
assert abs(G["perim"] - 728.52) < 0.5, f"perim mismatch: {G['perim']}"
assert abs(G["ry"] - 64.0) < 0.01, f"ry mismatch: {G['ry']}"
assert abs(G["straight_half_len"] - 81.6) < 0.1, f"straight_half_len mismatch: {G['straight_half_len']}"
assert abs(METERS_PER_UNIT - 2.021) < 0.01, f"METERS_PER_UNIT mismatch: {METERS_PER_UNIT}"
assert abs(CORNER_R_M - 129.3) < 0.5, f"CORNER_R_M mismatch: {CORNER_R_M}"


def physics_geometry(circumference_m, home_stretch_m):
    """物理シミュレーション専用の厳密なオーバル幾何(クランプ無し)。SVG描画用の
    track_geometry()/Gとは完全に独立しており、こちらは実測の周長・直線長を
    そのまま使う(track_geometry()のようなアスペクト比クランプによる歪みが無い)。
    単純な陸上トラック型のオーバル(直線2本+半円2つ)として厳密に計算する。
    """
    corner_r_m = (circumference_m - 2 * home_stretch_m) / (2 * math.pi)
    turn_frac_single = (math.pi * corner_r_m) / circumference_m
    return {
        "circumference_m": circumference_m,
        "home_stretch_m": home_stretch_m,
        "corner_r_m": corner_r_m,
        "corner_len_m": turn_frac_single * circumference_m,
        "str_frac_single": home_stretch_m / circumference_m,
        "turn_frac_single": turn_frac_single,
    }


def apply_physics_geometry(circumference_m, home_stretch_m):
    """physics_geometry()の結果をこのモジュールの物理定数(CIRCUMFERENCE_M/CORNER_R_M/
    HOME_STRETCH_M/CORNER_LEN_M/METERS_PER_UNIT)へ上書きする。プロセス分離設計により
    1レース=1プロセスで一度だけ呼ぶ想定(horse_pair_sim.py側の関数はこれらをモジュール
    属性として参照するだけなので、呼び出し側のコードは変更不要)。METERS_PER_UNITは
    SVG描画用の固定オーバル形状(G["perim"])に対する実測周長の比率として再計算する
    (SVG側の見た目プロポーションは全レース共通のまま、メートル換算だけレースごとに正しくする)。
    """
    global CIRCUMFERENCE_M, CORNER_R_M, HOME_STRETCH_M, CORNER_LEN_M, METERS_PER_UNIT
    geo = physics_geometry(circumference_m, home_stretch_m)
    CIRCUMFERENCE_M = geo["circumference_m"]
    CORNER_R_M = geo["corner_r_m"]
    HOME_STRETCH_M = geo["home_stretch_m"]
    CORNER_LEN_M = geo["corner_len_m"]
    METERS_PER_UNIT = CIRCUMFERENCE_M / G["perim"]
    return geo


def lap_t_at_distance(d_rail, start_lap_t, laps):
    """horse_run_all.html の lapTAtDistance() と同じ。d_railは0起点の走行距離(m)。"""
    return ((start_lap_t + d_rail / CIRCUMFERENCE_M) % 1 + 1) % 1


def start_lap_t(finish_lap_t, laps):
    return ((finish_lap_t - laps) % 1 + 1) % 1


def is_in_corner(lap_t):
    """pointAtLapTのセグメント分岐と同じ境界で、直線かコーナーかを判定する。

    2026-08-13修正: 従来はここで固定のSVG比率(G["str_frac_single"]=22.4%/
    G["turn_frac_single"]=27.6%、1472m/353.9mという汎用コースの値)を使っていたが、
    apply_physics_geometry()が更新するのはモジュール変数HOME_STRETCH_M/CORNER_LEN_M/
    CIRCUMFERENCE_M側だけで、Gは固定のまま参照され続けていたため、実際のコーナー・
    直線比率(レースごとに16〜30%台でばらつく)が一切反映されないバグだった。
    is_final_stretch()が既にHOME_STRETCH_M(実測値)を使っているのと同じ発想で、
    こちらもレースごとの実測比率を使うよう修正する。"""
    str_frac = HOME_STRETCH_M / CIRCUMFERENCE_M
    turn_frac = CORNER_LEN_M / CIRCUMFERENCE_M
    t = lap_t
    if t < str_frac:
        return False  # 最初の直線(ホームストレッチ)
    t -= str_frac
    if t < turn_frac:
        return True   # 右コーナー
    t -= turn_frac
    if t < str_frac:
        return False  # 向こう正面
    return True        # 左コーナー(最終コーナー)


def is_final_stretch(d_rail, d_total):
    """ホームストレッチ判定は必ず「残り距離」で行う。lapT基準では、周回数が
    端数(このレースは1.223周)の場合にスタート直後で誤爆するため使わない。"""
    return (d_total - d_rail) <= HOME_STRETCH_M


if __name__ == "__main__":
    print("perim(SVG)=%.3f  ry(SVG)=%.3f  straight_half_len(SVG)=%.3f" % (
        G["perim"], G["ry"], G["straight_half_len"]))
    print("METERS_PER_UNIT=%.4f  CORNER_R_M=%.2f  HOME_STRETCH_M=%.2f  CORNER_LEN_M=%.2f" % (
        METERS_PER_UNIT, CORNER_R_M, HOME_STRETCH_M, CORNER_LEN_M))
    print("self-check assertions passed")
