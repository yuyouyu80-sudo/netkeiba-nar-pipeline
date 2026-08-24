# -*- coding: utf-8 -*-
"""NAR「5位以下」予測モデル用の新規候補シグナル(サイドカー)。

既存 nar_signals.py は無改造(import参照のみ)。既存24本(死にシグナル5本除く)は
そのまま `nar_signals.build_signals()` の出力を「向き反転」で流用するので、ここでは
新規候補シグナルだけを追加実装する。

符号規約: 全シグナルは「高いほど4着以内で終わりやすい(good)」に統一する
(既存24本の規約と合わせることで、下位予測側では argsort の向きを1箇所反転するだけで済む)。

新規候補(Track Bのみ、newspaperのdata_*系スキーマが2026-07-25以降にしか存在しないため):
  class_hike        : 前走からのクラス昇格幅(既存class_dropの符号反転版、格上げ幅が
                       大きいほど低スコア)。全期間で利用可能(past1_race_nameのみ使用)なので
                       Track A側でも使える。
  distance_breadth  : 今回距離以外の距離帯(data_distance_slot2-4、slot5は「全成績」への
                       重複フォールバックのため除外)での実績ブレンド。
  course_breadth    : 別回り(data_course_slot2-3、右回り/左回りの集計。slot4は
                       distance_slot5と同じ「全成績」重複列のため除外)での実績ブレンド。
  overall_rate      : 生涯win/place3/return率。data_cushion_slot1から取るが、実体は
                       クッション値別集計ではなく「全成績」への構造的フォールバックと実測で
                       判明した値(2026-08-24確認)。素直に生涯レートとして扱う。
  hold_l3f          : 持続タイムベンチマーク走(holdtime_just)の上がり3F。小さいほど良い。
  hold_jyuni        : 持続タイムベンチマーク走の着順。小さいほど良い。
  hold_babasa       : 持続タイムベンチマーク走の馬場差。符号方向が自明でないため
                       HOLD_BABASA_SIGN で明示的に指定する(Phase 0診断で実測してから確定)。
  (hold_kyakusitu は脚質カテゴリで単調順序がなく今回は見送り、data_others_slot3=馬体重帯は
   実測で全行NULLの死にシグナルと判明したため不採用)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LIB_DIR))
import nar_signals as NS  # noqa: E402  既存モジュール、無改造・関数を読むだけ

CLASS_HIKE_SCALE = 3.0  # class_dropと同じ「3クラス分で満点」スケール

# Phase 0診断(nar_search_bottom_2026_08_24.py内)で実測した符号方向をここに反映する。
# 既定値は「馬場差が大きい(遅い馬場だった)ほど不利」という直感に基づく暫定値であり、
# Phase 0の相関実測結果と食い違う場合はこの値を実測に合わせて更新してから本探索に進む。
HOLD_BABASA_SIGN = -1.0

SHRINK_SPECS_BOTTOM = {
    "distance_breadth_slot2_win": ("data_distance_slot2_win_rate", "data_distance_slot2_runs"),
    "distance_breadth_slot2_place3": ("data_distance_slot2_place3_rate", "data_distance_slot2_runs"),
    "distance_breadth_slot2_return": ("data_distance_slot2_win_return_rate", "data_distance_slot2_runs"),
    "distance_breadth_slot3_win": ("data_distance_slot3_win_rate", "data_distance_slot3_runs"),
    "distance_breadth_slot3_place3": ("data_distance_slot3_place3_rate", "data_distance_slot3_runs"),
    "distance_breadth_slot3_return": ("data_distance_slot3_win_return_rate", "data_distance_slot3_runs"),
    "distance_breadth_slot4_win": ("data_distance_slot4_win_rate", "data_distance_slot4_runs"),
    "distance_breadth_slot4_place3": ("data_distance_slot4_place3_rate", "data_distance_slot4_runs"),
    "distance_breadth_slot4_return": ("data_distance_slot4_win_return_rate", "data_distance_slot4_runs"),
    "course_breadth_slot2_win": ("data_course_slot2_win_rate", "data_course_slot2_runs"),
    "course_breadth_slot2_place3": ("data_course_slot2_place3_rate", "data_course_slot2_runs"),
    "course_breadth_slot2_return": ("data_course_slot2_win_return_rate", "data_course_slot2_runs"),
    "course_breadth_slot3_win": ("data_course_slot3_win_rate", "data_course_slot3_runs"),
    "course_breadth_slot3_place3": ("data_course_slot3_place3_rate", "data_course_slot3_runs"),
    "course_breadth_slot3_return": ("data_course_slot3_win_return_rate", "data_course_slot3_runs"),
    "overall_win": ("data_cushion_slot1_win_rate", "data_cushion_slot1_runs"),
    "overall_place3": ("data_cushion_slot1_place3_rate", "data_cushion_slot1_runs"),
    "overall_return": ("data_cushion_slot1_win_return_rate", "data_cushion_slot1_runs"),
}

NEW_SIGNALS_TRACK_A = ["class_hike"]  # 全期間で利用可能
NEW_SIGNALS_TRACK_B_ONLY = [
    "distance_breadth", "course_breadth", "overall_rate",
    "hold_l3f", "hold_jyuni", "hold_babasa",
]
NEW_SIGNALS_ALL = NEW_SIGNALS_TRACK_A + NEW_SIGNALS_TRACK_B_ONLY


def _shrink_local(df: pd.DataFrame, key: str, priors: dict, specs: dict) -> pd.Series:
    """nar_signals._shrink の複製版。SHRINK_SPECS(既存モジュール直書き)ではなく、
    引数で渡されたspecsを見る点だけが違う(既存ファイルを無改造にするための最小限の複製)。"""
    rate_col, runs_col = specs[key]
    rate = NS._pct(NS._col(df, rate_col))
    runs = NS._num(NS._col(df, runs_col)).fillna(0.0)
    prior = priors.get(key, np.nan)
    if pd.isna(prior):
        return pd.Series(np.nan, index=df.index)
    return (rate.fillna(prior) * runs + prior * NS.SHRINK_K) / (runs + NS.SHRINK_K)


def make_priors_bottom(entries: list, specs: dict = SHRINK_SPECS_BOTTOM) -> dict:
    """nar_signals.make_priors の複製版(specsを引数化)。fold内学習側だけで呼ぶこと。"""
    priors = {}
    for key, (rate_col, _runs) in specs.items():
        vals = pd.concat([NS._pct(NS._col(e["df"], rate_col)) for e in entries], ignore_index=True)
        priors[key] = float(vals.mean(skipna=True))
    return priors


def build_bottom_signals(df: pd.DataFrame, current_class: float, priors_bottom: dict,
                         track_b: bool) -> dict:
    """1レース分の新規候補シグナル辞書を返す(0..1正規化、高いほどgood)。
    track_b=False の場合は Track B専用シグナル(distance_breadth等)を計算しない
    (列自体が存在しないため計算しても全NaNになるだけ、無駄な計算を避ける)。"""
    sig = {}

    # --- class_hike: 前走からのクラス昇格幅(class_dropの符号反転版、正しい極性は
    # 「1 - 昇格幅/SCALE」。class_dropの値をそのまま符号反転すると[-1,0]域になり
    # combine()が前提とする0..1規約を壊すため、この形にする)。
    past1_class = NS._col(df, "past1_race_name").map(NS.class_ordinal)
    past1_class = past1_class.fillna(NS._col(df, "past1_race_class").map(NS.class_ordinal))
    hike = (current_class - past1_class).clip(lower=0.0)
    sig["class_hike"] = 1.0 - (hike / CLASS_HIKE_SCALE).clip(upper=1.0)

    if not track_b:
        for name in NEW_SIGNALS_TRACK_B_ONLY:
            sig[name] = pd.Series(np.nan, index=df.index)
        return sig

    # --- distance_breadth: 今回距離以外(slot2-4)の実績ブレンド
    sig["distance_breadth"] = NS._blend_minmax(
        _shrink_local(df, "distance_breadth_slot2_win", priors_bottom, SHRINK_SPECS_BOTTOM),
        _shrink_local(df, "distance_breadth_slot2_place3", priors_bottom, SHRINK_SPECS_BOTTOM),
        _shrink_local(df, "distance_breadth_slot2_return", priors_bottom, SHRINK_SPECS_BOTTOM),
        _shrink_local(df, "distance_breadth_slot3_win", priors_bottom, SHRINK_SPECS_BOTTOM),
        _shrink_local(df, "distance_breadth_slot3_place3", priors_bottom, SHRINK_SPECS_BOTTOM),
        _shrink_local(df, "distance_breadth_slot3_return", priors_bottom, SHRINK_SPECS_BOTTOM),
        _shrink_local(df, "distance_breadth_slot4_win", priors_bottom, SHRINK_SPECS_BOTTOM),
        _shrink_local(df, "distance_breadth_slot4_place3", priors_bottom, SHRINK_SPECS_BOTTOM),
        _shrink_local(df, "distance_breadth_slot4_return", priors_bottom, SHRINK_SPECS_BOTTOM),
    )

    # --- course_breadth: 右回り/左回り集計(slot2-3)の実績ブレンド
    sig["course_breadth"] = NS._blend_minmax(
        _shrink_local(df, "course_breadth_slot2_win", priors_bottom, SHRINK_SPECS_BOTTOM),
        _shrink_local(df, "course_breadth_slot2_place3", priors_bottom, SHRINK_SPECS_BOTTOM),
        _shrink_local(df, "course_breadth_slot2_return", priors_bottom, SHRINK_SPECS_BOTTOM),
        _shrink_local(df, "course_breadth_slot3_win", priors_bottom, SHRINK_SPECS_BOTTOM),
        _shrink_local(df, "course_breadth_slot3_place3", priors_bottom, SHRINK_SPECS_BOTTOM),
        _shrink_local(df, "course_breadth_slot3_return", priors_bottom, SHRINK_SPECS_BOTTOM),
    )

    # --- overall_rate: 生涯win/place3/return率(実体は「全成績」フォールバック列)
    sig["overall_rate"] = NS._blend_minmax(
        _shrink_local(df, "overall_win", priors_bottom, SHRINK_SPECS_BOTTOM),
        _shrink_local(df, "overall_place3", priors_bottom, SHRINK_SPECS_BOTTOM),
        _shrink_local(df, "overall_return", priors_bottom, SHRINK_SPECS_BOTTOM),
    )

    # --- hold_l3f: 持続タイムベンチマーク走の上がり3F(小さいほど良い)
    l3f = NS._num(NS._col(df, "holdtime_just_l3f"))
    sig["hold_l3f"] = NS._minmax(-l3f)

    # --- hold_jyuni: 持続タイムベンチマーク走の着順(小さいほど良い、DNF考慮)
    jyuni = NS._finish_with_dnf_penalty(NS._col(df, "holdtime_just_jyuni"))
    sig["hold_jyuni"] = NS._minmax(-jyuni)

    # --- hold_babasa: 持続タイムベンチマーク走の馬場差。符号はHOLD_BABASA_SIGNに従う。
    babasa = NS._num(NS._col(df, "holdtime_just_babasa"))
    sig["hold_babasa"] = NS._minmax(HOLD_BABASA_SIGN * babasa)

    return sig


def detect_dead_bottom(entries: list, priors_bottom: dict, names: list = None) -> list:
    """_minmax後に全レース全馬NaNになる新規シグナルを検出する(nar_signals.detect_deadと
    同じ設計、対象をNEW_SIGNALS_*に絞ったもの)。"""
    names = names or NEW_SIGNALS_ALL
    alive = {n: 0 for n in names}
    for e in entries:
        sig = build_bottom_signals(e["df"], NS.class_ordinal(e["race_name"]), priors_bottom,
                                   track_b=e.get("track_b", False))
        for n in names:
            if n in sig and sig[n].notna().any():
                alive[n] += 1
    return [n for n in names if alive[n] == 0]


def signal_matrices_bottom(entries: list, priors_all: dict, priors_bottom: dict,
                           names: list) -> list:
    """既存24本(向き反転流用)+新規候補シグナルを1つの(S, A)行列に統合する。
    names 内の各シグナル名は nar_signals.ALL_SIGNALS のいずれか、または NEW_SIGNALS_ALL の
    いずれかである必要がある。score = (S @ w) / (A @ w) がcombine()と一致する(既存パターン踏襲)。
    符号規約は「高いほど4着以内で終わりやすい」なので、下位予測で使う際は
    呼び出し側でargsortの向きを反転すること(ここでは反転しない)。"""
    mats = []
    for e in entries:
        current_class = NS.class_ordinal(e["race_name"])
        base_sig = NS.build_signals(e["df"], current_class, priors_all)
        new_sig = build_bottom_signals(e["df"], current_class, priors_bottom,
                                       track_b=e.get("track_b", False))
        merged = {**base_sig, **new_sig}
        cols = [merged[n].to_numpy(dtype=float) for n in names]
        M = np.column_stack(cols)
        A = (~np.isnan(M)).astype(float)
        S = np.nan_to_num(M, nan=0.0)
        mats.append({"S": S, "A": A})
    return mats
