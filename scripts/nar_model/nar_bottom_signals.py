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

後方互換拡張(2026-08-25、K=3〜8スイープ対応、SEレビュー反映版)。凡走専用の新規4本
(既存シグナルの反転流用ではなく、凡走という事象そのものをモデル化する):
  pace_clash_risk    : Track A全期間。既存style signalと同一のpace_pressure定義
                       (自分を含む「逃/先」比率、レース単位で全馬共通値)を再利用し、
                       方向(差/追=+1、逃/先=-1、不明=NaN)との積で「展開との相性」を
                       0..1固定スケールに写像する。逃げ/先行馬がハイペース(pace_pressure
                       高)な展開に飛び込むほど値が低い(=凡走リスク高)。不明ラベルは
                       中立値ではなくNaN(combine()の重み再配分に委ねる)。Phase 0で
                       既存style/nigeとの相関を実測し、閾値超なら自動除外するVIFゲート対象。
  layoff_return_risk : Track A全期間。kaisai_date(YYYYMMDD)とpast1_date("%Y.%m.%d")の
                       日数差(gap_days)をlog1p変換し、休み明けほど値が低い(=凡走
                       リスク高)固定スケールに写像する。kaisai_date未指定時・
                       gap_days<=0(異常値)はNaN。
  class_rank_vs_field: Track A全期間(旧称class_jump_relative_risk)。直近1〜3走の
                       平均クラス序列(class_hike/class_dropと同じpast{i}_race_name/
                       past{i}_race_class参照パターン)と、レース内全馬平均との差分を
                       固定スケールで0..1に写像する。フィールド平均より格上のクラスを
                       走ってきた馬ほど値が高い(=凡走リスク低)。
  distance_stretch_risk: Track Bのみ(既存distanceと同じ制約)。data_distance_slot1_runs
                       (今回と同じ距離での経験本数)をlog1p変換し、経験が少ないほど
                       値が低い(=凡走リスク高)固定スケールに写像する。

SCALE定数(LAYOFF_SCALE_DAYS/CLASS_RANK_SCALE/DISTANCE_STRETCH_SCALE_RUNS)は、
nar_search_bottom_k_sweep_2026_08_25.py のPhase 0で実測した分布(p90等)を根拠に
確定した値。
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LIB_DIR))
import nar_signals as NS  # noqa: E402  既存モジュール、無改造・関数を読むだけ

CLASS_HIKE_SCALE = 3.0  # class_dropと同じ「3クラス分で満点」スケール

# 凡走専用新規4本(2026-08-25)のSCALE定数。Phase 0実測分布に基づき確定した値。
LAYOFF_SCALE_DAYS = 60.0          # gap日数分布(p90≈43)を踏まえたlog1p変換スケール
CLASS_RANK_SCALE = 3.0            # class_hike/class_dropと同じ「3クラス分で満点」スケール
DISTANCE_STRETCH_SCALE_RUNS = 20.0  # 同距離経験本数分布(p90≈25)を踏まえたlog1p変換スケール

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

NEW_SIGNALS_TRACK_A = ["class_hike"]  # 全期間で利用可能(旧来のまま、無改造)
NEW_SIGNALS_TRACK_B_ONLY = [
    "distance_breadth", "course_breadth", "overall_rate",
    "hold_l3f", "hold_jyuni", "hold_babasa",
]  # 旧来のまま、無改造
NEW_SIGNALS_ALL = NEW_SIGNALS_TRACK_A + NEW_SIGNALS_TRACK_B_ONLY

# 凡走専用新規4本(2026-08-25、K=3〜8スイープ用)。既存のNEW_SIGNALS_TRACK_A/
# NEW_SIGNALS_TRACK_B_ONLY/NEW_SIGNALS_ALLを直接拡張すると、これらをモジュール属性経由で
# 参照しているnar_search_bottom_2026_08_24.py(無改造のまま残す前回スクリプト)の
# 挙動まで変わってしまう(dead_bottom_a/dead_bottom_bの対象シグナル集合が変わり、
# POOL_2A/POOL_2B以下すべての数値が再現しなくなる)。そのため既存3リストには一切
# 触れず、新規スイープ専用の別リストとして追加する。
NEW_SIGNALS_TRACK_A_BOTTOM_V2 = ["pace_clash_risk", "layoff_return_risk", "class_rank_vs_field"]
NEW_SIGNALS_TRACK_B_ONLY_BOTTOM_V2 = ["distance_stretch_risk"]
NEW_SIGNALS_ALL_BOTTOM_V2 = (
    NEW_SIGNALS_ALL + NEW_SIGNALS_TRACK_A_BOTTOM_V2 + NEW_SIGNALS_TRACK_B_ONLY_BOTTOM_V2
)


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


def _parse_kaisai_date(raw) -> "dt.date | None":
    try:
        return dt.datetime.strptime(str(raw).strip(), "%Y%m%d").date()
    except (ValueError, TypeError):
        return None


def build_bottom_signals(df: pd.DataFrame, current_class: float, priors_bottom: dict,
                         track_b: bool, kaisai_date=None) -> dict:
    """1レース分の新規候補シグナル辞書を返す(0..1正規化、高いほどgood)。
    track_b=False の場合は Track B専用シグナル(distance_breadth等)を計算しない
    (列自体が存在しないため計算しても全NaNになるだけ、無駄な計算を避ける)。

    kaisai_date(YYYYMMDD、既定None)はlayoff_return_riskの算出にのみ使う。省略時
    (既存呼び出しとの後方互換のため既定None)はlayoff_return_riskが全NaNになる
    (combine()の重み再配分に委ねる、他の新規シグナルには影響しない)。"""
    sig = {}

    # --- class_hike: 前走からのクラス昇格幅(class_dropの符号反転版、正しい極性は
    # 「1 - 昇格幅/SCALE」。class_dropの値をそのまま符号反転すると[-1,0]域になり
    # combine()が前提とする0..1規約を壊すため、この形にする)。
    past1_class = NS._col(df, "past1_race_name").map(NS.class_ordinal)
    past1_class = past1_class.fillna(NS._col(df, "past1_race_class").map(NS.class_ordinal))
    hike = (current_class - past1_class).clip(lower=0.0)
    sig["class_hike"] = 1.0 - (hike / CLASS_HIKE_SCALE).clip(upper=1.0)

    # --- pace_clash_risk(Track A全期間、VIFゲート対象): 既存styleと同一定義の
    # pace_pressure(自分を含む「逃/先」比率、レース単位で全馬共通)× 展開方向
    # (差/追=+1、逃/先=-1、不明=NaN)。逃げ/先行馬がハイペースな展開に飛び込むほど
    # 値が低い(=凡走リスク高)。不明ラベルは中立値(0.5)ではなくNaNにする
    # (weight_trendプレースホルダ0バグと同型の「欠損を中立扱いしない」規律)。
    style_label = NS._col(df, "ca_running_style_category_label").astype(str).str.strip()
    n_field = max(len(df), 1)  # ゼロ除算ガード
    pace_pressure = style_label.isin(NS.RUNNING_STYLE_FRONT).sum() / n_field
    direction = style_label.map(
        lambda s: 1.0 if s in NS.RUNNING_STYLE_CLOSE else (-1.0 if s in NS.RUNNING_STYLE_FRONT else np.nan)
    )
    good_raw = direction * pace_pressure
    sig["pace_clash_risk"] = ((good_raw + 1.0) / 2.0).clip(lower=0.0, upper=1.0)

    # --- layoff_return_risk(Track A全期間): kaisai_date - past1_date の日数差(gap_days)を
    # log1p変換した固定スケール写像。休み明けほど値が低い(=凡走リスク高)。
    kd = _parse_kaisai_date(kaisai_date) if kaisai_date is not None else None
    if kd is None:
        sig["layoff_return_risk"] = pd.Series(np.nan, index=df.index)
    else:
        past1_date = pd.to_datetime(NS._col(df, "past1_date"), format="%Y.%m.%d", errors="coerce")
        gap_days = (pd.Timestamp(kd) - past1_date).dt.days.astype(float)
        gap_days = gap_days.where(gap_days > 0, np.nan)  # gap_days<=0は異常値としてNaN
        risk = np.log1p(gap_days) / np.log1p(LAYOFF_SCALE_DAYS)
        sig["layoff_return_risk"] = (1.0 - risk.clip(upper=1.0)).clip(lower=0.0)

    # --- class_rank_vs_field(Track A全期間、旧称class_jump_relative_risk): 直近1〜3走の
    # 平均クラス序列と、レース内全馬平均との差分を固定スケールで写像する。フィールド平均
    # より格上のクラスを走ってきた馬ほど値が高い(=凡走リスク低)。
    past_classes = []
    for i in (1, 2, 3):
        c = NS._col(df, f"past{i}_race_name").map(NS.class_ordinal)
        c = c.fillna(NS._col(df, f"past{i}_race_class").map(NS.class_ordinal))
        past_classes.append(c)
    own_avg_class = pd.concat(past_classes, axis=1).mean(axis=1, skipna=True)
    field_avg_class = own_avg_class.mean(skipna=True)
    if pd.isna(field_avg_class):
        sig["class_rank_vs_field"] = pd.Series(np.nan, index=df.index)
    else:
        diff = own_avg_class - field_avg_class
        sig["class_rank_vs_field"] = ((diff / CLASS_RANK_SCALE).clip(lower=-1.0, upper=1.0) + 1.0) / 2.0

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

    # --- distance_stretch_risk(Track Bのみ、既存distanceと同じ制約): 今回と同じ距離での
    # 経験本数(data_distance_slot1_runs)をlog1p変換した固定スケール写像。経験が少ないほど
    # 値が低い(=凡走リスク高)。
    stretch_runs = NS._num(NS._col(df, "data_distance_slot1_runs")).clip(lower=0.0)
    stretch_risk = np.log1p(stretch_runs) / np.log1p(DISTANCE_STRETCH_SCALE_RUNS)
    sig["distance_stretch_risk"] = stretch_risk.clip(upper=1.0)

    return sig


def detect_dead_bottom(entries: list, priors_bottom: dict, names: list = None) -> list:
    """_minmax後に全レース全馬NaNになる新規シグナルを検出する(nar_signals.detect_deadと
    同じ設計、対象をNEW_SIGNALS_*に絞ったもの)。"""
    names = names or NEW_SIGNALS_ALL
    alive = {n: 0 for n in names}
    for e in entries:
        sig = build_bottom_signals(e["df"], NS.class_ordinal(e["race_name"]), priors_bottom,
                                   track_b=e.get("track_b", False), kaisai_date=e.get("kaisai_date"))
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
                                       track_b=e.get("track_b", False), kaisai_date=e.get("kaisai_date"))
        merged = {**base_sig, **new_sig}
        cols = [merged[n].to_numpy(dtype=float) for n in names]
        M = np.column_stack(cols)
        A = (~np.isnan(M)).astype(float)
        S = np.nan_to_num(M, nan=0.0)
        mats.append({"S": S, "A": A})
    return mats
