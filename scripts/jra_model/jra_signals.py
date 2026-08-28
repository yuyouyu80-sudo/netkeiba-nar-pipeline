# -*- coding: utf-8 -*-
"""JRA通常戦(pattern29系)モデルのシグナル構築 — 単一の真実の源(single source of truth)。

これまで scripts/predict_pattern29.py と scratchpad の predict.py / predict_box4.py /
predict_box3.py がそれぞれ別実装のシグナル生成を持っていた(SHA256で別ファイルと確認済み)。
これはNAR側が過去に実際に踏んだ地雷と同型: scripts/nar_model/predict_box4_nar.py の
docstringが記録する通り、探索側と本番側でpriorsが食い違い(15キー中12キー、distance_returnは
78.23 vs 60.64と29%乖離)、71レース中3レースでBOX4の顔ぶれ自体が変わっていた事故がNARにはあった。
以後NARは nar_signals.py を単一の真実の源として統合した。JRA側も同じ設計にする。

設計方針(NARのnar_signals.pyと同じ):
  * weights・priors・class_ordinal_map は必ず引数で渡す。モジュールグローバルへの暗黙依存は
    作らない(探索スクリプトが別のpriors/weightsを試すときに、本番と食い違う経路を作らせないため)。
  * 欠損シグナルの重みは、値のあるシグナルへ再配分する(combine_signals)。
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# mark_honshi/mark_cp/mark_other(◎○▲△☆の記号文字列)の数値化は、パーサー側の
# MARK_SCORE(mark_list_parser.py)を単一の真実の源として再利用する(NARのnar_signals.pyと
# 同じ、重複定義しない、2026-08-28 JRA Stage2)。
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.netkeiba_pipeline.parsers.mark_list_parser import MARK_SCORE  # noqa: E402

# --- 既存10シグナル(scripts/predict_pattern29.py L73-78のWEIGHTSキーと同一)。
LEGACY_SIGNALS = ["speed", "form", "style", "jt", "waku", "apt", "train", "distance", "sire", "bms"]
# --- 候補シグナル(2026-08-11、未使用データ調査を受けて追加)。
CANDIDATE_SIGNALS = ["course", "concerned", "interval", "kinryo", "nige", "margin", "timediff",
                     "agari", "holdtime"]
# --- 候補シグナル第2弾(2026-08-12)。専門家レビューで「surf_ketto/surf_jockeyは既存シグナル
# (sire/bms/jt)と高相関のはず」という当初の除外仮定が実データで否定された(相関係数0.03〜0.31)
# ことを受けて追加。surf_ketto_*は血統×調教の掛け合わせ集計、surf_jockey_*/surf_odds_jockeyは
# 騎手×調教師・騎手×オッズ・騎手×前走騎手(乗り替わり関連)の掛け合わせ集計。
CANDIDATE_SIGNALS_V2 = ["ketto_training", "ketto_comment", "odds_jockey", "surf_jt",
                        "jockey_owner", "prevjockey"]
# --- 候補シグナル第3弾(2026-08-21)。NARのCANDIDATE_SIGNALS_V4(scripts/nar_model/nar_signals.py
# L396-427)をJRAへ移植。JRA固有の列差分に合わせて2本を変更している:
#   jockey_change: NARはbias_jockey(今回騎手)とpast1_jockey(前走騎手)の単純不一致判定だが、
#     JRAのbias_jockeyは略称形式(例: "松本")でpast1_jockeyはフルネーム形式(例: "富田暁")のため
#     一致率が0.3%しかなく直接比較すると常時「乗り替わり」判定になり壊れる。代わりに
#     フルネーム形式のca_jockey_category_label(past1_jockeyとの一致率41.9%、NARの39.8%と近い
#     水準)を「今回騎手」として使う。
#   class_drop: NARのCLASS_DROP_SCALE=3.0は15段階クラス向けの値で、JRAの8段階整数刻みには
#     粗すぎる(3クラス分降格は3勝→未勝利のような極めて稀なケースでしか満点にならない)ため
#     1.2に再較正する。また"重賞"/"JpnII"/"JpnI"は現行CLASS_ORDINAL(部分文字列一致)の
#     どのキーにも一致しない(Gの文字を含まないため。"JGIII"/"JGII"は既存の"GIII"/"GII"に
#     既に一致する)ので、class_drop専用にCLASS_ORDINAL_V3EXTを新設して対応する。
CANDIDATE_SIGNALS_V3 = ["hold_just", "hold_wide", "jockey_change", "class_drop", "weight_trend"]
ALL_SIGNALS = LEGACY_SIGNALS + CANDIDATE_SIGNALS + CANDIDATE_SIGNALS_V2 + CANDIDATE_SIGNALS_V3
# --- 候補シグナル第4弾(2026-08-28、JRA Stage2: 全ファクター統合・高確信度選抜計画のJRA移植)。
# NARのCANDIDATE_SIGNALS_V5(scripts/nar_model/nar_signals.py L122-127、2026-08-27新設)を
# そのままJRAへ移植。予想印(mark_honshi/mark_cp/mark_other、netkeiba「本紙・CP予想・その他」)と
# 展開予想(corner3/4_rank・corner3/4_gap_lengths・corner4_speedup、netkeiba「AI展開」)。
# 列名・MARK_SCORE・符号の向きはNARと完全同一(JRAのnewspaper CSVも同じスキーマで取得済み、
# 2026-08-28実データ確認: 504ファイル中504がmark_honshi充足・489が corner4_rank充足)。
#   mark_honshi_score/mark_cp_score/mark_other_score: 各印(◎○▲△☆)をMARK_SCOREで数値化し
#     レース内minmax。列自体が欠測のレースは全馬0.0→_minmaxのhi==lo分岐で自動的に全NaN化
#     (class_dropと同型の安全設計)。
#   mark_composite_score: 本紙・CP・その他3列のMARK_SCORE合計をminmax。
#   corner4_position/corner3_position: 4/3コーナー通過順位(小さいほど先頭)を符号反転して
#     minmax(高いほど先頭に近い=高評価、他シグナルと符号を揃える)。
#   corner4_gap/corner3_gap: 4/3コーナー時点の先頭との推定差(馬身)を符号反転してminmax。
#   corner4_speedup: 4コーナーでの加速マーク(▶)の数(0-3)をそのままminmax。
#   corner_transition_rank/corner_transition_gap: 3→4コーナーの順位変化・先頭差変化
#     (正=押し上げ/詰めた)をminmax。
CANDIDATE_SIGNALS_V4_MARK = ["mark_honshi_score", "mark_cp_score", "mark_other_score",
                             "mark_composite_score"]
CANDIDATE_SIGNALS_V4_CORNER = ["corner4_position", "corner4_gap", "corner4_speedup",
                               "corner3_position", "corner3_gap",
                               "corner_transition_rank", "corner_transition_gap"]
CANDIDATE_SIGNALS_V4 = CANDIDATE_SIGNALS_V4_MARK + CANDIDATE_SIGNALS_V4_CORNER
# --- 重要: ALL_SIGNALSにV4(mark/corner)を含めない。既存スクリプト(jra_search500_2026_08_21_
# v3signals.py等)がJS.ALL_SIGNALSを実行時に直接参照しているため(NARのALL_SIGNALS/ALL_SIGNALS_V5
# と同じ理由、nar_signals.py L130-134のコメント参照)。新規スクリプトはこちらのALL_SIGNALS_V4を使う。
ALL_SIGNALS_V4 = ALL_SIGNALS + CANDIDATE_SIGNALS_V4

TRAIN_RANK_MAP = {"S": 6, "A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
DNF_FINISH_PENALTY = 20
DNF_CODES = {"中止", "取消", "除外", "失格", "中", "取", "除"}
CLASS_ORDINAL = {
    "新馬": 0, "未勝利": 0,
    "1勝": 1, "2勝": 2, "3勝": 3,
    "OP": 4, "オープン": 4, "L": 4,
    "G3": 5, "GIII": 5, "G2": 6, "GII": 6, "G1": 7, "GI": 7,
}
CLASS_ADJUST_PER_LEVEL = 1.5
# class_drop専用の拡張クラス序列(2026-08-21)。既存CLASS_ORDINAL(formシグナル等が依存する
# 本体)は変更せず、past1側の降格幅計算にのみ使う加算専用マップ。実データ確認済み: JRAの
# past1_race_class等に"重賞"/"JGIII"/"JpnII"/"JpnI"/"JGII"(計15件)が出現するが、"JGIII"/"JGII"は
# 既存の"GIII"/"GII"キーに部分文字列一致で既に解決されるため、未解決の3キー("重賞"/"JpnII"/
# "JpnI")のみ追加する。"JpnIII"は現行データに出現しないが将来出現に備えて併せて追加する。
# _class_ordinal()は部分文字列一致を先勝ちで採用するため、"JpnI"が"JpnII"/"JpnIII"の
# プレフィックスになる(2026-08-21統計学者ゲート1レビュー指摘)。dict挿入順=判定順なので、
# 必ず長い方から先に並べる("JpnIII"→"JpnII"→"JpnI"の順、崩すと短い方に誤マッチする)。
CLASS_ORDINAL_V3EXT = {**CLASS_ORDINAL, "重賞": 5.0, "JpnIII": 5.0, "JpnII": 6.0, "JpnI": 7.0}
CLASS_DROP_SCALE = 1.2  # NARの3.0(15段階クラス向け)をJRAの8段階整数刻みに合わせて再較正
RUNNING_STYLE_FRONT = {"逃", "先"}
RUNNING_STYLE_CLOSE = {"差", "追"}
SHRINK_K = 12.0

# 既存10シグナルのシュリンケージ仕様(predict_pattern29.py L85-101と同一)。
SHRINK_SPECS = {
    "style_win": ("ca_running_style_win_rate", "ca_running_style_runs"),
    "style_place3": ("ca_running_style_place3_rate", "ca_running_style_runs"),
    "jockey_win": ("ca_jockey_win_rate", "ca_jockey_runs"),
    "trainer_win": ("ca_trainer_win_rate", "ca_trainer_runs"),
    "waku_win": ("ca_waku_win_rate", "ca_waku_runs"),
    "apt_win": ("ca_speed_index_win_rate", "ca_speed_index_runs"),
    "distance_win": ("data_distance_slot1_win_rate", "data_distance_slot1_runs"),
    "distance_place3": ("data_distance_slot1_place3_rate", "data_distance_slot1_runs"),
    "distance_return": ("data_distance_slot1_win_return_rate", "data_distance_slot1_runs"),
    "sire_win": ("ca_sire_win_rate", "ca_sire_runs"),
    "sire_place3": ("ca_sire_place3_rate", "ca_sire_runs"),
    "sire_return": ("ca_sire_win_return_rate", "ca_sire_runs"),
    "bms_win": ("ca_broodmare_sire_win_rate", "ca_broodmare_sire_runs"),
    "bms_place3": ("ca_broodmare_sire_place3_rate", "ca_broodmare_sire_runs"),
    "bms_return": ("ca_broodmare_sire_win_return_rate", "ca_broodmare_sire_runs"),
    # --- 候補シグナル用(2026-08-11追加)。course/concernedはca_*と同じ構造の集計列。
    "course_win": ("data_course_slot1_win_rate", "data_course_slot1_runs"),
    "course_place3": ("data_course_slot1_place3_rate", "data_course_slot1_runs"),
    "course_return": ("data_course_slot1_win_return_rate", "data_course_slot1_runs"),
    "concerned_win": ("concerned_win_rate", "concerned_runs"),
    "concerned_place3": ("concerned_place3_rate", "concerned_runs"),
    "concerned_return": ("concerned_win_return_rate", "concerned_runs"),
    # interval(休養日数帯)・kinryo(斤量帯)は data.html?mode=others の集団統計。
    # 実データ確認済み: JRAではslot番号がNARと異なる(NARはslot1=休養/slot2=斤量、
    # JRAの実サンプルはslot1=休養/slot2=クラス/slot3=斤量/slot4=馬体重)。slot番号を
    # 固定で信頼せず、_label列の文字列パターンで動的に解決する(_resolve_others_slot)。
    "interval_win": (None, None),  # _resolve_others_slotで実列名に解決してから_shrinkに渡す
    "interval_place3": (None, None),
    "interval_return": (None, None),
    "kinryo_win": (None, None),
    "kinryo_place3": (None, None),
    "kinryo_return": (None, None),
    # --- 候補シグナル第2弾用(2026-08-12)。surf_ketto_*/surf_jockey_*/surf_odds_jockeyは
    # win_rate/place3_rate/win_return_rate + runs という既存のcourse/concernedと全く同じ構造。
    "ketto_training_win": ("surf_ketto_training_win_rate", "surf_ketto_training_runs"),
    "ketto_training_place3": ("surf_ketto_training_place3_rate", "surf_ketto_training_runs"),
    "ketto_training_return": ("surf_ketto_training_win_return_rate", "surf_ketto_training_runs"),
    "ketto_comment_win": ("surf_ketto_comment_win_rate", "surf_ketto_comment_runs"),
    "ketto_comment_place3": ("surf_ketto_comment_place3_rate", "surf_ketto_comment_runs"),
    "ketto_comment_return": ("surf_ketto_comment_win_return_rate", "surf_ketto_comment_runs"),
    "odds_jockey_win": ("surf_odds_jockey_win_rate", "surf_odds_jockey_runs"),
    "odds_jockey_place3": ("surf_odds_jockey_place3_rate", "surf_odds_jockey_runs"),
    "odds_jockey_return": ("surf_odds_jockey_win_return_rate", "surf_odds_jockey_runs"),
    "surf_jt_win": ("surf_jockey_trainer_win_rate", "surf_jockey_trainer_runs"),
    "surf_jt_place3": ("surf_jockey_trainer_place3_rate", "surf_jockey_trainer_runs"),
    "surf_jt_return": ("surf_jockey_trainer_win_return_rate", "surf_jockey_trainer_runs"),
    "jockey_owner_win": ("surf_jockey_owner_win_rate", "surf_jockey_owner_runs"),
    "jockey_owner_place3": ("surf_jockey_owner_place3_rate", "surf_jockey_owner_runs"),
    "jockey_owner_return": ("surf_jockey_owner_win_return_rate", "surf_jockey_owner_runs"),
    # prevjockey(前走騎手×今回騎手の掛け合わせ、乗り替わり関連): 専門家レビューで既存jt
    # シグナルとほぼ無相関(sire/bms)、jtとも中程度(r=0.24)と判明した最有望候補。
    "prevjockey_win": ("surf_jockey_prevjockey_win_rate", "surf_jockey_prevjockey_runs"),
    "prevjockey_place3": ("surf_jockey_prevjockey_place3_rate", "surf_jockey_prevjockey_runs"),
    "prevjockey_return": ("surf_jockey_prevjockey_win_return_rate", "surf_jockey_prevjockey_runs"),
}

_MARGIN_RE = re.compile(r"\(([-+]?\d+\.?\d*)\)")
_OTHERS_SLOTS = ["data_others_slot1", "data_others_slot2", "data_others_slot3", "data_others_slot4"]


# --------------------------------------------------------------------------- utils
def _num(series: pd.Series) -> pd.Series:
    cleaned = series.where(~series.astype(str).isin(["-", "--", "nan", ""]), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def _pct(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace("%", "", regex=False)
    cleaned = cleaned.where(~cleaned.isin(["-", "--", "nan", ""]), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name and name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index)


def _finish_with_dnf_penalty(series: pd.Series) -> pd.Series:
    text = series.astype(str)
    is_dnf = text.isin(DNF_CODES)
    numeric = _num(series)
    return numeric.where(~is_dnf, DNF_FINISH_PENALTY)


def _minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.nan, index=s.index)
    return (s - lo) / (hi - lo)


def _blend_minmax(*series_list) -> pd.Series:
    cols = [_minmax(s) for s in series_list]
    return pd.concat(cols, axis=1).mean(axis=1, skipna=True)


def _wavg(frame: pd.DataFrame, weights) -> pd.Series:
    wa = np.array(weights, dtype=float)
    ws = (frame.fillna(0) * wa).sum(axis=1)
    wt = (frame.notna() * wa).sum(axis=1).replace(0, np.nan)
    return ws / wt


def _first_corner(s):
    try:
        return float(str(s).split("-")[0])
    except Exception:
        return np.nan


def _margin(s):
    m = _MARGIN_RE.search(str(s))
    return float(m.group(1)) if m else np.nan


_TIME_RE = re.compile(r"^(\d+):(\d+\.?\d*)$")


def _time_to_sec(s) -> float:
    """"1:28.9" 形式(分:秒.コンマ)を秒(float)に変換する。holdtime_*_time用(NARから移植)。"""
    m = _TIME_RE.match(str(s).strip())
    if not m:
        return np.nan
    minutes, seconds = m.groups()
    return float(minutes) * 60.0 + float(seconds)


def _class_ordinal(text, class_ordinal_map: dict) -> float:
    if pd.isna(text):
        return np.nan
    t = str(text).strip()
    for key, val in class_ordinal_map.items():
        if key in t:
            return val
    return np.nan


def _drop_scratched(df: pd.DataFrame) -> pd.DataFrame:
    odds = _num(df["bias_win_odds"])
    ninki = _num(df["bias_ninki"])
    return df[odds.notna() & ninki.notna()].reset_index(drop=True)


def _resolve_others_slot(df: pd.DataFrame, kind: str) -> tuple:
    """kind: "interval"(休養日数帯、ラベルに"ヶ月"か"連闘")または"kinryo"(斤量帯、ラベルに"kg")。
    JRAのdata.html?mode=othersはslot番号がNARと異なり(実データ確認済み: JRAはslot1=休養/
    slot2=クラス/slot3=斤量/slot4=馬体重)、レースによって順序が変わる可能性も排除できないため、
    固定slot番号を信頼せず_label列の中身を見て動的に解決する。該当スロットが1件も無ければ
    (None, None)を返す(呼び出し側はNaNシグナル=欠損として扱う)。"""
    for base in _OTHERS_SLOTS:
        label_col = f"{base}_label"
        if label_col not in df.columns:
            continue
        labels = df[label_col].astype(str)
        if kind == "kinryo":
            hit = labels.str.contains("kg", case=False, na=False)
        else:  # interval
            hit = labels.str.contains("ヶ月|連闘|週", na=False)
        if hit.any():
            return f"{base}_win_rate", f"{base}_runs"
    return None, None


def _shrink(df: pd.DataFrame, key: str, priors: dict, shrink_specs: dict = None) -> pd.Series:
    specs = shrink_specs if shrink_specs is not None else SHRINK_SPECS
    rate_col, runs_col = specs[key]
    rate = _pct(_col(df, rate_col))
    runs = _num(_col(df, runs_col)).fillna(0.0)
    # priorsに該当キーが無い(旧いwinner_v3.jsonにはcourse/concerned/interval/kinryoの
    # priorsが無い等)場合はNaNを返し、combine_signals側の重み再配分に委ねる(NARのnar_signals.py
    # と同じ防御的パターン)。
    prior = priors.get(key, np.nan)
    if pd.isna(prior):
        return pd.Series(np.nan, index=df.index)
    return (rate.fillna(prior) * runs + prior * SHRINK_K) / (runs + SHRINK_K)


# --------------------------------------------------------------------------- priors
def make_priors(dfs: list) -> dict:
    """渡されたレースのdf群(=学習fold)からシュリンケージの事前値を作る。NARのmake_priorsと同じ:
    全欠損の項目はNaNになり、そのシグナルは_shrinkで常に欠損=combine_signalsが重み再配分する。"""
    priors = {}
    resolved_specs_cache = {}
    for key, (rate_col, _runs) in SHRINK_SPECS.items():
        if rate_col is None:
            # interval/kinryoはdf(レース)ごとにslotが動的なため、_others_shrink_specsで
            # 実列名を解決してから集計する(列名の文字列組み立てをここで再実装して重複させない
            # ため、compute_signalsと同じ_others_shrink_specsを再利用する)。
            kind = "kinryo" if key.startswith("kinryo") else "interval"
            vals_list = []
            for df in dfs:
                cache_key = (id(df), kind)
                if cache_key not in resolved_specs_cache:
                    resolved_specs_cache[cache_key] = _others_shrink_specs(df, kind)
                col, _runs_col = resolved_specs_cache[cache_key].get(key, (None, None))
                if col is None:
                    continue
                vals_list.append(_pct(_col(df, col)))
            vals = pd.concat(vals_list, ignore_index=True) if vals_list else pd.Series([np.nan])
        else:
            vals = pd.concat([_pct(_col(df, rate_col)) for df in dfs], ignore_index=True)
        priors[key] = float(vals.mean(skipna=True))
    return priors


# --------------------------------------------------------------------------- signals
def compute_signals(df: pd.DataFrame, current_class_ordinal: float, priors: dict,
                    class_ordinal_map: dict = None) -> dict:
    """1レース分のシグナル辞書を返す。値はすべて0..1に正規化されたpd.Series(欠損はNaN)。
    weightsに一切依存しない(=候補シグナルを1000パターン探索する際、レースごとに1回だけ
    呼べば良い。重み合成だけをcombine_signalsで繰り返す)。"""
    class_map = class_ordinal_map if class_ordinal_map is not None else CLASS_ORDINAL
    sig = {}

    # --- speed (既存)
    speed_cols = ["speed_max_index", "speed_avg_index_5races", "speed_index_1race_ago"]
    speed_avg = pd.concat([_num(_col(df, c)) for c in speed_cols], axis=1).mean(axis=1, skipna=True)
    sig["speed"] = _minmax(speed_avg)

    # --- form (既存、直近3走)
    finishes = {}
    for i in (1, 2, 3):
        raw = _finish_with_dnf_penalty(_col(df, f"past{i}_finish"))
        past_class = _col(df, f"past{i}_race_class").map(lambda t: _class_ordinal(t, class_map))
        class_gap = current_class_ordinal - past_class
        adjustment = (class_gap * CLASS_ADJUST_PER_LEVEL).where(class_gap.notna(), 0.0)
        finishes[i] = raw - adjustment
    past_finish_df = pd.DataFrame(finishes)
    sig["form"] = _minmax(-_wavg(past_finish_df, [3, 2, 1]))

    # --- style (既存)
    style_rate = _blend_minmax(_shrink(df, "style_win", priors), _shrink(df, "style_place3", priors))
    style_label = _col(df, "ca_running_style_category_label").astype(str).str.strip()
    front_count = style_label.isin(RUNNING_STYLE_FRONT).sum()
    field_n = max(len(df), 1)
    pace_pressure = front_count / field_n
    pace_direction = style_label.map(
        lambda s: 1.0 if s in RUNNING_STYLE_CLOSE else (-1.0 if s in RUNNING_STYLE_FRONT else 0.0)
    )
    pace_adjustment = pace_direction * (pace_pressure - 0.35)
    sig["style"] = _minmax(style_rate.fillna(0.5) + pace_adjustment)

    # --- jt / waku / apt / train (既存)
    jt_rate = pd.concat([_shrink(df, "jockey_win", priors), _shrink(df, "trainer_win", priors)],
                        axis=1).mean(axis=1)
    sig["jt"] = _minmax(jt_rate)
    sig["waku"] = _minmax(_shrink(df, "waku_win", priors))
    sig["apt"] = _minmax(_shrink(df, "apt_win", priors))
    training = _col(df, "training_rank").astype(str).str.strip().str.upper().map(TRAIN_RANK_MAP)
    sig["train"] = _minmax(training)

    # --- distance / sire / bms (既存)
    sig["distance"] = _blend_minmax(
        _shrink(df, "distance_win", priors), _shrink(df, "distance_place3", priors),
        _shrink(df, "distance_return", priors)
    )
    sig["sire"] = _blend_minmax(
        _shrink(df, "sire_win", priors), _shrink(df, "sire_place3", priors),
        _shrink(df, "sire_return", priors)
    )
    sig["bms"] = _blend_minmax(
        _shrink(df, "bms_win", priors), _shrink(df, "bms_place3", priors),
        _shrink(df, "bms_return", priors)
    )

    # --- course(候補、2026-08-11): 当該競馬場での成績(data_course_slot1)
    sig["course"] = _blend_minmax(
        _shrink(df, "course_win", priors), _shrink(df, "course_place3", priors),
        _shrink(df, "course_return", priors)
    )
    # --- concerned(候補): 当該コース+距離ちょうどでの成績
    sig["concerned"] = _blend_minmax(
        _shrink(df, "concerned_win", priors), _shrink(df, "concerned_place3", priors),
        _shrink(df, "concerned_return", priors)
    )
    # --- interval(候補): 休養日数帯適性。slotはlabel文字列から動的解決。
    interval_specs = _others_shrink_specs(df, "interval")
    sig["interval"] = _blend_minmax(
        _shrink(df, "interval_win", priors, interval_specs),
        _shrink(df, "interval_place3", priors, interval_specs),
        _shrink(df, "interval_return", priors, interval_specs),
    )
    # --- kinryo(候補): 斤量帯適性。slotはlabel文字列から動的解決。
    kinryo_specs = _others_shrink_specs(df, "kinryo")
    sig["kinryo"] = _blend_minmax(
        _shrink(df, "kinryo_win", priors, kinryo_specs),
        _shrink(df, "kinryo_place3", priors, kinryo_specs),
        _shrink(df, "kinryo_return", priors, kinryo_specs),
    )

    # --- nige(候補): 過去5走の(第1)コーナー通過順を頭数で割った相対位置(小さいほど前)。
    cps = []
    for i in range(1, 6):
        c = _col(df, f"past{i}_corner_positions").map(_first_corner)
        fs = _num(_col(df, f"past{i}_field_size"))
        cps.append((c / fs.where(fs > 0)).rename(i))
    sig["nige"] = _minmax(-_wavg(pd.concat(cps, axis=1), [5, 4, 3, 2, 1]))

    # --- margin(候補): 直近3走の着差("相手馬名(1.2)"形式の括弧内、単位=馬身)。
    ms3 = pd.concat([_col(df, f"past{i}_beaten_by").map(_margin).rename(i) for i in (1, 2, 3)], axis=1)
    sig["margin"] = _minmax(-_wavg(ms3, [3, 2, 1]))

    # --- timediff(候補): marginの5走版(NARで「一番惜しかったが不採用」となった前例あり)。
    ms5 = pd.concat([_col(df, f"past{i}_beaten_by").map(_margin).rename(i) for i in range(1, 6)], axis=1)
    sig["timediff"] = _minmax(-_wavg(ms5, [5, 4, 3, 2, 1]))

    # --- agari(候補): 直近3走の上がり3F平均(小さいほど良い)。
    ag = pd.concat([_num(_col(df, f"past{i}_agari_3f")).rename(i) for i in (1, 2, 3)], axis=1)
    sig["agari"] = _minmax(-ag.mean(axis=1, skipna=True))

    # --- holdtime(候補、JRA独自・NARに前例なし): 「持続時間」データの近い距離帯(just)の
    # 上がり3F。全馬に該当レースがあるとは限らず欠損率が高い可能性があるため、死にシグナル
    # 検出(非NaN率の実測)で生存を確認してから探索プールに残す前提。
    sig["holdtime"] = _minmax(-_num(_col(df, "holdtime_just_l3f")))

    # --- 候補シグナル第2弾(2026-08-12): surf_ketto_*/surf_jockey_*/surf_odds_jockey。
    # 2026-08-11の専門家レビューで「既存sire/bms/jtと高相関のはず」という当初の除外仮定が
    # 実データ(相関係数0.03〜0.31)で否定されたため追加。
    sig["ketto_training"] = _blend_minmax(
        _shrink(df, "ketto_training_win", priors), _shrink(df, "ketto_training_place3", priors),
        _shrink(df, "ketto_training_return", priors)
    )
    sig["ketto_comment"] = _blend_minmax(
        _shrink(df, "ketto_comment_win", priors), _shrink(df, "ketto_comment_place3", priors),
        _shrink(df, "ketto_comment_return", priors)
    )
    sig["odds_jockey"] = _blend_minmax(
        _shrink(df, "odds_jockey_win", priors), _shrink(df, "odds_jockey_place3", priors),
        _shrink(df, "odds_jockey_return", priors)
    )
    sig["surf_jt"] = _blend_minmax(
        _shrink(df, "surf_jt_win", priors), _shrink(df, "surf_jt_place3", priors),
        _shrink(df, "surf_jt_return", priors)
    )
    sig["jockey_owner"] = _blend_minmax(
        _shrink(df, "jockey_owner_win", priors), _shrink(df, "jockey_owner_place3", priors),
        _shrink(df, "jockey_owner_return", priors)
    )
    sig["prevjockey"] = _blend_minmax(
        _shrink(df, "prevjockey_win", priors), _shrink(df, "prevjockey_place3", priors),
        _shrink(df, "prevjockey_return", priors)
    )

    # --- 候補シグナル第3弾(2026-08-21、NARのCANDIDATE_SIGNALS_V4をJRAへ移植)。
    # hold_just/hold_wide: 持続タイム。速いほど高評価なので符号反転してminmax。既存holdtime
    # (holdtime_just_l3f=上がり3F専用)とは別列だが同一データソース由来のため、探索時に
    # 相関を確認すること。
    sig["hold_just"] = _minmax(-_col(df, "holdtime_just_time").map(_time_to_sec))
    wide_cols = ["holdtime_short_time", "holdtime_middle_time", "holdtime_long_time"]
    wide = pd.concat([_col(df, c).map(_time_to_sec) for c in wide_cols], axis=1).mean(axis=1, skipna=True)
    sig["hold_wide"] = _minmax(-wide)

    # --- jockey_change: 前走からの騎手乗り替わり(1.0=乗り替わり/0.0=継続/NaN=前走情報なし)。
    # 「今回騎手」はbias_jockey(略称、past1_jockeyとの一致率0.3%で使うと壊れる)ではなく
    # ca_jockey_category_label(フルネーム、一致率41.9%)を使う(JRA固有の対応、モジュール
    # docstring・CANDIDATE_SIGNALS_V3コメント参照)。
    today_jockey = _col(df, "ca_jockey_category_label").astype(str).str.strip()
    past1_jockey = _col(df, "past1_jockey").astype(str).str.strip()
    has_past1 = _col(df, "past1_jockey").notna() & (past1_jockey != "")
    changed = (today_jockey != past1_jockey).astype(float)
    sig["jockey_change"] = changed.where(has_past1, np.nan)

    # --- class_drop: 前走からのクラス降格幅(下がった分のみ、上がっても0)。レース内相対の
    # minmaxではなく固定スケール(CLASS_DROP_SCALE)へ写像する(降格馬0頭のレースでも
    # "降格していない"という情報が全馬0.0のまま残る設計、NARで発見された_minmaxのhi==lo
    # 全NaN化バグを踏襲して回避)。past1側の解決にはCLASS_ORDINAL_V3EXT(重賞/Jpn系を追加
    # 収録)を使う。currentClassOrdinal自体は既存のclass_ordinal_map(呼び出し元指定、
    # 通常CLASS_ORDINAL)のまま変更しない(formシグナルの挙動に影響させないため)。
    past1_class_v3 = _col(df, "past1_race_name").map(lambda t: _class_ordinal(t, CLASS_ORDINAL_V3EXT))
    past1_class_v3 = past1_class_v3.fillna(
        _col(df, "past1_race_class").map(lambda t: _class_ordinal(t, CLASS_ORDINAL_V3EXT))
    )
    drop = (past1_class_v3 - current_class_ordinal).clip(lower=0.0)
    sig["class_drop"] = (drop / CLASS_DROP_SCALE).clip(upper=1.0)

    # --- weight_trend: 過去1〜5走の馬体重増減の加重平均トレンド。増加方向+変化の小ささ
    # (安定)をブレンドする(NARと完全同一ロジック、JRA側で現状未使用の列)。
    # 2026-08-21競馬予想家ゲート1レビュー指摘・実データで独立検証済み: past{i}_race_nameが
    # 「新馬」(=その過去走自体がデビュー戦)の場合、netkeibaは比較対象の前走が無いため
    # past{i}_horse_weight_diffをプレースホルダの0で埋めている(実測: 前走が新馬の馬は
    # 25/25=100%が0固定、通常は545頭中73頭=13.4%)。これを実測値のまま使うと
    # 「体重変化なし=安定」と誤判定する系統的バイアスになるため、該当スロットはNaN扱いにする。
    wt_cols = []
    for i in range(1, 6):
        diff = _num(_col(df, f"past{i}_horse_weight_diff"))
        is_debut = _col(df, f"past{i}_race_name").astype(str).str.contains("新馬", na=False)
        wt_cols.append(diff.where(~is_debut, np.nan).rename(i))
    wt = pd.concat(wt_cols, axis=1)
    trend = _wavg(wt, [5, 4, 3, 2, 1])
    sig["weight_trend"] = _blend_minmax(trend, -trend.abs())

    # --- mark_*(2026-08-28、候補シグナル第4弾・JRA Stage2): 予想印(本紙・CP予想・その他)を
    # MARK_SCORE(mark_list_parser.pyと共通)で数値化してminmax。列が丸ごと欠測のレース
    # (印データ未取得)は全馬0.0→_minmaxのhi==lo分岐で自動的に全NaN化される(NARのnar_signals.py
    # と完全同一ロジック)。
    mark_honshi_num = _col(df, "mark_honshi").map(MARK_SCORE).fillna(0.0)
    mark_cp_num = _col(df, "mark_cp").map(MARK_SCORE).fillna(0.0)
    mark_other_num = _col(df, "mark_other").map(MARK_SCORE).fillna(0.0)
    sig["mark_honshi_score"] = _minmax(mark_honshi_num)
    sig["mark_cp_score"] = _minmax(mark_cp_num)
    sig["mark_other_score"] = _minmax(mark_other_num)
    sig["mark_composite_score"] = _minmax(mark_honshi_num + mark_cp_num + mark_other_num)

    # --- corner_*(2026-08-28、候補シグナル第4弾・JRA Stage2): 展開予想(AI展開)の3・4コーナー
    # 通過順位・先頭との推定差(馬身)。順位・差はいずれも「小さいほど先頭に近い」ため
    # 符号反転してminmax(他シグナルと「高いほど高評価」の向きを揃える、NARと完全同一ロジック)。
    c4_rank = _num(_col(df, "corner4_rank"))
    c4_gap = _num(_col(df, "corner4_gap_lengths"))
    c3_rank = _num(_col(df, "corner3_rank"))
    c3_gap = _num(_col(df, "corner3_gap_lengths"))
    sig["corner4_position"] = _minmax(-c4_rank)
    sig["corner4_gap"] = _minmax(-c4_gap)
    sig["corner4_speedup"] = _minmax(_num(_col(df, "corner4_speedup")))
    sig["corner3_position"] = _minmax(-c3_rank)
    sig["corner3_gap"] = _minmax(-c3_gap)
    # 3→4角の変化: 正=順位アップ/先頭差を詰めた。片方でも欠測ならNaN(pd.Seriesの引き算は
    # 自動的にNaN伝播するため追加処理は不要)。
    sig["corner_transition_rank"] = _minmax(c3_rank - c4_rank)
    sig["corner_transition_gap"] = _minmax(c3_gap - c4_gap)

    return sig


def _others_shrink_specs(df: pd.DataFrame, kind: str) -> dict:
    """このdf(1レース)におけるinterval/kinryoの実列名をSHRINK_SPECS形式で返す。"""
    rate_col, runs_col = _resolve_others_slot(df, kind)
    if rate_col is None:
        return {f"{kind}_win": (None, None), f"{kind}_place3": (None, None),
                f"{kind}_return": (None, None)}
    base = rate_col.rsplit("_win_rate", 1)[0]
    return {
        f"{kind}_win": (f"{base}_win_rate", runs_col),
        f"{kind}_place3": (f"{base}_place3_rate", runs_col),
        f"{kind}_return": (f"{base}_win_return_rate", runs_col),
    }


def combine_signals(signals: dict, weights: dict) -> pd.Series:
    """欠損シグナルの重みを、値のあるシグナルへ再配分して合成する。"""
    index = next(iter(signals.values())).index
    total_score = pd.Series(0.0, index=index)
    total_weight = pd.Series(0.0, index=index)
    for name, w in weights.items():
        if w <= 0 or name not in signals:
            continue
        s = signals[name]
        avail = s.notna()
        total_score = total_score + s.fillna(0) * w
        total_weight = total_weight + avail.astype(float) * w
    return pd.Series(np.where(total_weight > 0, total_score / total_weight, np.nan), index=index)


def score_race(df: pd.DataFrame, current_class_ordinal: float, weights: dict, priors: dict,
              class_ordinal_map: dict = None) -> pd.DataFrame:
    """compute_signals + combine_signals を呼ぶ薄いラッパー。既存のscore_race(df, current_class)
    と同じ戻り値契約(df に _score 列を追加して返す)を維持する。"""
    df = df.copy()
    signals = compute_signals(df, current_class_ordinal, priors, class_ordinal_map)
    df["_score"] = combine_signals(signals, weights).to_numpy()
    return df


def signal_matrices(races: list, priors: dict, names: list, class_ordinal_map: dict = None) -> list:
    """重み探索の高速化用。レースごとに(S, A)行列を事前計算しておけば、以後は
    score = (S @ w) / (A @ w) という行列演算だけで何百通りもの重みベクトルを評価できる
    (combine_signalsをレースごと・候補ごとに毎回呼ぶより大幅に速い)。S[i,j]はシグナルjの
    馬iの値(欠損は0埋め)、A[i,j]は値の有無(1.0/0.0)。combine_signalsの
    「sum(w*fillna(0)) / sum(w*avail)」という重み付き平均の定義と数学的に同値。"""
    class_map = class_ordinal_map if class_ordinal_map is not None else CLASS_ORDINAL
    mats = []
    for r in races:
        current_class = _class_ordinal(r["race_name"], class_map)
        sig = compute_signals(r["df"], current_class, priors, class_map)
        S = np.column_stack([sig[n].fillna(0.0).to_numpy(dtype=float) for n in names])
        A = np.column_stack([sig[n].notna().to_numpy(dtype=float) for n in names])
        mats.append({"S": S, "A": A})
    return mats
