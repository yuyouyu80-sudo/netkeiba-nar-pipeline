# -*- coding: utf-8 -*-
"""NAR予想モデルのシグナル構築 — 単一の真実の源(single source of truth)。

これまで search_patterns_box4_nar.py / predict_pattern29.py / confidence_sweep_box4_nar.py が
それぞれ別実装のシグナル生成を持っており、同じ重みでもスクリプト間で回収率が最大8.6pt
食い違っていた(priorsの不一致が原因)。以後はこのモジュールだけを使う。

レビュー指摘に基づく変更点:
  * 死にシグナルの判定を「列の充填率」ではなく「_minmax後の非NaN率」で行う。
    ca_broodmare_sire_* は列は100%埋まっているが全馬が値0のため _minmax の hi==lo 分岐で
    全NaNになる。列の充填率では捕まらない。
  * クラス補正の参照列を past{i}_race_class(NAR充填率2.2%・序列マッチ1.5%)から
    past{i}_race_name(マッチ率84.5%)に変更し、C4/B3/B4/A3/OP を序列に追加した。
  * priors は「渡されたレース集合」から計算する。fold内学習側だけで計算できるようにして、
    探索側と本番側で別のpriorsを使ってしまう事故を構造的に防ぐ。
"""
import re

import numpy as np
import pandas as pd

# --- 旧スキーマの10キー。winner_*.json / predict_pattern29.py / build_artifact_nar.py が
# --- このキー集合を前提にしているため、値が0でも必ず残す。
LEGACY_SIGNALS = ["speed", "form", "style", "jt", "waku", "apt", "train", "distance", "sire", "bms"]
# --- 4頭BOX再モデリング(2026-07-29)で追加したシグナル。
NEW_SIGNALS = ["jockey", "trainer", "nige", "concerned", "course", "jt_return", "margin", "agari"]
# --- 3頭BOX再モデリング(2026-07-29)で追加した候補シグナル。休養日数・斤量帯は
# data.html?mode=others の集団統計(この馬自身が過去その帯でどう走ったか)。
# 馬場状態適性は発走前に確定しないため採用を見送った(専門家・エンジニア両レビューで、
# 検証時だけ実測値が使えて本番では使えない=未来情報のリークになると指摘されたため)。
# 血統系統(父系統・母父系統)も、in-sample構築のリーク対策コストと予測力への懸念から
# 今回は見送った。「bmsが死んでいる原因の切り分けが先」という当時の指摘は2026-08-04に
# 解消済み: NARのcoursedata cid=4(broodmare_sire)ページを実際に取得し、cid=1(sire、
# NARで生存)・JRAのcid=4ページと突き合わせた結果、パーサー/URLの不具合ではなく
# netkeiba側がNAR向け母父集計を公開していないこと自体が原因と確定した(詳細は
# src/netkeiba_pipeline/scrapers/course_data.py のコメント参照)。bmsは今後も
# 構造欠測として扱ってよく、修正の見込みはない。血統系統シグナルの見送り自体は
# (リーク対策コスト・予測力の懸念という)別の理由で引き続き有効なため変更なし。
CANDIDATE_SIGNALS = ["interval", "kinryo"]
# --- ユーザー依頼(2026-07-29)で追加した候補シグナル。馬柱データ(past1-5)・
# 当日再取得したbias_horse_weightから作る、この馬自身の個別成績ベースの新規シグナル。
#   timediff    : 直近5走の1着とのタイム差(秒)。既存marginの3走版を5走に拡張した独立シグナル。
#   class_ninki : 直近5走を「当時のクラス差」+「当時の人気順を上回った/下回った度合い」で
#                 補正した近走成績。formのクラス補正に、人気(市場の事前予想)との乖離を追加。
#   weight      : 直近で再取得したbias_horse_weightの増減幅(絶対値)。増減が小さいほど
#                 コンディション安定とみなし高スコアにする。
CANDIDATE_SIGNALS_V2 = ["timediff", "class_ninki", "weight"]
# --- ユーザー依頼(2026-08-01)で追加した候補シグナル。course_analysisの全期間waku勝率とは
# 別に、レース当日の直前"連続2暦日"(D-1・D-2の両方に該当競馬場の開催実績が無い場合は
# 欠損=NaN)における同競馬場の枠番別勝率。馬場は数日単位で偏ることがあるという着眼。
# 値の算出は nar_factor_test_waku_recent2d.py 側(race_results履歴が必要なため
# build_signals()の入力であるdfに事前注入する形を取る、他のCANDIDATE同様)。
CANDIDATE_SIGNALS_V3 = ["waku_recent2d"]
ALL_SIGNALS = LEGACY_SIGNALS + NEW_SIGNALS + CANDIDATE_SIGNALS + CANDIDATE_SIGNALS_V2 + CANDIDATE_SIGNALS_V3

# NARでは値が構造的に存在しないことを実測で確認したシグナル(686頭全件)。
# ハードコードではなく detect_dead() で毎回検出するが、既定値としても持っておく。
KNOWN_DEAD = ["speed", "apt", "train", "bms"]

SHRINK_K = 12.0
DNF_FINISH_PENALTY = 20
DNF_CODES = {"中止", "取消", "除外", "失格", "中", "取", "除"}
CLASS_ADJUST_PER_LEVEL = 1.5
NINKI_ADJUST_WEIGHT = 1.0
RUNNING_STYLE_FRONT = {"逃", "先"}
RUNNING_STYLE_CLOSE = {"差", "追"}

# 地方競馬のクラス序列。数字が大きいほど上位クラス。
CLASS_ORDINAL_NAR = {
    "新馬": 0.0, "未勝利": 0.5,
    "C4": 0.5, "C3": 1.0, "C1C2": 2.5, "C2": 2.0, "C1": 3.0,
    "B4": 3.2, "B3": 3.5, "B2": 4.0, "B1": 5.0,
    "A3": 5.5, "A2": 6.0, "A1": 7.0,
    "OP": 7.5, "オープン": 7.5,
    "重賞": 8.0, "G1": 8.0, "G2": 8.0, "G3": 8.0,
    "Jpn1": 8.0, "Jpn2": 8.0, "Jpn3": 8.0,
}
# 長いキーから先に照合しないと "C1" が "C1C2" を食う。
_CLASS_KEYS = sorted(CLASS_ORDINAL_NAR, key=len, reverse=True)

SHRINK_SPECS = {
    "style_win": ("ca_running_style_win_rate", "ca_running_style_runs"),
    "style_place3": ("ca_running_style_place3_rate", "ca_running_style_runs"),
    "jockey_win": ("ca_jockey_win_rate", "ca_jockey_runs"),
    "jockey_return": ("ca_jockey_win_return_rate", "ca_jockey_runs"),
    "trainer_win": ("ca_trainer_win_rate", "ca_trainer_runs"),
    "trainer_return": ("ca_trainer_win_return_rate", "ca_trainer_runs"),
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
    "course_win": ("data_course_slot1_win_rate", "data_course_slot1_runs"),
    "course_place3": ("data_course_slot1_place3_rate", "data_course_slot1_runs"),
    "course_return": ("data_course_slot1_win_return_rate", "data_course_slot1_runs"),
    "concerned_win": ("concerned_win_rate", "concerned_runs"),
    "concerned_place3": ("concerned_place3_rate", "concerned_runs"),
    "concerned_return": ("concerned_win_return_rate", "concerned_runs"),
    # data.html?mode=others の slot1=休養日数帯("中3週"等)・slot2=斤量帯("57kg"等)。
    # この馬自身が過去その帯でどう走ったかの集団統計(distance/courseと同じ形の項目別データ)。
    "interval_win": ("data_others_slot1_win_rate", "data_others_slot1_runs"),
    "interval_place3": ("data_others_slot1_place3_rate", "data_others_slot1_runs"),
    "interval_return": ("data_others_slot1_win_return_rate", "data_others_slot1_runs"),
    "kinryo_win": ("data_others_slot2_win_rate", "data_others_slot2_runs"),
    "kinryo_place3": ("data_others_slot2_place3_rate", "data_others_slot2_runs"),
    "kinryo_return": ("data_others_slot2_win_return_rate", "data_others_slot2_runs"),
    # 直近連続2暦日・同競馬場の枠番別勝率(%スケール、他のrate列と同じ規約)。
    # nar_factor_test_waku_recent2d.pyがdfに事前注入する列を読む。
    "waku_recent2d_win": ("recent2d_waku_win_rate", "recent2d_waku_runs"),
}

_MARGIN_RE = re.compile(r"\(([-+]?\d+\.?\d*)\)")
# bias_horse_weight の生値("462(+2)"/"430(0)"/"445()"/"0()"/空欄)。build_artifact_nar.pyの
# format_horse_weight()と同じ正規表現・同じ「0=測定不能」規約を使う。
_WEIGHT_RE = re.compile(r"^\s*(\d+)\(([+-]?\d*)\)\s*$")


def _weight_delta(raw) -> float:
    if pd.isna(raw) or str(raw).strip() == "":
        return np.nan
    m = _WEIGHT_RE.match(str(raw))
    if not m:
        return np.nan
    weight, diff = m.group(1), m.group(2)
    if weight == "0" or diff == "":
        return np.nan
    try:
        return float(diff)
    except ValueError:
        return np.nan


# --------------------------------------------------------------------------- utils
def _num(s: pd.Series) -> pd.Series:
    c = s.where(~s.astype(str).isin(["-", "--", "nan", ""]), np.nan)
    return pd.to_numeric(c, errors="coerce")


def _pct(s: pd.Series) -> pd.Series:
    c = s.astype(str).str.replace("%", "", regex=False)
    c = c.where(~c.isin(["-", "--", "nan", ""]), np.nan)
    return pd.to_numeric(c, errors="coerce")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def _minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.nan, index=s.index)
    return (s - lo) / (hi - lo)


def _blend_minmax(*series_list) -> pd.Series:
    return pd.concat([_minmax(s) for s in series_list], axis=1).mean(axis=1, skipna=True)


def class_ordinal(text) -> float:
    """レース名から地方競馬のクラス序列を取り出す。見つからなければNaN。"""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return np.nan
    t = str(text).strip()
    for key in _CLASS_KEYS:
        if key in t:
            return CLASS_ORDINAL_NAR[key]
    return np.nan


def _finish_with_dnf_penalty(series: pd.Series) -> pd.Series:
    is_dnf = series.astype(str).isin(DNF_CODES)
    return _num(series).where(~is_dnf, DNF_FINISH_PENALTY)


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


# --------------------------------------------------------------------------- priors
def make_priors(entries: list) -> dict:
    """渡されたレース集合(=学習fold)からシュリンケージの事前値を作る。
    全欠損の項目はNaNになる。NaNのままだと _shrink が全NaNを返し、combine() が
    そのシグナルの重みを他へ再配分する(=実質的に使われない)。意図した挙動。"""
    priors = {}
    for key, (rate_col, _runs) in SHRINK_SPECS.items():
        vals = pd.concat([_pct(_col(e["df"], rate_col)) for e in entries], ignore_index=True)
        priors[key] = float(vals.mean(skipna=True))
    return priors


def _shrink(df: pd.DataFrame, key: str, priors: dict) -> pd.Series:
    rate_col, runs_col = SHRINK_SPECS[key]
    rate = _pct(_col(df, rate_col))
    runs = _num(_col(df, runs_col)).fillna(0.0)
    # priorsは旧バージョンのwinner_*.jsonから読まれることがあり、そのシグナルが
    # 追加される前に保存されたJSONにはキー自体が無い。その場合はNaNを返し
    # (=このモデルではそのシグナルは常に欠損)、combine()側の重み再配分に委ねる。
    prior = priors.get(key, np.nan)
    if pd.isna(prior):
        return pd.Series(np.nan, index=df.index)
    return (rate.fillna(prior) * runs + prior * SHRINK_K) / (runs + SHRINK_K)


# --------------------------------------------------------------------------- signals
def build_signals(df: pd.DataFrame, current_class: float, priors: dict) -> dict:
    """1レース分のシグナル辞書を返す。値はすべて 0..1 に正規化された pd.Series(欠損はNaN)。"""
    sig = {}

    # --- speed: NARでは speed_* 列が全て空(実測0%)。列があれば使う。
    speed_cols = ["speed_max_index", "speed_avg_index_5races", "speed_index_1race_ago"]
    speed_avg = pd.concat([_num(_col(df, c)) for c in speed_cols], axis=1).mean(axis=1, skipna=True)
    sig["speed"] = _minmax(speed_avg)

    # --- form: 直近3走の着順(クラス差で補正)。クラスは past{i}_race_name から取る。
    finishes = {}
    for i in (1, 2, 3):
        raw = _finish_with_dnf_penalty(_col(df, f"past{i}_finish"))
        past_class = _col(df, f"past{i}_race_name").map(class_ordinal)
        fallback = _col(df, f"past{i}_race_class").map(class_ordinal)
        past_class = past_class.fillna(fallback)
        gap = current_class - past_class
        finishes[i] = raw - (gap * CLASS_ADJUST_PER_LEVEL).where(gap.notna(), 0.0)
    sig["form"] = _minmax(-_wavg(pd.DataFrame(finishes), [3, 2, 1]))

    # --- style: 脚質別成績 + そのレースの前残りしやすさによる補正
    style_rate = _blend_minmax(_shrink(df, "style_win", priors), _shrink(df, "style_place3", priors))
    label = _col(df, "ca_running_style_category_label").astype(str).str.strip()
    pace_pressure = label.isin(RUNNING_STYLE_FRONT).sum() / max(len(df), 1)
    direction = label.map(lambda s: 1.0 if s in RUNNING_STYLE_CLOSE else (-1.0 if s in RUNNING_STYLE_FRONT else 0.0))
    sig["style"] = _minmax(style_rate.fillna(0.5) + direction * (pace_pressure - 0.35))

    # --- jt(騎手・調教師の勝率を平均)と、それを分離した jockey / trainer
    jk = _shrink(df, "jockey_win", priors)
    tr = _shrink(df, "trainer_win", priors)
    sig["jt"] = _minmax(pd.concat([jk, tr], axis=1).mean(axis=1))
    sig["jockey"] = _minmax(jk)
    sig["trainer"] = _minmax(tr)
    sig["jt_return"] = _blend_minmax(_shrink(df, "jockey_return", priors),
                                     _shrink(df, "trainer_return", priors))

    sig["waku"] = _minmax(_shrink(df, "waku_win", priors))
    sig["apt"] = _minmax(_shrink(df, "apt_win", priors))
    sig["train"] = _minmax(_col(df, "training_rank").astype(str).str.strip().str.upper()
                           .map({"S": 6, "A": 5, "B": 4, "C": 3, "D": 2, "E": 1}))

    sig["distance"] = _blend_minmax(_shrink(df, "distance_win", priors),
                                    _shrink(df, "distance_place3", priors),
                                    _shrink(df, "distance_return", priors))
    sig["sire"] = _blend_minmax(_shrink(df, "sire_win", priors), _shrink(df, "sire_place3", priors),
                                _shrink(df, "sire_return", priors))
    sig["bms"] = _blend_minmax(_shrink(df, "bms_win", priors), _shrink(df, "bms_place3", priors),
                               _shrink(df, "bms_return", priors))
    sig["course"] = _blend_minmax(_shrink(df, "course_win", priors),
                                  _shrink(df, "course_place3", priors),
                                  _shrink(df, "course_return", priors))
    # concerned = 当該コース+距離ちょうどでのその馬自身の成績
    sig["concerned"] = _blend_minmax(_shrink(df, "concerned_win", priors),
                                     _shrink(df, "concerned_place3", priors),
                                     _shrink(df, "concerned_return", priors))

    # --- nige: 過去5走の1コーナー通過順を頭数で割った相対位置(小さいほど前)。
    cps = []
    for i in range(1, 6):
        c = _col(df, f"past{i}_corner_positions").map(_first_corner)
        fs = _num(_col(df, f"past{i}_field_size"))
        cps.append((c / fs.where(fs > 0)).rename(i))
    sig["nige"] = _minmax(-_wavg(pd.concat(cps, axis=1), [5, 4, 3, 2, 1]))

    # --- margin: 直近3走の着差(秒)。"相手馬名(1.2)" 形式なので括弧内を取る。
    ms = pd.concat([_col(df, f"past{i}_beaten_by").map(_margin).rename(i) for i in (1, 2, 3)], axis=1)
    sig["margin"] = _minmax(-_wavg(ms, [3, 2, 1]))

    # --- agari: 直近3走の上がり3F平均(小さいほど良い)
    ag = pd.concat([_num(_col(df, f"past{i}_agari_3f")).rename(i) for i in (1, 2, 3)], axis=1)
    sig["agari"] = _minmax(-ag.mean(axis=1, skipna=True))

    # --- interval: 休養日数帯適性(3頭BOX再モデリングで追加した候補シグナル)
    sig["interval"] = _blend_minmax(_shrink(df, "interval_win", priors),
                                    _shrink(df, "interval_place3", priors),
                                    _shrink(df, "interval_return", priors))
    # --- kinryo: 斤量帯適性(同上)
    sig["kinryo"] = _blend_minmax(_shrink(df, "kinryo_win", priors),
                                  _shrink(df, "kinryo_place3", priors),
                                  _shrink(df, "kinryo_return", priors))

    # --- timediff: 直近5走の1着とのタイム差(秒)。marginの3走版を5走に拡張した独立シグナル。
    ms5 = pd.concat([_col(df, f"past{i}_beaten_by").map(_margin).rename(i) for i in range(1, 6)], axis=1)
    sig["timediff"] = _minmax(-_wavg(ms5, [5, 4, 3, 2, 1]))

    # --- class_ninki(2026-07-29 v2、ユーザー指摘を反映): 「当時のクラスで当時の人気順」は
    # 当時の実力でのその馬の市場価値、という考え方に基づき2成分をブレンドする。
    #   market_value = 当時のクラス序列 × (1/当時の人気順)。クラスが高いほど・人気が高い
    #     (人気順の数字が小さい)ほど高評価。市場価値そのものの代理指標(結果は問わない)。
    #   upset_bonus  = 人気順を上回る着順(実際の着順 < 人気順)となった場合のみ加点する
    #     片側ボーナス。下回った(凡走した)場合は減点しない。
    mv, bonus = {}, {}
    for i in range(1, 6):
        raw = _finish_with_dnf_penalty(_col(df, f"past{i}_finish"))
        ninki = _num(_col(df, f"past{i}_ninki"))
        past_class = _col(df, f"past{i}_race_name").map(class_ordinal)
        fallback = _col(df, f"past{i}_race_class").map(class_ordinal)
        past_class = past_class.fillna(fallback)
        mv[i] = past_class * (1.0 / ninki)
        bonus[i] = (ninki - raw).clip(lower=0)
    market_value = _wavg(pd.DataFrame(mv), [5, 4, 3, 2, 1])
    upset_bonus = _wavg(pd.DataFrame(bonus), [5, 4, 3, 2, 1])
    sig["class_ninki"] = _blend_minmax(market_value, upset_bonus)

    # --- weight(2026-07-29 v2、ユーザー指摘を反映): 当日再取得したbias_horse_weightの
    # 増減幅。プラス(増加)ほど高評価の成分と、絶対値が小さいほど高評価の成分をブレンドする
    # (他の複合シグナルと同じ _blend_minmax パターン)。
    wd = _col(df, "bias_horse_weight").map(_weight_delta)
    sig["weight"] = _blend_minmax(wd, -wd.abs())

    # --- waku_recent2d(2026-08-01、ユーザー依頼): 直前連続2暦日・同競馬場の枠番別勝率。
    # 通常のwaku(全期間)と違い標本が極小(1枠あたり数レース)になるため、SHRINK_K=12.0の
    # シュリンケージで大半がpriorに引き戻される設計。
    sig["waku_recent2d"] = _minmax(_shrink(df, "waku_recent2d_win", priors))

    return sig


def combine(sig: dict, weights: dict, index) -> np.ndarray:
    """欠損シグナルの重みを、値のあるシグナルへ再配分して合成する。"""
    total_score = pd.Series(0.0, index=index)
    total_weight = pd.Series(0.0, index=index)
    for name, w in weights.items():
        if w <= 0:
            continue
        s = sig[name]
        total_score = total_score + s.fillna(0) * w
        total_weight = total_weight + s.notna().astype(float) * w
    return np.where(total_weight > 0, total_score / total_weight, np.nan)


# --------------------------------------------------------------------------- dead detection
def detect_dead(entries: list, priors: dict, names=None) -> list:
    """_minmax後に「全レースで全馬NaN」になるシグナルを死にシグナルとして返す。
    列の充填率ではなくシグナル値で判定するため、bms のような「値はあるが全馬同値」も捕まる。"""
    names = names or ALL_SIGNALS
    alive = {n: 0 for n in names}
    for e in entries:
        sig = build_signals(e["df"], class_ordinal(e["race_name"]), priors)
        for n in names:
            if sig[n].notna().any():
                alive[n] += 1
    return [n for n in names if alive[n] == 0]


def signal_matrices(entries: list, priors: dict, names: list) -> list:
    """ベクトル化バックテスト用に、レースごとの (S, A) を作る。
      S: (n_horses, n_signals) NaNを0で埋めた値
      A: (n_horses, n_signals) 値が存在すれば1、欠損なら0
    score = (S @ w) / (A @ w) が combine() と厳密に一致する。"""
    mats = []
    for e in entries:
        sig = build_signals(e["df"], class_ordinal(e["race_name"]), priors)
        cols = [sig[n].to_numpy(dtype=float) for n in names]
        M = np.column_stack(cols)
        A = (~np.isnan(M)).astype(float)
        S = np.nan_to_num(M, nan=0.0)
        mats.append({"S": S, "A": A})
    return mats
