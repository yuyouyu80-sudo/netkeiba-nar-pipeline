# -*- coding: utf-8 -*-
"""ユーザー依頼(2026-09-04、「予想ファクター充足度マップで『有』と判定済みなのに
ファクター検証データベース側で未露出のファクターを全部足してほしい」)による追加ファクター v3。

v1(`jra_candidate_factors_v1.py`)・v2(`jra_candidate_factors_v2.py`)と同じ設計方針を踏襲:
既存の本番スコア(`jra_model_scoring.py`)・レーダー8カテゴリ(`jra_radar_categories.py`)は
無改造で、このモジュールが完全に自己完結して計算する追加専用の列を返す。
build_factor_dataset.py から呼ばれ、factor_database.json のファクターとしてのみ使われる
(predict_pattern29.py 等の本番予想には一切反映しない)。

v1/v2 が「既存 df + race_results だけで計算できる項目」だったのに対し、v3 は
**2026-09-02のTier0〜Tier4で新設された外部CSVを結合する**点が新しい:

  - data/jockey_profile/{jockey_id}.csv  … 騎手の年間リーディング順位・所属エリア・所属形態・通算勝利数
  - data/trainer_profile/{trainer_id}.csv … 調教師の年間リーディング順位・所属エリア
  - data/horse_profile/{horse_id}.csv    … 生産牧場(breeder)・馬主(owner)
  - data/jra_baba/{year}/{venue}_{kai}.csv … クッション値・含水率・仮柵(JRA公式PDF由来)
  - data/jra_course_master.csv           … 回り方向・直線距離・高低差・周長(JRA10場の静的表)

加えて、充足度マップで「有(取得済み)」と判定済みなのにデータベース側のフィルタとして
露出していなかった **既存 df 列の基本条件**(性別・年齢・斤量・馬体重・単勝オッズ帯・
レース間隔・想定ペース・4コーナー想定位置・サーフェス)もここでまとめて列化する。

結合キーの重要な注意(2026-09-04、実データ確認済み):
  現母集団(2026-07-11〜08-23の491レース)の newspaper CSV には
  `bias_jockey_id`/`bias_trainer_id`/`bias_jockey_allowance_mark` が **1件も存在しない**
  (bias_parser.py がこれらを出力するようになったのは2026-09-02以降の新規取得分から)。
  そのため騎手・調教師・馬主のIDは **data/race_results 側の jockey_id/trainer_id/owner_id**
  を (race_id, horse_id) キーで引いて使う(実測で6,443行すべて突合成功)。
  減量ジョッキー記号は本母集団では取得不能のため、代替として jockey_profile の
  通算勝利数バンド(jockey_career_wins_band)を露出する。

point-in-time 整合性について:
  - 馬主の過去勝率(owner_prior_win_rate)のみ、当該レース日より前の走に限定して集計している。
  - 騎手・調教師プロフィール(リーディング順位・通算勝利数)は netkeiba の**現在値スナップ
    ショット**であり、レース時点の値ではない(fetch_person_profile.py は時系列データとして
    毎回上書きする設計)。母集団が2026年7〜8月・プロフィールが2026シーズン値なので
    大きなズレは無い見込みだが、厳密な point-in-time ではない点を extra_note に明記する。
"""
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

sys.path.insert(0, str(LIB_DIR))
import jra_signals as JS  # noqa: E402

# affiliation_area / affiliation_type は netkeiba の表記ゆれで
# "地方 ( 騎手成績 )" のような括弧付き接尾辞が混ざる(実データ確認済み)。
_PAREN_RE = re.compile(r"\s*[\(（].*?[\)）]\s*")
_DATE_DOT_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}$")

# 美浦=関東、栗東=関西のホーム開催場(輸送の有無判定用)。中京は関西圏として扱う。
_KANTO_VENUES = {"東京", "中山"}
_KANSAI_VENUES = {"京都", "阪神", "中京"}

_SHADAI_GROUP_OTHER = {"社台コーポレーション白老ファーム", "追分ファーム"}


def _clean(s):
    if s is None or (isinstance(s, float) and pd.isna(s)) or pd.isna(s):
        return None
    t = _PAREN_RE.sub("", str(s)).strip()
    return t or None


def _to_int(s):
    if s is None or pd.isna(s):
        return None
    try:
        return int(str(s).replace(",", "").strip())
    except ValueError:
        return None


def _to_float(s):
    if s is None or pd.isna(s):
        return None
    try:
        return float(str(s).replace(",", "").replace("％", "").strip())
    except ValueError:
        return None


def _parse_dot_date(s):
    """past1_date の "2026.07.05" 形式 -> Timestamp(v2 と同じ、パース不能は NaT)。"""
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    if not _DATE_DOT_RE.match(s):
        return pd.NaT
    return pd.to_datetime(s, format="%Y.%m.%d", errors="coerce")


# --------------------------------------------------------------------------
# 外部CSVのロード
# --------------------------------------------------------------------------
def load_person_profiles(kind: str) -> dict:
    """data/{kind}_profile/*.csv を {id: {…}} で返す(1ファイル1行)。"""
    out = {}
    for path in glob.glob(str(DATA_ROOT / f"{kind}_profile" / "*.csv")):
        try:
            d = pd.read_csv(path, dtype=str)
        except Exception:
            continue
        if d.empty:
            continue
        row = d.iloc[0]
        pid = str(row.get("id") or os.path.basename(path)[:-4]).strip()
        out[pid] = {
            "area": _clean(row.get("affiliation_area")),
            "affiliation_type": _clean(row.get("affiliation_type")),
            "season": _clean(row.get("jra_season")),
            "season_rank": _to_int(row.get("jra_season_rank")),
            "career_wins": _to_int(row.get("jra_career_wins")),
        }
    return out


def load_horse_profiles() -> dict:
    out = {}
    for path in glob.glob(str(DATA_ROOT / "horse_profile" / "*.csv")):
        try:
            d = pd.read_csv(path, dtype=str)
        except Exception:
            continue
        if d.empty:
            continue
        row = d.iloc[0]
        hid = str(row.get("horse_id") or os.path.basename(path)[:-4]).strip()
        out[hid] = {"breeder": _clean(row.get("breeder")), "owner": _clean(row.get("owner"))}
    return out


def load_course_master() -> dict:
    """venue_code(2桁ゼロ埋め文字列) -> dict。"""
    path = DATA_ROOT / "jra_course_master.csv"
    d = pd.read_csv(path, dtype={"venue_code": str})
    out = {}
    for _, row in d.iterrows():
        out[str(row["venue_code"]).zfill(2)] = {
            "venue": row["venue"],
            "turn_direction": row["turn_direction"],
            "turf_straight_m": _to_float(row.get("turf_straight_m")),
            "turf_elevation_m": _to_float(row.get("turf_elevation_m")),
            "turf_circumference_m": _to_float(row.get("turf_circumference_m")),
            "dirt_straight_m": _to_float(row.get("dirt_straight_m")),
            "dirt_elevation_m": _to_float(row.get("dirt_elevation_m")),
            "dirt_circumference_m": _to_float(row.get("dirt_circumference_m")),
        }
    return out


def load_course_turn_type() -> dict:
    """(venue_code, surface, distance_m) -> turn_type("単一"/"内回り"/"外回り"/"内外"/"unknown")。

    2026-09-06追加(改善計画1-D)。`data/jra_course_turn_type.csv`
    (`build_course_turn_type_table.py`生成、新規スクレイピング不要の静的表)を読む。
    """
    path = DATA_ROOT / "jra_course_turn_type.csv"
    if not path.exists():
        return {}
    d = pd.read_csv(path, dtype={"venue_code": str})
    out = {}
    for _, row in d.iterrows():
        key = (str(row["venue_code"]).zfill(2), row["surface"], int(row["distance_m"]))
        out[key] = row["turn_type"]
    return out


class BabaLookup:
    """data/jra_baba/{year}/{venue_code}_{kai}.csv を遅延ロードし、開催日で1行引く。

    2025年以降の開催回しか存在しない(2024年以前は別PDFレイアウトで未対応)、かつ
    進行中の開催回はPDF自体が未公開のため、引けない日付は None を返すのが正常系。
    """

    def __init__(self):
        self._cache = {}

    def _load(self, year: str, venue_code: str, kai: str):
        key = (year, venue_code, kai)
        if key not in self._cache:
            path = DATA_ROOT / "jra_baba" / year / f"{venue_code}_{kai}.csv"
            if path.exists():
                try:
                    self._cache[key] = pd.read_csv(path, dtype=str)
                except Exception:
                    self._cache[key] = None
            else:
                self._cache[key] = None
        return self._cache[key]

    def get(self, race_id: str, kaisai_date: str):
        if not isinstance(race_id, str) or len(race_id) != 12:
            return None
        d = self._load(race_id[:4], race_id[4:6], race_id[6:8])
        if d is None or d.empty:
            return None
        iso = f"{kaisai_date[:4]}-{kaisai_date[4:6]}-{kaisai_date[6:8]}"
        hit = d[d["date"].astype(str) == iso]
        if hit.empty:
            return None
        return hit.iloc[0]


class RaceEntryIndex:
    """(race_id, horse_id) -> {jockey_id, trainer_id, owner_id} を data/race_results から作る。

    騎手・調教師・馬主はいずれも発走前に確定している情報なので、結果CSVから引いても
    リークにはならない(着順・タイム等の結果列は一切使わない)。
    """

    def __init__(self, results: pd.DataFrame):
        # 2026-09-13追記(JRAデータ資産棚卸し監査B-4): odds_final(確定オッズ)も同じ索引に
        # 相乗りさせる。こちらは「発走前に確定済み」ではなく事後値なので、compute_for_race()
        # 側では従来のbias_win_odds(取得時点値)へのフォールバック込みで使う
        # (jra_pending_factor_scoring.py はこのContext/RaceEntryIndexを一切使わないため
        # pending日の挙動には影響しない)。
        cols = ["race_id", "horse_id", "jockey_id", "trainer_id", "owner_id", "odds_final"]
        sub = results[[c for c in cols if c in results.columns]].copy()
        self.map = {}
        for row in sub.itertuples(index=False):
            key = (str(row.race_id), str(row.horse_id))
            self.map[key] = {
                "jockey_id": _norm_id(getattr(row, "jockey_id", None)),
                "trainer_id": _norm_id(getattr(row, "trainer_id", None)),
                "owner_id": _norm_id(getattr(row, "owner_id", None)),
                "odds_final": _to_float(getattr(row, "odds_final", None)),
            }

    def get(self, race_id, horse_id):
        return self.map.get((str(race_id), str(horse_id)))


def _norm_id(v):
    if v is None or pd.isna(v):
        return None
    s = str(v).strip()
    return s or None


class OwnerPriorStats:
    """馬主ごとの「当該レース日より前」の勝率(%)。min_n 未満は None(対象外)。"""

    def __init__(self, results: pd.DataFrame, min_n: int = 30):
        self.min_n = min_n
        sub = results[["owner_id", "race_date", "finish_num"]].dropna(subset=["owner_id"]).copy()
        sub["owner_id"] = sub["owner_id"].map(_norm_id)
        sub = sub.dropna(subset=["owner_id"]).sort_values("race_date", kind="stable")
        self.map = {}
        for oid, g in sub.groupby("owner_id", sort=False):
            dates = g["race_date"].astype(str).to_numpy()
            fin = pd.to_numeric(g["finish_num"], errors="coerce").to_numpy()
            self.map[oid] = (dates, np.cumsum(fin == 1))

    def win_rate(self, owner_id, kaisai_date: str):
        if owner_id is None:
            return None
        e = self.map.get(owner_id)
        if e is None:
            return None
        dates, cum_wins = e
        iso = f"{kaisai_date[:4]}-{kaisai_date[4:6]}-{kaisai_date[6:8]}"
        k = int(np.searchsorted(dates, iso, side="left"))
        if k < self.min_n:
            return None
        return round(100.0 * float(cum_wins[k - 1]) / k, 1)


class Context:
    """v3の全ファクターが使う参照データをまとめて1度だけ構築する。"""

    def __init__(self, results: pd.DataFrame):
        self.jockeys = load_person_profiles("jockey")
        self.trainers = load_person_profiles("trainer")
        self.horses = load_horse_profiles()
        self.course = load_course_master()
        self.turn_type = load_course_turn_type()
        self.baba = BabaLookup()
        self.entries = RaceEntryIndex(results)
        self.owner_stats = OwnerPriorStats(results)


# --------------------------------------------------------------------------
# バンド分け(すべて実データ分位点を見て決めた固定閾値、コメントに根拠を残す)
# --------------------------------------------------------------------------
def _band(v, cuts, labels):
    """v < cuts[0] -> labels[0], cuts[0] <= v < cuts[1] -> labels[1], ... """
    if v is None or pd.isna(v):
        return None
    for cut, lab in zip(cuts, labels):
        if v < cut:
            return lab
    return labels[-1]


def _career_wins_band(n):
    # JRA現行の減量規定(通算30勝以下=▲3kg / 31〜50勝=△2kg / 51〜100勝=☆1kg)に対応する
    # 区切りをそのまま使う。ただし通算勝利数は「現在値スナップショット」であり、
    # レース時点の値ではない(境界をまたいだ騎手は1つ上の区分に見える可能性がある)。
    return _band(n, [31, 51, 101, 501], ["le30", "w31_50", "w51_100", "w101_500", "over500"])


def _transport_flag(trainer_area, racecourse):
    if trainer_area not in ("美浦", "栗東"):
        return "unknown"
    if racecourse in _KANTO_VENUES:
        return "home" if trainer_area == "美浦" else "cross_region"
    if racecourse in _KANSAI_VENUES:
        return "home" if trainer_area == "栗東" else "cross_region"
    return "local_away"  # 札幌・函館・福島・新潟・小倉(両エリアとも輸送)


def _breeder_group(name):
    if not name:
        return "unknown"
    if name == "ノーザンファーム":
        return "northern_farm"
    if name == "社台ファーム":
        return "shadai_farm"
    if name in _SHADAI_GROUP_OTHER:
        return "shadai_group_other"
    return "other"


def _cushion_band(v):
    # 2026年7〜8月の実測分布 min6.2 / Q1 7.8 / 中央 9.6 / Q3 9.8 / max 10.8。
    # 充足度マップの勝率検証(8.5以下=硬め)と同じ区切りを踏襲する。
    return _band(v, [8.5, 9.5], ["hard", "standard", "soft"])


def _moisture_band(v, surface):
    # 芝と砂で水分量のスケールが1桁違う(実測中央値: 芝13.0% / ダート3.2%)ため、
    # サーフェス別の閾値で「乾き気味 / 標準 / 湿った」に揃える。
    if v is None:
        return None
    if surface == "芝":
        return _band(v, [12.0, 15.0], ["dry", "standard", "wet"])
    return _band(v, [4.0, 8.0], ["dry", "standard", "wet"])


def _odds_band(v):
    if v is None or pd.isna(v):
        return None
    if v <= 1.5:
        return "ultra_fav"
    if v < 4.0:
        return "fav"
    if v < 10.0:
        return "mid"
    if v < 30.0:
        return "chuana"      # 充足度マップの「単勝10〜29.9倍の中穴ゾーン」
    return "longshot"


def _interval_band(days):
    if days is None or pd.isna(days):
        return "no_history"
    if days <= 7:
        return "renchaku"     # 連闘
    if days <= 28:
        return "w1_4"         # 中1〜3週
    if days <= 56:
        return "w5_8"         # 中4〜7週
    if days <= 180:
        return "m2_6"
    return "over6m"


def _weight_carried_band(v):
    if v is None or pd.isna(v):
        return None
    if v <= 52.0:
        return "le52"
    if v <= 54.5:
        return "w52_54"
    if v <= 56.5:
        return "w55_56"
    return "ge57"


def _horse_weight_band(v):
    return _band(v, [440, 480, 520], ["small", "medium", "large", "xlarge"])


def _horse_weight_diff_band(v):
    if v is None or pd.isna(v):
        return None
    if v <= -10:
        return "minus10"
    if v < 0:
        return "minus_small"
    if v == 0:
        return "zero"
    if v < 10:
        return "plus_small"
    return "plus10"


def _past1_finish_band(v):
    if v is None or pd.isna(v):
        return "no_history"
    v = int(v)
    if v == 1:
        return "win"
    if v <= 3:
        return "f2_3"
    if v <= 5:
        return "f4_5"
    if v <= 9:
        return "f6_9"
    return "f10plus"


def _training_course_type(s):
    """training_course("栗坂"/"美Ｗ"/"ＣＷ"/"札ダ"/"美Ｐ"/"函芝" 等、"一番時計"接尾辞あり)を
    調教コースの種別へ分類する。全角/半角どちらの英字表記も実データに出現する。"""
    if pd.isna(s):
        return None
    t = str(s).strip()
    if not t:
        return None
    if "坂" in t:
        return "sakaro"       # 坂路
    if "Ｐ" in t or "P" in t:
        return "poly"         # ポリトラック・DP
    if "Ｗ" in t or "W" in t:
        return "wood"         # ウッドチップ(美W・函W・CW)
    if "芝" in t:
        return "turf"
    if "ダ" in t:
        return "dirt"
    return "other"


def _training_best_time_flag(s):
    """training_courseの末尾に「一番時計」が付くか(その日そのコースの最速時計)。"""
    if pd.isna(s):
        return None
    return "best_time" if "一番時計" in str(s) else "normal"


_SEX_AGE_RE = re.compile(r"^\s*([牡牝セせん]+)\s*(\d+)")


def _parse_sex_age(s):
    if pd.isna(s):
        return None, None
    m = _SEX_AGE_RE.match(str(s))
    if not m:
        return None, None
    sex = m.group(1)
    sex = "セ" if sex.startswith("セ") or sex.startswith("せ") else sex[0]
    return sex, int(m.group(2))


_HW_RE = re.compile(r"^\s*(\d+)\s*(?:\(([-+]?\d+)\))?")


def _parse_horse_weight(s):
    """"418(-2)" -> (418, -2)。増減が無い(初出走等)場合は diff=None。"""
    if pd.isna(s):
        return None, None
    m = _HW_RE.match(str(s))
    if not m:
        return None, None
    w = int(m.group(1))
    d = int(m.group(2)) if m.group(2) is not None else None
    return w, d


# --------------------------------------------------------------------------
# レース1本ぶんの計算
# --------------------------------------------------------------------------
def compute_for_race(df: pd.DataFrame, race_id: str, kaisai_date: str, racecourse: str,
                     surface, ctx: Context, distance_m=None) -> dict:
    """dfのindexに揃えたSeries dictを返す。全列とも欠損は None(数値)またはセンチネル
    文字列("unknown"等)で表し、jra_factor_registry の options に登録しないことで
    UI上「常に非該当(絞り込み対象外)」として扱われるようにする(v1/v2と同じ規約)。

    `distance_m`(2026-09-06追加、改善計画1-D): course_turn_type(内回り/外回り)の
    lookupキーとして使う。呼び出し側(build_factor_dataset.py)がjra_lap33_signals経由の
    race_metaから発走前に判明済みの距離を渡す想定(リークにはならない)。
    """
    idx = df.index
    horse_ids = JS._col(df, "horse_id")

    venue_code = race_id[4:6] if isinstance(race_id, str) and len(race_id) == 12 else None
    cm = ctx.course.get(venue_code) if venue_code else None
    baba = ctx.baba.get(race_id, kaisai_date)

    # --- レース単位(全馬同値) -------------------------------------------------
    course_turn = cm["turn_direction"] if cm else None
    course_turn_type = None
    if venue_code and surface and distance_m:
        # surfaceの表記揺れ対策: jra_lap33_signals.load_race_surface_distance()由来の
        # surfaceは「ダ」のように省略される場合がある(2026-09-06発覚)。
        # jra_course_turn_type.csv側は「芝」「ダート」の正式表記でキー化しているため正規化する。
        surface_key = "芝" if surface == "芝" else "ダート"
        try:
            course_turn_type = ctx.turn_type.get((venue_code, surface_key, int(distance_m)))
        except (TypeError, ValueError):
            course_turn_type = None
    if course_turn_type is None:
        course_turn_type = "unknown"
    if cm:
        pre = "turf" if surface == "芝" else "dirt"
        straight = cm.get(f"{pre}_straight_m")
        elevation = cm.get(f"{pre}_elevation_m")
        circumference = cm.get(f"{pre}_circumference_m")
    else:
        straight = elevation = circumference = None
    # 実測レンジ(芝): 直線262〜659m / 高低差0.7〜5.3m / 周長1600〜2223m。
    course_straight_band = _band(straight, [300, 450], ["short", "mid", "long"])
    course_elevation_band = _band(elevation, [2.0, 3.0], ["small", "mid", "large"])
    course_size_band = _band(circumference, [1700, 1900], ["small", "mid", "large"])

    if baba is not None:
        cushion_band = _cushion_band(_to_float(baba.get("cushion_value")))
        m_col = "moisture_turf_goal_pct" if surface == "芝" else "moisture_dirt_goal_pct"
        moisture_band = _moisture_band(_to_float(baba.get(m_col)), surface)
        variant = _clean(baba.get("turf_course_variant"))
    else:
        cushion_band = moisture_band = variant = None

    pace_col = JS._col(df, "race_pace_label")
    pace = None
    if pace_col is not None and len(pace_col):
        v = pace_col.iloc[0]
        pace = str(v).strip() if pd.notna(v) else None

    # --- 馬単位 ---------------------------------------------------------------
    kaisai_dt = pd.to_datetime(kaisai_date, format="%Y%m%d", errors="coerce")
    past1_date = JS._col(df, "past1_date").map(_parse_dot_date)
    sex_age = JS._col(df, "bias_sex_age")
    weight_carried = JS._num(JS._col(df, "bias_weight_carried"))
    horse_weight_raw = JS._col(df, "bias_horse_weight")
    # 2026-09-13追記: race_results.odds_final(確定値)を優先し、無い馬(出走取消等・
    # ctx.entriesにrace_id自体が無いpending日相当)だけbias_win_odds(取得時点値)へ
    # フォールバックする(EV計算と同じ方針、build_factor_dataset.py側のnotes参照)。
    win_odds_bias = JS._num(JS._col(df, "bias_win_odds"))
    win_odds_final = pd.Series(
        [(ctx.entries.get(race_id, hid) or {}).get("odds_final") for hid in horse_ids],
        index=idx, dtype="float64",
    )
    win_odds = win_odds_final.where(win_odds_final.notna(), win_odds_bias)
    corner4 = JS._num(JS._col(df, "corner4_rank"))
    past1_finish = JS._num(JS._col(df, "past1_finish"))
    training_course = JS._col(df, "training_course")
    training_track = JS._col(df, "training_track_condition")
    # ca_jockey/ca_trainer の win_rate は "6%" のような文字列(_numでは全件NaNになる)。
    ca_jockey_wr = JS._pct(JS._col(df, "ca_jockey_win_rate"))
    ca_trainer_wr = JS._pct(JS._col(df, "ca_trainer_win_rate"))

    cols = {k: [] for k in (
        "jockey_lead_rank", "jockey_area", "jockey_affiliation_type", "jockey_career_wins_band",
        "trainer_lead_rank", "trainer_area", "transport_flag",
        "owner_prior_win_rate", "breeder_group",
        "horse_sex", "horse_age", "weight_carried_band",
        "horse_weight_band", "horse_weight_diff_band", "odds_band", "interval_band",
        "corner4_expected_rank", "past1_finish_band",
        "training_course_type", "training_track_condition", "training_best_time_flag",
        "ca_jockey_win_rate", "ca_trainer_win_rate",
    )}

    for i in idx:
        hid = str(horse_ids.at[i]) if i in horse_ids.index and pd.notna(horse_ids.at[i]) else None
        entry = ctx.entries.get(race_id, hid) if hid else None

        jp = ctx.jockeys.get(entry["jockey_id"]) if entry and entry["jockey_id"] else None
        cols["jockey_lead_rank"].append(jp["season_rank"] if jp else None)
        cols["jockey_area"].append((jp["area"] if jp and jp["area"] else "unknown"))
        if jp is None:
            cols["jockey_affiliation_type"].append("unknown")
        elif jp["affiliation_type"] is None:
            # 所属形態が空欄 = 厩舎所属(所属厩舎名が入る欄が空なのは実データ上フリー以外の扱い)。
            cols["jockey_affiliation_type"].append("unknown")
        elif jp["affiliation_type"].startswith("フリー"):
            cols["jockey_affiliation_type"].append("free")
        else:
            cols["jockey_affiliation_type"].append("stable")
        cols["jockey_career_wins_band"].append(
            _career_wins_band(jp["career_wins"]) if jp else None)

        tp = ctx.trainers.get(entry["trainer_id"]) if entry and entry["trainer_id"] else None
        cols["trainer_lead_rank"].append(tp["season_rank"] if tp else None)
        t_area = tp["area"] if tp and tp["area"] else None
        cols["trainer_area"].append(t_area or "unknown")
        cols["transport_flag"].append(_transport_flag(t_area, racecourse))

        oid = entry["owner_id"] if entry else None
        cols["owner_prior_win_rate"].append(ctx.owner_stats.win_rate(oid, kaisai_date))

        hp = ctx.horses.get(hid) if hid else None
        cols["breeder_group"].append(_breeder_group(hp["breeder"] if hp else None))

        sex, age = _parse_sex_age(sex_age.at[i] if i in sex_age.index else np.nan)
        cols["horse_sex"].append(sex or "unknown")
        cols["horse_age"].append(age)

        wc = weight_carried.at[i] if i in weight_carried.index else np.nan
        cols["weight_carried_band"].append(_weight_carried_band(None if pd.isna(wc) else float(wc)))

        hw, hd = _parse_horse_weight(horse_weight_raw.at[i] if i in horse_weight_raw.index else np.nan)
        cols["horse_weight_band"].append(_horse_weight_band(hw))
        cols["horse_weight_diff_band"].append(_horse_weight_diff_band(hd))

        od = win_odds.at[i] if i in win_odds.index else np.nan
        cols["odds_band"].append(_odds_band(None if pd.isna(od) else float(od)))

        p1 = past1_date.at[i] if i in past1_date.index else pd.NaT
        days = None if (pd.isna(p1) or pd.isna(kaisai_dt)) else (kaisai_dt - p1).days
        cols["interval_band"].append(_interval_band(days))

        c4 = corner4.at[i] if i in corner4.index else np.nan
        cols["corner4_expected_rank"].append(None if pd.isna(c4) else int(c4))

        cols["past1_finish_band"].append(
            _past1_finish_band(past1_finish.at[i] if i in past1_finish.index else np.nan))

        tcv = training_course.at[i] if i in training_course.index else np.nan
        cols["training_course_type"].append(_training_course_type(tcv))
        cols["training_best_time_flag"].append(_training_best_time_flag(tcv))
        ttv = training_track.at[i] if i in training_track.index else np.nan
        cols["training_track_condition"].append(
            None if pd.isna(ttv) or not str(ttv).strip() else str(ttv).strip())

        jw = ca_jockey_wr.at[i] if i in ca_jockey_wr.index else np.nan
        cols["ca_jockey_win_rate"].append(None if pd.isna(jw) else float(jw))
        tw = ca_trainer_wr.at[i] if i in ca_trainer_wr.index else np.nan
        cols["ca_trainer_win_rate"].append(None if pd.isna(tw) else float(tw))

    out = {k: pd.Series(v, index=idx, dtype=object) for k, v in cols.items()}
    # レース単位の値は全馬同値のスカラーとして返す(build_factor_dataset側でそのまま入れる)
    out["_race_level"] = {
        "surface": surface if surface else None,
        "course_turn": course_turn,
        "course_turn_type": course_turn_type,
        "course_straight_band": course_straight_band,
        "course_elevation_band": course_elevation_band,
        "course_size_band": course_size_band,
        "cushion_band": cushion_band,
        "moisture_band": moisture_band,
        "turf_course_variant": variant,
        "race_pace_label": pace,
    }
    return out
