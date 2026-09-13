# -*- coding: utf-8 -*-
"""ユーザー依頼(2026-09-01、「予想ファクター充足度マップ」の「計算のみで対応可」項目)による
追加ファクター v1。既存の本番スコア(jra_model_scoring.py)・レーダー8カテゴリ
(jra_radar_categories.py)は無改造、このモジュールが完全に自己完結で計算する追加専用シグナル。
build_factor_dataset.py から呼ばれ、ファクター検証データベース(factor_database.json)の
「新規候補ファクター v1」タイア(jra_factor_registry.py)としてのみ使われる
(predict_pattern29.py等の本番予想には一切反映しない)。

データ源: jra_history.JH.load_results()(2016〜2026年、2026-09-13〜。以前は2024〜2026年。
data/race_results/{year}/*.csv全件)を
horse_id単位で日付昇順に束ね、各レースのkaisai_dateより前の走のみを参照する
(race_date文字列(YYYYMMDD)の厳密不等号`<`比較。JH.HorseHistoryIndexと同じ前提=同日2走の
重複が無いことは検証済みなのでリークしない)。

コース属性の静的テーブル(回り/坂/芝種/ローカルか否か)はJRA10場のコース設計に基づく一般的な
分類(安定した既知の事実、出典検証不要)。
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent

sys.path.insert(0, str(LIB_DIR))
import jra_history as JH  # noqa: E402
import jra_signals as JS  # noqa: E402

# --- コース静的属性テーブル(JRA10場、変わりにくい一般的分類) ---
LEFT_HANDED = {"東京", "中京", "新潟"}                      # それ以外(7場)は右回り
SLOPE_COURSES = {"中山", "阪神", "中京"}                     # ゴール前の急坂で知られる3場
LOCAL_COURSES = {"札幌", "函館", "福島", "新潟", "小倉"}       # ローカル5場(それ以外=中央5場)
KANTO_TURF_COURSES = {"札幌", "函館"}                        # 洋芝2場(それ以外=野芝)

_HIST_COLS = ["race_date", "finish_num", "distance_m", "surface", "racecourse",
              "race_name", "trainer_name"]

_TRAINER_BRACKET_RE = re.compile(r"^\[[^\]]*\]")


def _strip_trainer_bracket(name):
    """race_results.trainer_nameの先頭"[東]"/"[西]"を除去する。newspaperのbias_trainerは
    括弧が付かず、かつ姓のみ(苗字が一般的な場合は姓のみ、まれな苗字はフルネーム)で表記される
    ことが実データで確認済み(例: 括弧除去後"岩戸孝樹" にbias_trainer"岩戸"が前方一致)。
    このため同一人物判定は完全一致ではなく前方一致(startswith)で行う。"""
    if pd.isna(name):
        return name
    return _TRAINER_BRACKET_RE.sub("", str(name)).strip()


def _course_turn(racecourse) -> str:
    return "left" if racecourse in LEFT_HANDED else "right"


def build_history_index(results: pd.DataFrame | None = None) -> dict:
    """horse_id(str) -> race_date昇順のDataFrame(distance_m/surface等の拡張列付き)。
    jra_history.JH.load_results()を再利用(過去に検証済みの派生列計算をそのまま使う)。
    `results`を渡さない場合は自前でロードするが、呼び出し側が既に
    `jra_lap33_signals.build_history_index()`等でJH.load_results()を実行済みの場合は
    その結果(HorseHistoryIndexの`_results`)を渡すことで二重ロードを避けられる。"""
    if results is None:
        results = JH.load_results()
    if results.empty:
        return {}
    results = results.copy()
    results["distance_m"] = pd.to_numeric(results["distance_m"], errors="coerce")
    # race_resultsのrace_dateは"2026-08-30"(ハイフン区切り)だが、build_factor_dataset.py側の
    # kaisai_dateは"20260830"(ハイフン無し)。ハイフン("-"=0x2D)は数字(0x30-0x39)よりASCII上で
    # 小さいため、正規化せずに文字列比較すると"2026-08-30" < "20260830"が常にTrueになり、
    # 未来の走まで「過去走」として取り込む重大なリークが発生する(実測で確認済みの致命的バグ、
    # 2026-09-02修正)。ここで両者とも"YYYYMMDD"形式に統一する。
    results["race_date"] = results["race_date"].astype(str).str.replace("-", "", regex=False)
    idx = {}
    for hid, g in results.groupby("horse_id"):
        g = g.sort_values("race_date")
        idx[str(hid)] = g[_HIST_COLS].reset_index(drop=True)
    return idx


def _hist_before(hist_idx: dict, horse_id, kaisai_date: str) -> pd.DataFrame:
    g = hist_idx.get(str(horse_id))
    if g is None or g.empty:
        return pd.DataFrame(columns=_HIST_COLS)
    return g[g["race_date"] < kaisai_date]


def _place_rate(sub: pd.DataFrame):
    if sub.empty:
        return None
    n = len(sub)
    k = int((sub["finish_num"] <= 3).sum())
    return round(100.0 * k / n, 1)


def kaisai_late_flag(race_id: str):
    """race_id構造(YYYY[0:4]+venue2[4:6]+kai2[6:8]+nichime2[8:10]+race2[10:12]、計12桁)から
    「日目」(nichime、[8:10])を取り出し、7日目以降(開催後半、馬場が荒れやすいとされる時期)かを
    判定する。回(kai、[6:8])と取り違えないよう注意(実測で確認済みの対応関係)。"""
    if not isinstance(race_id, str) or len(race_id) != 12:
        return None
    try:
        return int(race_id[8:10]) >= 7
    except ValueError:
        return None


def compute_for_race(df: pd.DataFrame, racecourse: str, distance_m, surface: str,
                      kaisai_date: str, hist_idx: dict) -> dict:
    """レース1本ぶん(馬N頭)の追加ファクターをdfのindexに揃えたSeries dictで返す。"""
    horse_ids = JS._col(df, "horse_id")
    bias_trainer = JS._col(df, "bias_trainer")
    past1_jockey = JS._col(df, "past1_jockey")
    past2_jockey = JS._col(df, "past2_jockey") if "past2_jockey" in df.columns else pd.Series(
        [np.nan] * len(df), index=df.index)
    bias_jockey = JS._col(df, "bias_jockey")

    turn_today = _course_turn(racecourse)
    is_local_today = racecourse in LOCAL_COURSES
    is_slope_today = racecourse in SLOPE_COURSES
    is_kanto_turf_today = racecourse in KANTO_TURF_COURSES
    dist_today = float(distance_m) if pd.notna(distance_m) else None

    trainer_counts = bias_trainer.value_counts()

    cols = {
        "career_place_rate": [], "distance_band_place_rate": [], "turn_apt_place_rate": [],
        "course_size_apt_place_rate": [], "slope_apt_place_rate": [], "turf_type_apt_place_rate": [],
        "grade_best": [], "first_time_flag": [], "jockey_switch": [], "trainer_change": [],
        "stable_multi_entry": [],
    }

    for i in df.index:
        hid = horse_ids.at[i]
        hist = _hist_before(hist_idx, hid, kaisai_date)
        n_runs = len(hist)

        cols["career_place_rate"].append(_place_rate(hist))

        if dist_today is not None and n_runs:
            band = hist[(hist["distance_m"] - dist_today).abs() <= 200]
            cols["distance_band_place_rate"].append(_place_rate(band))
        else:
            cols["distance_band_place_rate"].append(None)

        if n_runs:
            turn_sub = hist[hist["racecourse"].apply(_course_turn) == turn_today]
            cols["turn_apt_place_rate"].append(_place_rate(turn_sub))
            size_sub = hist[hist["racecourse"].isin(LOCAL_COURSES) == is_local_today]
            cols["course_size_apt_place_rate"].append(_place_rate(size_sub))
        else:
            cols["turn_apt_place_rate"].append(None)
            cols["course_size_apt_place_rate"].append(None)

        if is_slope_today and n_runs:
            cols["slope_apt_place_rate"].append(_place_rate(hist[hist["racecourse"].isin(SLOPE_COURSES)]))
        else:
            cols["slope_apt_place_rate"].append(None)

        if is_kanto_turf_today and n_runs:
            cols["turf_type_apt_place_rate"].append(_place_rate(hist[hist["racecourse"].isin(KANTO_TURF_COURSES)]))
        else:
            cols["turf_type_apt_place_rate"].append(None)

        # 重賞最高着順(race_nameをCLASS_ORDINAL_V3EXTで評価、5=G3/重賞,6=G2,7=G1)
        if n_runs == 0:
            grade = "no_history"
        else:
            levels = hist["race_name"].map(lambda t: JS._class_ordinal(t, JS.CLASS_ORDINAL_V3EXT))
            graded_mask = levels >= 5
            if not graded_mask.any():
                grade = "no_graded_starts"
            else:
                top3_graded = graded_mask & (hist["finish_num"] <= 3)
                if not top3_graded.any():
                    grade = "graded_no_top3"
                else:
                    best_level = int(levels[top3_graded].max())
                    grade = {7: "g1_top3", 6: "g2_top3", 5: "g3_top3"}.get(best_level, "graded_no_top3")
        cols["grade_best"].append(grade)

        # 初サーフェス・初距離帯フラグ(このモデルが参照できる2016〜2026年履歴の範囲内での「初」、
        # 2026-09-13〜。以前は2024〜2026年)
        if n_runs == 0:
            ft = "both_first"
        else:
            first_surface = pd.notna(surface) and not (hist["surface"] == surface).any()
            first_distance = dist_today is not None and not ((hist["distance_m"] - dist_today).abs() <= 200).any()
            if first_surface and first_distance:
                ft = "both_first"
            elif first_surface:
                ft = "surface_first"
            elif first_distance:
                ft = "distance_first"
            else:
                ft = "experienced"
        cols["first_time_flag"].append(ft)

        # 乗り替わり・継続騎乗(newspaperのpast1/2_jockey列とbias_jockey列を使用)。
        # past1/2_jockeyはフルネーム(例: "伴啓太")、bias_jockeyは姓のみ等の短縮表記
        # (例: "伴")で書式が異なることが実測で判明したため、完全一致ではなく前方一致で
        # 同一人物判定する(trainer_changeと同じ修正、完全一致だと継続騎乗が過小検出される
        # バグになる。ただし同姓の別人を誤って「継続」と判定するリスクは残る)。
        pj = past1_jockey.at[i] if i in past1_jockey.index else np.nan
        cj = bias_jockey.at[i] if i in bias_jockey.index else np.nan
        pj_s, cj_s = (str(pj).strip() if pd.notna(pj) else ""), (str(cj).strip() if pd.notna(cj) else "")
        if not pj_s or not cj_s:
            jw = "unknown"
        elif pj_s.startswith(cj_s) or cj_s.startswith(pj_s):
            p2j = past2_jockey.at[i] if i in past2_jockey.index else np.nan
            p2j_s = str(p2j).strip() if pd.notna(p2j) else ""
            same2 = bool(p2j_s) and (p2j_s.startswith(cj_s) or cj_s.startswith(p2j_s))
            jw = "continuous2plus" if same2 else "continuous1"
        else:
            jw = "switch"
        cols["jockey_switch"].append(jw)

        # 転厩初戦(直近の履歴上の調教師名 vs 今回のbias_trainer)。race_results.trainer_nameは
        # "[東]岩戸孝樹"のように括弧+フルネーム、newspaper.bias_trainerは括弧無し・姓のみ
        # (または稀少姓はフルネーム)という異なる表記なので、括弧除去後は前方一致で判定する
        # (完全一致にすると表記差だけで常に「転厩」判定になる致命的バグになるため、実測で修正済み)。
        if n_runs == 0:
            tc = "unknown"
        else:
            last_trainer = _strip_trainer_bracket(hist.iloc[-1]["trainer_name"])
            ct = bias_trainer.at[i] if i in bias_trainer.index else np.nan
            if pd.isna(last_trainer) or pd.isna(ct) or not str(ct).strip():
                tc = "unknown"
            else:
                ct_s = str(ct).strip()
                tc = "same" if (last_trainer.startswith(ct_s) or ct_s.startswith(last_trainer)) else "changed"
        cols["trainer_change"].append(tc)

        # 同一レース内・同厩舎多頭出し(履歴不要、このレースのbias_trainerだけで判定)
        t = bias_trainer.at[i] if i in bias_trainer.index else np.nan
        cols["stable_multi_entry"].append("multi" if (pd.notna(t) and trainer_counts.get(t, 0) > 1) else "single")

    # dtype=objectを明示: 数値とNoneが混在するリストをpd.Seriesにそのまま渡すと、pandasが
    # float64列へ暗黙に型強制しNoneがnp.nanへ化ける(build_factor_dataset.py側の_mark_valの
    # 注記と同じ既知の落とし穴)。np.nanのままjson.dumpsするとJSON非準拠のNaNトークンを
    # 出力し、ブラウザのJSON.parseがクラッシュする(実測: 2026-09-02、修正前は2,903件混入)。
    return {k: pd.Series(v, index=df.index, dtype=object) for k, v in cols.items()}
