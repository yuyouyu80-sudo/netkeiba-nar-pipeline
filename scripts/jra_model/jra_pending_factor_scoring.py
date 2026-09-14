# -*- coding: utf-8 -*-
"""結果未確定(pending)日の通常戦レースに対して、ファクター検証データベースと
同じロジックでpred_rank(BOX5/4/3)・レーダー8カテゴリ順位を再現する。

`build_factor_dataset.py`は`jra_dataset_wide.load()`(race_results/payoutsが両方
存在する日付のみ)を前提にしており、結果未確定のpending日は対象外。本スクリプトは
その制約を回避し、newspaper CSVだけから同じ計算(`jra_model_scoring.py`/
`jra_radar_categories.py`、いずれも無改造)を行う。priorsは既存の母集団
(jra_dataset_wide/jra_dataset)からそのまま流用する(pending日1日分だけでpriorsを
作り直すと標本が薄すぎるため)。

出力: data/jra_pipeline/pending_factor_{date}.json
  {race_id: {race_type, horses: [{umaban, waku, pred_rank, pred_rank_box4,
                                   pred_rank_box3, radar_rank_pedigree, radar_rank_jt,
                                   radar_rank_mark, radar_rank_ability, horse_age,
                                   weight_carried_band, odds_band, course_size_band,
                                   course_elevation_band, race_pace_label, jockey_area,
                                   jockey_lead_rank}]}}
新馬戦・未勝利戦・障害戦(race_type != "normal")はBOX4/BOX3モデルが存在しないため
pred_rank_box4/box3=null、レーダー系(pedigree/jt/mark)は通常戦と同じロジックで計算する
(priorsだけ通常戦と別)が、当面の利用目的(通常戦限定フィルタ)ではnormal以外は
使わない想定。

2026-09-05追記(ユーザー依頼「予想印×年齢×オッズ帯 単勝適合レース帳」パターンの組み込み)で
`jra_candidate_factors_v3.py`のv3ファクター7項目を追加。ただしv3本体の`compute_for_race()`
(`RaceEntryIndex`経由でdata/race_results由来のjockey_id/trainer_idを結合)はpending日の
race_idがrace_resultsにまだ存在しないため使えない。2026-09-02以降に取得したnewspaper CSVには
`bias_jockey_id`が直接含まれるため(bias_parser.py拡張済み)、本スクリプトはv3のバンド分け
関数(`_parse_sex_age`/`_weight_carried_band`/`_odds_band`/`_band`)だけを再利用し、
jockey_id自体はrace_results経由ではなくnewspaper CSVの`bias_jockey_id`から直接引く
(発走前に確定済みの情報なのでリークではない)。course_size_bandは同じ理由でrace_meta
(race_results由来、pending日は未ヒット)ではなく`race_names_{date}.csv`のsurface/distance_m
列(predict_pattern29.py出力、発走前に判明済み)から求めた周長で判定する。

2026-09-06追記(ユーザー依頼「予想力×高低差×ペース×斤量 馬連頭軸レース帳」パターンの
組み込み)で`radar_rank_ability`(能力・調教評価順位、RC.build_axis_matrix()が既に計算済みの
軸値を血統適性等と同じ手順でランク化するだけ)・`course_elevation_band`(コース高低差、
jra_course_master.csvの静的値をv3と同じ閾値[2.0, 3.0]でバンド化)・`race_pace_label`
(想定ペース、newspaper CSVの`race_pace_label`列、netkeiba公式AI予想ペース区分で発走前に
公開済み)を追加。いずれも発走前に確定済みの情報のためリークではない。

2026-09-12追記(ユーザー依頼「外し/複外/W外し消し材料マークをpending日にも必ず表示してほしい」):
「低的中率・低回収率 候補条件200」由来の3台帳(単勝10%未満/複勝25%未満/ワイド10%未満、計169atom)が
参照する34種類のsourceフィールド(jra_factor_registry.FACTOR_GROUPS基準)のうち、本スクリプトが
従来出力していなかった25項目を追加する。**追加分はいずれも「発走前に確定済みの情報」または
「horse_idの過去走(=今回のkaisai_dateより前)だけを使う履歴集計」のみで構成し、race_results
(今回のレース結果)には一切依存しない**:

  - `jra_candidate_factors_v1.compute_for_race()`(CF1)をそのまま呼ぶ: grade_best・
    jockey_switch・career_place_rate・distance_band_place_rate・turn_apt_place_rate・
    course_size_apt_place_rate。CF1はdf(newspaper CSV)+コース属性+
    `JH.load_results()`から作るhorse_id別履歴インデックス(race_date < kaisai_dateで
    厳密に絞り込み済み)だけで完結しており、race_results呼び出し元がbuild_factor_dataset.py
    (検証済み日)かpending scoringか(このスクリプト)かを区別しない設計のため無改造で使える。
  - `jra_candidate_factors_v2.compute_for_race()`(CF2)も同様にそのまま呼ぶ:
    class_form_place_rate・main_jockey_return_flag。
  - v3(`jra_candidate_factors_v3.py`)の一部関数は個別に再利用する(v3本体の
    `compute_for_race()`自体は`RaceEntryIndex`経由でdata/race_results由来のjockey_id/
    trainer_id/owner_idを引く設計のためpending日では丸ごとは使えない、既存コメント参照):
    ca_jockey_win_rate・ca_trainer_win_rate(newspaper CSVの`ca_jockey/trainer_win_rate`列を
    直接パース)、corner4_expected_rank(newspaper CSVの`corner4_rank`列=netkeiba公式
    AI展開予想、発走前公開)、past1_finish_band・training_course_type・
    training_track_condition・interval_band(いずれも過去走・調教情報のnewspaper生列から
    バンド化関数`CF3._past1_finish_band`等を呼ぶだけ)、course_straight_band
    (jra_course_master.csvの静的値、course_size_band/course_elevation_bandと同じ`cm`を
    再利用)。trainer_lead_rank・transport_flag・jockey_career_wins_bandは、
    jockey_lead_rankと同じ手法(newspaper CSVの`bias_trainer_id`列、2026-09-02以降の
    bias_parser.py拡張で取得済み)でtrainer_profileを直接引く。
  - flop_safety_rank: `jra_reference_panel.py`が既にpending日向けに計算している
    `flop_rank`と同じ入力(`JS.compute_signals`+`JS.combine_signals`、ALL_SIGNALS_V4等重み)
    から、build_factor_dataset.pyと同じ向き(降順、rank1=最も凡走しにくい)で計算し直す
    (別ファイルの`reference_panel_{date}.csv`を読みに行かず自己完結させるため)。
  - radar_rank_aptitude・radar_rank_form・radar_rank_style: 既に呼んでいる
    `RC.build_axis_matrix()`の戻り値`axis`が元々8カテゴリ全て持っているため、
    radar_rank_pedigree/jt/mark/abilityと同じ手順で追加抽出するだけ(新規計算なし)。
  - **owner_prior_win_rate のみ非対応のまま残る**: 馬主IDの発走前ソース
    (newspaper CSVの`bias_owner_id`相当)が現状存在しないため計算不可。該当atom
    (`owner_prior_win_rate_low`)はpending日では常に非該当になる(将来
    fetch_newspaper.py側にbias_owner_id列が追加されれば対応可能)。

2026-09-14追記(ユーザー依頼「詳細7カテゴリ(レーダー)細分化」、レーダー解像度レビュー
Opus5サブエージェント調査結果を★3件+条件付き1件採用): 「脚質・展開」→3分割
(radar_rank_style_position/style_stamina/style_corner_move)、「騎手・厩舎」→3分割
(radar_rank_jt_base/jt_change/jt_stats)、「血統適性」→2分割(radar_rank_pedigree_pure/
pedigree_training)、「近走成績・調子」からagariのみ独立(radar_rank_form_agari、
weight_trendは単勝的中率がベースラインと無差別だったため独立化見送り)の計9本を追加。
いずれもjra_radar_categories.DISPLAY_SUBCATEGORY_MAP/DISPLAY_SUBCATEGORY_SLUG
(build_factor_dataset.pyと共有する単一の定義、2箇所の手書き食い違いを防ぐ)経由で、
既に呼んでいるRC.build_axis_matrix()のsignalsを再利用するだけ(新規のcompute_signals
呼び出しは発生しない)。既存radar_rank_pedigree/jt/mark/ability/aptitude/form/style
(7カテゴリ本体)は無変更。
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LIB_DIR))
from src.netkeiba_pipeline.storage.paths import newspaper_csv_path  # noqa: E402

import jra_candidate_factors_v1 as CF1  # noqa: E402
import jra_candidate_factors_v2 as CF2  # noqa: E402
import jra_candidate_factors_v3 as CF3  # noqa: E402
import jra_dataset as JD  # noqa: E402
import jra_dataset_wide as JDW  # noqa: E402
import jra_history as JH  # noqa: E402
import jra_lap33_signals as L33  # noqa: E402
import jra_model_scoring as MS  # noqa: E402
import jra_radar_categories as RC  # noqa: E402
import jra_signals as JS  # noqa: E402


def load_pending_races(date: str, race_names_csv: Path) -> list:
    names = pd.read_csv(race_names_csv, dtype=str, encoding="utf-8-sig")
    ped_cache = DATA_DIR / "pedigree_sire_features_cache.csv"
    ped_features = pd.read_csv(ped_cache, dtype=str, encoding="utf-8") if ped_cache.exists() else None

    races = []
    for _, row in names.iterrows():
        rtype = JDW.race_type_of(row["race_name"])
        if rtype is None:
            continue
        path = newspaper_csv_path(row["race_id"])
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
        if df.empty:
            continue
        df = JS._drop_scratched(df)
        if df.empty:
            continue
        if ped_features is not None:
            race_ped = ped_features[ped_features["race_id"] == row["race_id"]].drop(columns=["race_id"])
            df = df.merge(race_ped, on="horse_id", how="left")
        races.append({
            "race_id": row["race_id"], "kaisai_date": date,
            "racecourse": row["racecourse"], "race_name": row["race_name"],
            "race_type": rtype, "df": df,
            "surface": row.get("surface"),
            "distance_m": row.get("distance_m"),
        })
    return races


def _rank_val(rank_series: pd.Series, idx):
    v = rank_series.at[idx] if idx in rank_series.index else None
    return int(v) if pd.notna(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--race-names-csv", required=True, help="predict_pattern29.pyが出力するrace_names_{date}.csv")
    args = ap.parse_args()

    print(f"pending races読み込み中({args.date})...")
    races = load_pending_races(args.date, Path(args.race_names_csv))
    by_type = {}
    for r in races:
        by_type.setdefault(r["race_type"], []).append(r)
    print("race_type内訳:", {k: len(v) for k, v in by_type.items()})

    print("priors読み込み中(既存母集団から流用、pending日単独では作り直さない)...")
    wide = JDW.load(rebuild=False)
    wide_by_type = {}
    for r in wide["races"]:
        wide_by_type.setdefault(r["race_type"], []).append(r["df"])
    score_priors = MS.compute_priors_for_population(wide_by_type)
    weights = {
        "normal": MS.load_weights("normal"),
        "normal_box4": MS.load_weights("normal_box4"),
        "normal_box3": MS.load_weights("normal_box3"),
        "shinba": MS.load_weights("shinba"),
        "mishoubi": MS.load_weights("mishoubi"),
    }

    normal_246 = JD.load(rebuild=False)["races"]
    priors_v4 = JS.make_priors([r["df"] for r in normal_246])

    print("v3ファクター用の参照データ読み込み中(コースマスタ・騎手/調教師プロフィール)...")
    course_master = CF3.load_course_master()
    course_turn_type_table = CF3.load_course_turn_type()
    jockey_profiles = CF3.load_person_profiles("jockey")
    trainer_profiles = CF3.load_person_profiles("trainer")

    # 2026-09-12追記: 「外し/複外/W外し」消し材料マーク用のCF1/CF2ファクターを計算するため、
    # horse_id別の過去走履歴インデックスを構築する(race_dateが今回のkaisai_dateより前の
    # 走だけを使うので、今回race_idがまだrace_resultsに存在しないpending日でもリークしない)。
    print("CF1/CF2用の履歴インデックス構築中...")
    results_all = JH.load_results()
    cf1_hist_idx = CF1.build_history_index(results_all)

    print("レーダー用の履歴インデックス構築中(重い処理、一度だけ)...")
    lap33_lookup = L33.load_lap33_lookup()
    race_meta = L33.load_race_surface_distance()  # pendingのrace_idはヒットしない(lap33軸はNaNになるだけ、他カテゴリには影響しない)
    history_index = L33.build_history_index()

    print("レーダー8カテゴリ軸を計算中...")
    axis_records = RC.build_axis_matrix(races, priors_v4, history_index, lap33_lookup, race_meta)
    axis_by_race = {rec["race_id"]: rec["axis"] for rec in axis_records}
    # 2026-09-14追加(レーダー解像度レビュー、build_factor_dataset.pyと同じ9本の細分化)。
    # axis_recordsが保持済みのsigを再利用するため新規のcompute_signals呼び出しは発生しない。
    sub_axis_by_race = RC.build_display_subcategory_axis(axis_records)

    print("本番スコアを計算中...")
    out = {}
    for r in races:
        df = r["df"]
        rid, rtype = r["race_id"], r["race_type"]
        prior = score_priors.get(rtype)
        if prior is None:
            continue
        score = MS.score_race(df, r["race_name"], rtype, prior, weights[rtype])
        pred_rank = MS.pred_rank_of(score)

        pred_rank_box4 = pred_rank_box3 = None
        if rtype == "normal":
            score_box4 = MS.score_race(df, r["race_name"], "normal_box4", score_priors["normal"], weights["normal_box4"])
            score_box3 = MS.score_race(df, r["race_name"], "normal_box3", score_priors["normal"], weights["normal_box3"])
            pred_rank_box4 = MS.pred_rank_of(score_box4)
            pred_rank_box3 = MS.pred_rank_of(score_box3)

        axis = axis_by_race.get(rid)
        rank_pedigree = rank_jt = None
        if axis is not None:
            if "血統適性" in axis.columns and axis["血統適性"].notna().any():
                rank_pedigree = axis["血統適性"].rank(method="min", ascending=False, na_option="bottom")
                rank_pedigree = rank_pedigree.where(axis["血統適性"].notna())
            if "騎手・厩舎" in axis.columns and axis["騎手・厩舎"].notna().any():
                rank_jt = axis["騎手・厩舎"].rank(method="min", ascending=False, na_option="bottom")
                rank_jt = rank_jt.where(axis["騎手・厩舎"].notna())

        rank_mark = None
        if axis is not None and "予想印・専門家評価" in axis.columns and axis["予想印・専門家評価"].notna().any():
            rank_mark = axis["予想印・専門家評価"].rank(method="min", ascending=False, na_option="bottom")
            rank_mark = rank_mark.where(axis["予想印・専門家評価"].notna())

        # 2026-09-06追記(ユーザー依頼「馬連頭軸レース帳」パターンの組み込み)で
        # 能力・調教評価順位・想定ペース・コース高低差を追加。いずれもjra_candidate_factors_v3.py
        # (v3)がfactor_database.json(結果確定済みレース)向けに計算しているのと全く同じ
        # ロジックをここに再現する(v3本体のcompute_for_race()自体はrace_results依存のため
        # pending日には使えない、という既存のv3ファクター追加と同じ制約)。
        rank_ability = None
        if axis is not None and "能力・調教評価" in axis.columns and axis["能力・調教評価"].notna().any():
            rank_ability = axis["能力・調教評価"].rank(method="min", ascending=False, na_option="bottom")
            rank_ability = rank_ability.where(axis["能力・調教評価"].notna())

        waku = JS._num(JS._col(df, "waku"))
        umaban = JS._num(JS._col(df, "umaban"))

        # --- v3ファクター(予想印×年齢×オッズ帯パターン用、2026-09-05追加) ---
        sex_age_col = JS._col(df, "bias_sex_age")
        weight_carried = JS._num(JS._col(df, "bias_weight_carried"))
        win_odds = JS._num(JS._col(df, "bias_win_odds"))
        jockey_id_col = JS._col(df, "bias_jockey_id")

        venue_code = rid[4:6] if isinstance(rid, str) and len(rid) == 12 else None
        cm = course_master.get(venue_code) if venue_code else None
        surface = r.get("surface")
        circumference = None
        if cm and isinstance(surface, str) and surface.strip():
            pre = "turf" if surface.strip() == "芝" else "dirt"
            circumference = cm.get(f"{pre}_circumference_m")
        course_size_band = CF3._band(circumference, [1700, 1900], ["small", "mid", "large"])
        elevation = cm.get(f"{pre}_elevation_m") if cm and isinstance(surface, str) and surface.strip() else None
        course_elevation_band = CF3._band(elevation, [2.0, 3.0], ["small", "mid", "large"])
        # 2026-09-12追加: 直線長バンド(消し材料マーク用、course_size_band/elevationと同じcmを流用)。
        straight = cm.get(f"{pre}_straight_m") if cm and isinstance(surface, str) and surface.strip() else None
        course_straight_band = CF3._band(straight, [300, 450], ["short", "mid", "long"])

        pace_col = JS._col(df, "race_pace_label")
        race_pace_label = None
        if pace_col is not None and len(pace_col):
            _v = pace_col.iloc[0]
            race_pace_label = str(_v).strip() if pd.notna(_v) else None

        # 2026-09-06追加(改善計画1-D): 内回り/外回り区分。distance_mはrace_names_{date}.csv
        # (predict_pattern29.py出力、発走前に判明済み)由来なのでリークにはならない。
        course_turn_type = "unknown"
        distance_m = r.get("distance_m")
        if venue_code and isinstance(surface, str) and surface.strip() and pd.notna(distance_m):
            # race_names_{date}.csvのsurfaceは「ダ」のように省略される場合がある
            # (2026-09-06発覚、jra_candidate_factors_v3.compute_for_raceと同じ対策)。
            surface_key = "芝" if surface.strip() == "芝" else "ダート"
            try:
                key = (venue_code, surface_key, int(float(distance_m)))
                course_turn_type = course_turn_type_table.get(key, "unknown")
            except (TypeError, ValueError):
                course_turn_type = "unknown"

        # 2026-09-12追加: 「外し/複外/W外し」消し材料マーク用のCF1/CF2ファクター・
        # flop_safety_rank・残る3レーダー軸(適性・調子・脚質)。surfaceはCF1.compute_for_race
        # 内部で史実データ(race_results由来、"芝"/"ダート"の正式表記)と直接比較するため、
        # course_turn_type算出と同じ正規化(surface_key)を流用する(2026-09-06発覚の表記揺れ対策)。
        surface_norm = None
        if isinstance(surface, str) and surface.strip():
            surface_norm = "芝" if surface.strip() == "芝" else "ダート"
        distance_m_num = None
        if pd.notna(distance_m):
            try:
                distance_m_num = float(distance_m)
            except (TypeError, ValueError):
                distance_m_num = None

        cf1 = CF1.compute_for_race(df, r["racecourse"], distance_m_num, surface_norm, args.date, cf1_hist_idx)
        cf2 = CF2.compute_for_race(df, r["race_name"], args.date, cf1_hist_idx)

        current_class = JS._class_ordinal(r["race_name"], JS.CLASS_ORDINAL)
        sig = JS.compute_signals(df, current_class, priors_v4, JS.CLASS_ORDINAL)
        flop_score = JS.combine_signals(sig, {n: 1.0 for n in JS.ALL_SIGNALS_V4})
        flop_safety_rank = flop_score.rank(method="min", ascending=False, na_option="bottom")

        rank_aptitude = rank_form = rank_style = None
        if axis is not None:
            if "コース・距離適性" in axis.columns and axis["コース・距離適性"].notna().any():
                rank_aptitude = axis["コース・距離適性"].rank(method="min", ascending=False, na_option="bottom")
                rank_aptitude = rank_aptitude.where(axis["コース・距離適性"].notna())
            if "近走成績・調子" in axis.columns and axis["近走成績・調子"].notna().any():
                rank_form = axis["近走成績・調子"].rank(method="min", ascending=False, na_option="bottom")
                rank_form = rank_form.where(axis["近走成績・調子"].notna())
            if "脚質・展開" in axis.columns and axis["脚質・展開"].notna().any():
                rank_style = axis["脚質・展開"].rank(method="min", ascending=False, na_option="bottom")
                rank_style = rank_style.where(axis["脚質・展開"].notna())

        # 2026-09-14追加(レーダー解像度レビュー: 脚質・展開/騎手・厩舎/血統適性の細分化+
        # 近走成績・調子からのagari独立、計9本)。build_factor_dataset.pyと同じ
        # RC.DISPLAY_SUBCATEGORY_SLUG(単一の日本語名→slug対応表)を共有しているため、
        # 2箇所で手書きし直して食い違うリスクが無い。
        sub_axis = sub_axis_by_race.get(rid)
        sub_ranks = {}
        for cat, slug in RC.DISPLAY_SUBCATEGORY_SLUG.items():
            if sub_axis is not None and cat in sub_axis.columns and sub_axis[cat].notna().any():
                r_ = sub_axis[cat].rank(method="min", ascending=False, na_option="bottom")
                sub_ranks[slug] = r_.where(sub_axis[cat].notna())
            else:
                sub_ranks[slug] = None

        ninki = JS._num(JS._col(df, "bias_ninki"))
        ca_jockey_wr = JS._pct(JS._col(df, "ca_jockey_win_rate"))
        ca_trainer_wr = JS._pct(JS._col(df, "ca_trainer_win_rate"))
        corner4 = JS._num(JS._col(df, "corner4_rank"))
        past1_finish = JS._num(JS._col(df, "past1_finish"))
        training_course_col = JS._col(df, "training_course")
        training_track_col = JS._col(df, "training_track_condition")
        past1_date_col = JS._col(df, "past1_date").map(CF3._parse_dot_date)
        kaisai_dt = pd.to_datetime(args.date, format="%Y%m%d", errors="coerce")
        trainer_id_col = JS._col(df, "bias_trainer_id")

        horses = []
        for i in df.index:
            age = None
            if i in sex_age_col.index and pd.notna(sex_age_col.at[i]):
                age = CF3._parse_sex_age(sex_age_col.at[i])[1]
            jockey_id = None
            if i in jockey_id_col.index and pd.notna(jockey_id_col.at[i]):
                jockey_id = str(jockey_id_col.at[i]).strip() or None
            jp = jockey_profiles.get(jockey_id) if jockey_id else None
            trainer_id = None
            if i in trainer_id_col.index and pd.notna(trainer_id_col.at[i]):
                trainer_id = str(trainer_id_col.at[i]).strip() or None
            tp = trainer_profiles.get(trainer_id) if trainer_id else None
            t_area = tp["area"] if tp and tp["area"] else None

            p1d = past1_date_col.at[i] if i in past1_date_col.index else pd.NaT
            interval_days = None if (pd.isna(p1d) or pd.isna(kaisai_dt)) else (kaisai_dt - p1d).days

            horses.append({
                "umaban": int(umaban.at[i]) if pd.notna(umaban.at[i]) else None,
                "waku": int(waku.at[i]) if pd.notna(waku.at[i]) else None,
                "pred_rank": int(pred_rank.at[i]),
                "pred_rank_box4": int(pred_rank_box4.at[i]) if pred_rank_box4 is not None else None,
                "pred_rank_box3": int(pred_rank_box3.at[i]) if pred_rank_box3 is not None else None,
                "radar_rank_pedigree": _rank_val(rank_pedigree, i) if rank_pedigree is not None else None,
                "radar_rank_jt": _rank_val(rank_jt, i) if rank_jt is not None else None,
                "radar_rank_mark": _rank_val(rank_mark, i) if rank_mark is not None else None,
                "radar_rank_ability": _rank_val(rank_ability, i) if rank_ability is not None else None,
                # --- 2026-09-12追加(消し材料マーク用) ---
                "radar_rank_aptitude": _rank_val(rank_aptitude, i) if rank_aptitude is not None else None,
                "radar_rank_form": _rank_val(rank_form, i) if rank_form is not None else None,
                "radar_rank_style": _rank_val(rank_style, i) if rank_style is not None else None,
                # --- レーダー細分化フィルタ(2026-09-14追加) ---
                "radar_rank_style_position": _rank_val(sub_ranks["style_position"], i) if sub_ranks["style_position"] is not None else None,
                "radar_rank_style_stamina": _rank_val(sub_ranks["style_stamina"], i) if sub_ranks["style_stamina"] is not None else None,
                "radar_rank_style_corner_move": _rank_val(sub_ranks["style_corner_move"], i) if sub_ranks["style_corner_move"] is not None else None,
                "radar_rank_jt_base": _rank_val(sub_ranks["jt_base"], i) if sub_ranks["jt_base"] is not None else None,
                "radar_rank_jt_change": _rank_val(sub_ranks["jt_change"], i) if sub_ranks["jt_change"] is not None else None,
                "radar_rank_jt_stats": _rank_val(sub_ranks["jt_stats"], i) if sub_ranks["jt_stats"] is not None else None,
                "radar_rank_pedigree_pure": _rank_val(sub_ranks["pedigree_pure"], i) if sub_ranks["pedigree_pure"] is not None else None,
                "radar_rank_pedigree_training": _rank_val(sub_ranks["pedigree_training"], i) if sub_ranks["pedigree_training"] is not None else None,
                "radar_rank_form_agari": _rank_val(sub_ranks["form_agari"], i) if sub_ranks["form_agari"] is not None else None,
                "flop_safety_rank": _rank_val(flop_safety_rank, i),
                "bias_ninki": int(ninki.at[i]) if i in ninki.index and pd.notna(ninki.at[i]) else None,
                "grade_best": cf1["grade_best"].at[i] if i in cf1["grade_best"].index else None,
                "jockey_switch": cf1["jockey_switch"].at[i] if i in cf1["jockey_switch"].index else None,
                "career_place_rate": cf1["career_place_rate"].at[i] if i in cf1["career_place_rate"].index else None,
                "distance_band_place_rate": cf1["distance_band_place_rate"].at[i] if i in cf1["distance_band_place_rate"].index else None,
                "turn_apt_place_rate": cf1["turn_apt_place_rate"].at[i] if i in cf1["turn_apt_place_rate"].index else None,
                "course_size_apt_place_rate": cf1["course_size_apt_place_rate"].at[i] if i in cf1["course_size_apt_place_rate"].index else None,
                "class_form_place_rate": cf2["class_form_place_rate"].at[i] if i in cf2["class_form_place_rate"].index else None,
                "main_jockey_return_flag": cf2["main_jockey_return_flag"].at[i] if i in cf2["main_jockey_return_flag"].index else None,
                "ca_jockey_win_rate": float(ca_jockey_wr.at[i]) if i in ca_jockey_wr.index and pd.notna(ca_jockey_wr.at[i]) else None,
                "ca_trainer_win_rate": float(ca_trainer_wr.at[i]) if i in ca_trainer_wr.index and pd.notna(ca_trainer_wr.at[i]) else None,
                "corner4_expected_rank": int(corner4.at[i]) if i in corner4.index and pd.notna(corner4.at[i]) else None,
                "past1_finish_band": CF3._past1_finish_band(past1_finish.at[i] if i in past1_finish.index else None),
                "training_course_type": CF3._training_course_type(training_course_col.at[i] if i in training_course_col.index else None),
                "training_track_condition": (
                    str(training_track_col.at[i]).strip()
                    if i in training_track_col.index and pd.notna(training_track_col.at[i])
                    and str(training_track_col.at[i]).strip() else None
                ),
                "interval_band": CF3._interval_band(interval_days),
                "course_straight_band": course_straight_band,
                "trainer_lead_rank": tp["season_rank"] if tp else None,
                "transport_flag": CF3._transport_flag(t_area, r["racecourse"]),
                "jockey_career_wins_band": CF3._career_wins_band(jp["career_wins"]) if jp else None,
                "owner_prior_win_rate": None,  # 発走前ソース(bias_owner_id相当)が無いため常に非対応
                "horse_age": age,
                "weight_carried_band": CF3._weight_carried_band(weight_carried.at[i]) if i in weight_carried.index else None,
                "odds_band": CF3._odds_band(win_odds.at[i]) if i in win_odds.index else None,
                "course_size_band": course_size_band,
                "course_elevation_band": course_elevation_band,
                "course_turn_type": course_turn_type,
                "race_pace_label": race_pace_label,
                "jockey_area": (jp["area"] if jp and jp["area"] else "unknown"),
                "jockey_lead_rank": jp["season_rank"] if jp else None,
            })
        out[rid] = {
            "race_type": rtype, "race_name": r["race_name"],
            "kaisai_date": args.date, "horses": horses,
        }

    out_path = DATA_DIR / f"pending_factor_{args.date}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out_path} ({len(out)} races)")


if __name__ == "__main__":
    main()
