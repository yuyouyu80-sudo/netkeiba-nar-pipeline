# -*- coding: utf-8 -*-
"""凡走予測(flop)専用のシグナル定義(2026-09-11新設)。

計画: `C:\\Users\\yuyou\\.claude\\plans\\valiant-cuddling-aho.md`
(「JRA凡走予測モデル(flop)を『参考表示専用』から『本番予想に資する検証済みシグナル』へ」)

現行の`jra_reference_panel.py`/`build_factor_dataset.py.compute_flop_safety_rank()`は、
本番score(_score)が使う40シグナル(`jra_signals.ALL_SIGNALS_V4`)を等重みで合成して符号を
反転しただけで、独自の予測情報をほとんど持たない(NAR側で「既存シグナルの向き反転流用は
情報量が薄く不採用」と判定された構造的弱点と同じ)。本モジュールはそれとは別系統で、
既存の`jra_candidate_factors_v2.py`/`v3.py`(騎手・調教師・馬主の実データ)から
「陣営コンディション」寄りの凡走専用シグナルを導出する。

設計原則(シニアエンジニアレビューを反映済み):
  - `jra_signals.py`本体・`predict_pattern29.py`・既存の全win-score探索スクリプトは無改造。
    flopは独立した第二のcombineパイプラインとして完全に分離する
    (ALL_SIGNALS_V4/V5には一切追加しない)。
  - 生の特徴量は`jra_candidate_factors_v2.py`(`compute_layoff_factors`)・`v3.py`
    (`Context`, `compute_for_race`)が返す既存の計算結果をそのまま再利用する。二重の
    CSVロード・Context再構築を避けるため、本モジュールの関数は「呼び出し側が1レースにつき
    1回だけ計算済みのcf2/cf3辞書」を受け取る設計にする(build_factor_dataset.py・
    Phase0診断・G1ゲートスクリプトがいずれも同じcf2/cf3を共有できる)。
  - `_minmax`はjra_signals.pyと同じ「レース内(within-race)min-max」の意味論(1レースの
    出走馬間で相対順位化する)。全馬同値・全馬NaNの場合はNaN(重み再配分、危険側に倒さない)。
  - 極性(符号)は実データで検証済み(第一弾4候補、下表参照)。
      - owner_prior_win_rate(勝率%、高いほど良い) → 反転**必要**: 1 - _minmax(v)
      - jockey_lead_rank/trainer_lead_rank(season_rank、数値が大きいほど成績が悪い
        =既に危険方向) → 反転**しない**: _minmax(v) そのまま
        (当初案は誤って反転を指示していたが、レビューで発覚・修正済み。反転すると
        好成績の騎手・調教師ほど「危険」と誤判定する逆向きバグになる)
      - layoff_flag(カテゴリ文字列) → 固定マップ(_LAYOFF_DANGER_MAP)で0/0.5/1.0化
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LIB_DIR))
import jra_signals as JS  # noqa: E402 (_minmax/_num/combine_signalsを再利用)

# 第一弾候補プール(4本)。第二弾以降の候補(jockey_switch/trainer_change/
# stable_multi_entry/class_challenge/second_after_layoff/weight_diff_extreme/
# transport_cross_region/apprentice_jockey等)は、この4本でG1を通してから追加検討する
# (多重比較負荷とカテゴリ変換の実装負荷を先に小さくする、計画Phase0参照)。
FLOP_SIGNALS_V1 = [
    "flop_owner_low_winrate",
    "flop_jockey_lead_rank_high",
    "flop_trainer_lead_rank_high",
    "flop_layoff_long",
]

_LAYOFF_DANGER_MAP = {
    "long_layoff_180d_plus": 1.0,
    "mid_layoff_90_179d": 0.5,
    "normal_interval": 0.0,
    # "unknown" / "no_history" はマップに存在しない -> NaN(重み再配分、危険側に倒さない)
}


def compute_flop_signals(cf2: dict, cf3: dict) -> dict:
    """既に計算済みのcf2(`jra_candidate_factors_v2.compute_for_race()`出力)・
    cf3(`jra_candidate_factors_v3.compute_for_race()`出力)から、レース単位
    (within-race minmax)でflop(危険方向、0..1、高いほど危険)シグナルを導出する。

    cf2/cf3は呼び出し側が同じdf(1レース分)に対して既に計算済みのものをそのまま渡すこと
    (build_factor_dataset.pyの既存ループ内で計算されているcf2/cf3をそのまま流用できる)。
    """
    owner_wr = JS._num(cf3["owner_prior_win_rate"])
    jockey_rank = JS._num(cf3["jockey_lead_rank"])
    trainer_rank = JS._num(cf3["trainer_lead_rank"])
    layoff = cf2["layoff_flag"]

    sig = {}
    sig["flop_owner_low_winrate"] = 1.0 - JS._minmax(owner_wr)
    sig["flop_jockey_lead_rank_high"] = JS._minmax(jockey_rank)
    sig["flop_trainer_lead_rank_high"] = JS._minmax(trainer_rank)
    sig["flop_layoff_long"] = layoff.map(_LAYOFF_DANGER_MAP).astype(float)
    return sig


def compute_flop_composite(sig: dict, weights: dict = None) -> pd.Series:
    """4シグナルを合成した単一の危険度スコア(0..1、高いほど危険)を返す。
    weights省略時は等重み(第一弾G1で使う既定)。jra_signals.combine_signalsをそのまま
    再利用する(重み再配分の意味論を完全一致させる、独自実装しない)。"""
    if weights is None:
        weights = {n: 1.0 for n in FLOP_SIGNALS_V1}
    return JS.combine_signals(sig, weights)


def flop_signal_matrices(cf2_by_race: list, cf3_by_race: list, names: list = None) -> list:
    """jra_signals.signal_matrices()のflop版。cf2_by_race/cf3_by_raceは、母集団の各レースに
    対応するcf2/cf3辞書のリスト(レース順序はscore側のmats(jra_eval用)と揃えること)。
    戻り値は{"S": S, "A": A}のリストで、jra_eval.score_picks()と同じ行列演算
    (score = (S@w)/(A@w))でadjusted_picksから使う。"""
    names = names or FLOP_SIGNALS_V1
    mats = []
    for cf2, cf3 in zip(cf2_by_race, cf3_by_race):
        sig = compute_flop_signals(cf2, cf3)
        S = np.column_stack([sig[n].fillna(0.0).to_numpy(dtype=float) for n in names])
        A = np.column_stack([sig[n].notna().to_numpy(dtype=float) for n in names])
        mats.append({"S": S, "A": A})
    return mats



# ============================================================================
# FLOP_SIGNALS_V2(2026-09-11、候補200の網羅探索用)
#
# 計画: `C:\\Users\\yuyou\\.claude\\plans\\valiant-cuddling-aho.md`
# 「JRA凡走シグナル(flop) 全候補(候補200由来 約40本)の網羅探索」
#
# FLOP_SIGNALS_V1(4本)とは別系統。人間が事前に「危険方向」を defensible に説明できる
# 候補だけをプールし、Dirichlet多パターン探索(jra_flop_search_v2_*.py)に重み配分を委ねる
# (個別候補の単体有意性検定はしない)。
#
# 実装時に計画書の想定候補数(39)から以下の通り絞り込んだ(post-hoc符号決定をしないという
# 制約3を、個々のフィールドに厳密に適用し直した結果。計画の集計自体に軽微な数え間違いも
# あったため、ここで実コードとして確定させる):
#   - `bad_run_popularity_drop_flag`: 前走大敗+人気降下は「不利による割安」の解釈もでき、
#     符号が凡走でなく市場効率性(割安)寄りになりうる(FLOP_SIGNALS_V1設計時に既出の除外理由、
#     このモジュールの冒頭docstring該当なし)。除外。
#   - `jockey_affiliation_type`(フリー/厩舎所属): 現代JRAでは主要騎手の多くがフリーであり、
#     「フリー=危険」というa priori方向は defensible でない。除外。
#   - `weight_carried_band`/`horse_weight_band`: 斤量・馬体重そのものは軽い/重いのどちらが
#     危険かに単調な向きが無い(むしろU字型、かつ理論的にも論争がある)。除外。
#   - `training_best_time_flag`: 「一番時計」は好材料のみを示すフラグで、「normal」側に
#     危険方向の意味を持たせる根拠が無い(ポジティブ専用シグナル)。除外。
#   - `pred_rank`/`pred_rank_box4`/`pred_rank_box3`: box_nごとに異なる値を要する
#     (box_nに依存しない候補プールという設計を崩す)上、本番`_score`の言い換えという
#     restatement riskが最も高い候補でもあるため、今回のスコープからは外す(Phase0/1の結果を
#     見てから再検討する)。
#
# 確定候補数: CF_V1由来11 + CF_V2由来5(bad_run除外) + CF_V3由来14 = 30本。
FLOP_SIGNALS_V2_SPEC = [
    # --- jra_candidate_factors_v1.compute_for_race() 由来(11) ---
    {"name": "flop2_career_place_rate", "source": "cf1", "field": "career_place_rate",
     "kind": "high_good"},
    {"name": "flop2_distance_band_place_rate", "source": "cf1", "field": "distance_band_place_rate",
     "kind": "high_good"},
    {"name": "flop2_turn_apt_place_rate", "source": "cf1", "field": "turn_apt_place_rate",
     "kind": "high_good"},
    {"name": "flop2_course_size_apt_place_rate", "source": "cf1",
     "field": "course_size_apt_place_rate", "kind": "high_good"},
    {"name": "flop2_slope_apt_place_rate", "source": "cf1", "field": "slope_apt_place_rate",
     "kind": "high_good"},
    {"name": "flop2_turf_type_apt_place_rate", "source": "cf1", "field": "turf_type_apt_place_rate",
     "kind": "high_good"},
    {"name": "flop2_grade_best", "source": "cf1", "field": "grade_best", "kind": "category_map",
     "map": {"no_history": 1.0, "no_graded_starts": 0.8, "graded_no_top3": 0.5,
             "g3_top3": 0.2, "g2_top3": 0.1, "g1_top3": 0.0}},
    {"name": "flop2_first_time_flag", "source": "cf1", "field": "first_time_flag",
     "kind": "category_map",
     "map": {"both_first": 1.0, "surface_first": 0.5, "distance_first": 0.5, "experienced": 0.0}},
    {"name": "flop2_jockey_switch", "source": "cf1", "field": "jockey_switch",
     "kind": "category_map",
     "map": {"switch": 1.0, "continuous1": 0.3, "continuous2plus": 0.0}},
    {"name": "flop2_trainer_change", "source": "cf1", "field": "trainer_change",
     "kind": "category_map", "map": {"changed": 1.0, "same": 0.0}},
    {"name": "flop2_stable_multi_entry", "source": "cf1", "field": "stable_multi_entry",
     "kind": "category_map", "map": {"multi": 1.0, "single": 0.0}},
    # --- jra_candidate_factors_v2.compute_for_race() 由来(5、bad_run_popularity_drop_flagは除外) ---
    {"name": "flop2_class_form_place_rate", "source": "cf2", "field": "class_form_place_rate",
     "kind": "high_good"},
    {"name": "flop2_class_challenge_flag", "source": "cf2", "field": "class_challenge_flag",
     "kind": "category_map", "map": {"up_challenge": 1.0, "at_or_below_max": 0.0}},
    {"name": "flop2_layoff_flag", "source": "cf2", "field": "layoff_flag", "kind": "category_map",
     "map": dict(_LAYOFF_DANGER_MAP)},
    {"name": "flop2_second_after_layoff_flag", "source": "cf2", "field": "second_after_layoff_flag",
     "kind": "category_map", "map": {"second_after_layoff": 1.0, "not_applicable": 0.0}},
    {"name": "flop2_main_jockey_return_flag", "source": "cf2", "field": "main_jockey_return_flag",
     "kind": "category_map",
     "map": {"away_from_main": 1.0, "no_clear_main": 0.5, "return_to_main": 0.0}},
    # --- jra_candidate_factors_v3.compute_for_race() 由来(14) ---
    {"name": "flop2_jockey_lead_rank", "source": "cf3", "field": "jockey_lead_rank",
     "kind": "high_bad"},  # season_rank、数値大=成績悪い=既に危険方向(反転しない、M2の教訓)
    {"name": "flop2_trainer_lead_rank", "source": "cf3", "field": "trainer_lead_rank",
     "kind": "high_bad"},
    {"name": "flop2_owner_prior_win_rate", "source": "cf3", "field": "owner_prior_win_rate",
     "kind": "high_good"},
    {"name": "flop2_jockey_career_wins_band", "source": "cf3", "field": "jockey_career_wins_band",
     "kind": "category_map",
     "map": {"le30": 1.0, "w31_50": 0.7, "w51_100": 0.5, "w101_500": 0.2, "over500": 0.0}},
    {"name": "flop2_transport_flag", "source": "cf3", "field": "transport_flag",
     "kind": "category_map", "map": {"cross_region": 1.0, "local_away": 0.5, "home": 0.0}},
    {"name": "flop2_breeder_group", "source": "cf3", "field": "breeder_group",
     "kind": "category_map",
     "map": {"other": 1.0, "shadai_group_other": 0.3, "shadai_farm": 0.1, "northern_farm": 0.0}},
    {"name": "flop2_horse_age", "source": "cf3", "field": "horse_age", "kind": "high_bad"},
    {"name": "flop2_horse_weight_diff_band", "source": "cf3", "field": "horse_weight_diff_band",
     "kind": "category_map",  # U字型: 大幅増減どちらも軽度の危険、変化なし(zero)が最も安全
     "map": {"minus10": 0.7, "minus_small": 0.2, "zero": 0.0, "plus_small": 0.2, "plus10": 0.6}},
    {"name": "flop2_interval_band", "source": "cf3", "field": "interval_band",
     "kind": "category_map",  # 連闘(短すぎ)・6ヶ月超(長すぎ)がU字型に危険、3-8週が最も安全
     "map": {"renchaku": 0.6, "w1_4": 0.0, "w5_8": 0.0, "m2_6": 0.4, "over6m": 0.8}},
    {"name": "flop2_corner4_expected_rank", "source": "cf3", "field": "corner4_expected_rank",
     "kind": "high_bad"},
    {"name": "flop2_past1_finish_band", "source": "cf3", "field": "past1_finish_band",
     "kind": "category_map",
     "map": {"win": 0.0, "f2_3": 0.15, "f4_5": 0.35, "f6_9": 0.6, "f10plus": 1.0}},
    {"name": "flop2_training_track_condition", "source": "cf3", "field": "training_track_condition",
     "kind": "category_map",  # 実データの表記がここに無ければ充足率0(Phase0診断で検知できる)
     "map": {"良": 0.0, "稍重": 0.3, "重": 0.6, "不良": 1.0}},
    {"name": "flop2_ca_jockey_win_rate", "source": "cf3", "field": "ca_jockey_win_rate",
     "kind": "high_good"},
    {"name": "flop2_ca_trainer_win_rate", "source": "cf3", "field": "ca_trainer_win_rate",
     "kind": "high_good"},
]

FLOP_SIGNALS_V2 = [spec["name"] for spec in FLOP_SIGNALS_V2_SPEC]


def compute_flop_signals_v2(cf1: dict, cf2: dict, cf3: dict) -> dict:
    """FLOP_SIGNALS_V2_SPECの宣言的テーブルに従い、cf1(`jra_candidate_factors_v1.
    compute_for_race()`出力)・cf2(`jra_candidate_factors_v2.compute_for_race()`出力)・
    cf3(`jra_candidate_factors_v3.compute_for_race()`出力、いずれも呼び出し側が1レースにつき
    1回だけ計算済みのものを渡す)から、レース内(within-race)danger方向0..1のシグナル辞書を返す。

    kind:
      - "high_good": 元データは高いほど安全(勝率・複勝率等) -> 1 - _minmax(_num(v))
      - "high_bad" : 元データは高いほど既に危険方向(順位が大きい=下位、年齢が高い等、
                     符号反転はしない) -> _minmax(_num(v))
      - "category_map": 明示的な危険度辞書、マップに無いキー("unknown"/"no_history"等)は
                     自動的にNaN(pd.Series.mapの仕様、重み再配分・危険側に倒さない)
    """
    sources = {"cf1": cf1, "cf2": cf2, "cf3": cf3}
    sig = {}
    for spec in FLOP_SIGNALS_V2_SPEC:
        raw = sources[spec["source"]][spec["field"]]
        kind = spec["kind"]
        if kind == "high_good":
            sig[spec["name"]] = 1.0 - JS._minmax(JS._num(raw))
        elif kind == "high_bad":
            sig[spec["name"]] = JS._minmax(JS._num(raw))
        elif kind == "category_map":
            sig[spec["name"]] = raw.map(spec["map"]).astype(float)
        else:
            raise ValueError(f"unknown kind: {kind}")
    return sig


def flop_signal_matrices_v2(cf1_by_race: list, cf2_by_race: list, cf3_by_race: list,
                            names: list = None) -> list:
    """flop_signal_matrices()のV2版(cf1も受け取る3引数)。names省略時はFLOP_SIGNALS_V2全体。"""
    names = names or FLOP_SIGNALS_V2
    mats = []
    for cf1, cf2, cf3 in zip(cf1_by_race, cf2_by_race, cf3_by_race):
        sig = compute_flop_signals_v2(cf1, cf2, cf3)
        S = np.column_stack([sig[n].fillna(0.0).to_numpy(dtype=float) for n in names])
        A = np.column_stack([sig[n].notna().to_numpy(dtype=float) for n in names])
        mats.append({"S": S, "A": A})
    return mats


def flop_precision_table(flagged_mask_by_race: list, population_mask_by_race: list,
                         finish_by_race: list) -> dict:
    """Phase0の安価な事前スクリーニング用(統計的採否には使わない)。
    population(例: 人気1-5位、またはpred_rank1-5位)の中で、flaggedとnot-flaggedそれぞれの
    「複勝圏外(4着以下、またはDNF/NaN)」率を比較する。

    flagged_mask_by_race[i]/population_mask_by_race[i]: レースiの各馬に対する真偽値配列
    (populationでない馬はflaggedであってもカウントしない)。
    finish_by_race[i]: レースiの各馬の着順(数値、DNFはNaNまたは十分大きい値で表現しておく)。
    """
    flagged_outside = flagged_n = other_outside = other_n = 0
    for flag, pop, finish in zip(flagged_mask_by_race, population_mask_by_race, finish_by_race):
        flag = np.asarray(flag, dtype=bool)
        pop = np.asarray(pop, dtype=bool)
        finish = np.asarray(finish, dtype=float)
        for i in np.where(pop)[0]:
            outside = bool(np.isnan(finish[i]) or finish[i] > 3)
            if flag[i]:
                flagged_outside += outside
                flagged_n += 1
            else:
                other_outside += outside
                other_n += 1
    return {
        "flagged_outside_rate": flagged_outside / flagged_n if flagged_n else None,
        "baseline_outside_rate": other_outside / other_n if other_n else None,
        "lift_pt": (100 * (flagged_outside / flagged_n - other_outside / other_n)
                   if flagged_n and other_n else None),
        "n_flagged": flagged_n, "n_baseline": other_n,
    }
