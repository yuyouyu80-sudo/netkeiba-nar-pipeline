# -*- coding: utf-8 -*-
"""ファクター検証データベース用JSONの生成オーケストレーター。

`jra_dataset_wide.py`(3母集団)・`jra_model_scoring.py`(本番スコア再現)・
`jra_factor_registry.py`(ファクター定義)・既存の`jra_radar_categories.py`/
`jra_reference_panel.py`のロジックを統合し、`data/jra_pipeline/factor_database.json`を
書き出す。モデル計算側(`jra_signals.py`等)は無改造。

レーダー8カテゴリ・33ラップ理論・凡走予測のpriorsは、既存の日次レポート表示値との整合性を
優先し、通常戦246レースのみ(`jra_dataset.load()`)から計算する(計画セクション5の設計判断、
widened母集団用に作り直さない)。本番予想スコア(pred_rank)だけはレースタイプごとに
正しいpriorsを使う(jra_model_scoring参照)。
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
OUT_JSON = DATA_DIR / "factor_database.json"

sys.path.insert(0, str(LIB_DIR))
import jra_candidate_factors_v1 as CF1  # noqa: E402 (新規候補ファクターv1、2026-09-01追加)
import jra_candidate_factors_v2 as CF2  # noqa: E402 (新規候補ファクターv2、2026-09-02追加)
import jra_candidate_factors_v3 as CF3  # noqa: E402 (新規候補ファクターv3、2026-09-04追加)
import jra_dataset as JD  # noqa: E402 (通常戦246レース、レーダー/lap33/flopのpriors用)
import jra_dataset_wide as JDW  # noqa: E402
import jra_eval as JE  # noqa: E402
import jra_history as JH  # noqa: E402 (JH.load_results()をL33/CF1で共有し二重ロードを避ける)
import jra_lap33_signals as L33  # noqa: E402
import jra_model_scoring as MS  # noqa: E402
import jra_radar_categories as RC  # noqa: E402
import jra_signals as JS  # noqa: E402
from jra_factor_registry import FACTOR_GROUPS, GROUP_TIERS  # noqa: E402

LAP33_CSV = DATA_DIR / "jra_lap33_by_race.csv"

CATEGORY_SLUG = {
    "能力・調教評価": "ability", "近走成績・調子": "form", "脚質・展開": "style",
    "コース・距離適性": "aptitude", "血統適性": "pedigree", "騎手・厩舎": "jt",
    "予想印・専門家評価": "mark", "33ラップ理論適合度": "lap33",
}


def load_lap33_lookup_persistent() -> dict:
    """L33.load_lap33_lookup()のscratchpad依存を避けた版(L33本体は無改造、パスだけ
    git管理下のCSVに差し替える)。"""
    df = pd.read_csv(LAP33_CSV, dtype={"race_id": str})
    return dict(zip(df["race_id"], df["lap33"]))


def compute_flop_safety_rank(df: pd.DataFrame, race_name: str, priors_v4: dict) -> pd.Series:
    """jra_reference_panel.build_reference_panel()のflop_score計算を移植し、符号を反転した
    「安全度」順位(1位=最も凡走しにくい)にする。"""
    current_class = JS._class_ordinal(race_name, JS.CLASS_ORDINAL)
    sig = JS.compute_signals(df, current_class, priors_v4, JS.CLASS_ORDINAL)
    weights_equal = {n: 1.0 for n in JS.ALL_SIGNALS_V4}
    flop_score = JS.combine_signals(sig, weights_equal)
    return flop_score.rank(method="min", ascending=False, na_option="bottom")


def main():
    print("母集団ロード中...")
    wide = JDW.load(rebuild=False)
    races, actual, offered = wide["races"], wide["actual"], wide["offered_bet_types"]

    print("priors計算中...")
    by_type = {}
    for r in races:
        by_type.setdefault(r["race_type"], []).append(r["df"])
    score_priors = MS.compute_priors_for_population(by_type)
    weights = {rt: MS.load_weights(rt) for rt in ("normal", "shinba", "mishoubi")}
    # BOX4/BOX3(2026-09-02追加、ユーザー依頼)。通常戦(246レース)にのみ存在するモデルで、
    # シグナル構成・priorsは"normal"と完全に同一(重みだけが異なる)ため、priorsは
    # score_priors["normal"]をそのまま流用する(score_priors["normal_box4"]は作らない)。
    weights["normal_box4"] = MS.load_weights("normal_box4")
    weights["normal_box3"] = MS.load_weights("normal_box3")

    # レーダー8カテゴリ・33ラップ・凡走予測のpriorsは既存日次レポートとの整合性を優先し、
    # 通常戦246レース(jra_dataset.load())のみから計算する(計画セクション5)。
    normal_246 = JD.load(rebuild=False)["races"]
    priors_v4 = JS.make_priors([r["df"] for r in normal_246])

    print("レーダー8カテゴリ軸を計算中(491レース全件)...")
    lap33_lookup = load_lap33_lookup_persistent()
    race_meta = L33.load_race_surface_distance()
    # JH.load_results()は数万行規模のCSV結合+レースごとのbeaten_by_sec計算を伴い重いため、
    # L33(レーダー用)とCF1(新規候補ファクターv1用)で1回だけロードして共有する(二重ロード回避)。
    results_all = JH.load_results()
    history_index = JH.HorseHistoryIndex(results_all)

    # 2026-09-13追記(JRAデータ資産棚卸し監査B-4): jra_dataset_wide.load()が対象にするのは
    # race_results/payoutsが両方存在する=既に確定済みのレースのみ(pending日は
    # jra_pending_factor_scoring.py側の別経路であり、そちらはrace_results非依存のまま
    # 変更しない)。よってEV計算のオッズは、newspaper由来の取得時点値(bias_win_odds、
    # 既知の通り約25%のレースで最終確定オッズと乖離する)ではなく、race_results.odds_final
    # (確定値)を優先する。odds_finalが無い馬(出走取消等)だけbias_win_oddsへ個別に
    # フォールバックする。
    _odds_final_num = JS._num(results_all["odds_final"])
    _umaban_num = JS._num(results_all["umaban"])
    odds_final_by_race: dict = {}
    for rid_, uma_, odds_ in zip(results_all["race_id"], _umaban_num, _odds_final_num):
        if pd.isna(uma_) or pd.isna(odds_):
            continue
        odds_final_by_race.setdefault(rid_, {})[int(uma_)] = float(odds_)
    axis_records = RC.build_axis_matrix(races, priors_v4, history_index, lap33_lookup, race_meta)
    axis_by_race = {rec["race_id"]: rec["axis"] for rec in axis_records}
    # 2026-09-14追加(レーダー解像度レビュー、脚質・展開/騎手・厩舎/血統適性の細分化+
    # 近走成績・調子からのagari独立、計9本)。axis_recordsが保持済みのsigを再利用するため
    # compute_signals()/lap33_fit_matrix()の再計算は発生しない。
    sub_axis_by_race = RC.build_display_subcategory_axis(axis_records)

    print("新規候補ファクターv1を計算中...")
    cf1_hist_idx = CF1.build_history_index(results_all)

    print("新規候補ファクターv3の参照データ(騎手/調教師/馬プロフィール・コースマスタ・"
          "jra_baba・race_results由来のID索引)をロード中...")
    cf3_ctx = CF3.Context(results_all)

    print("本番スコア・参考ファクターを計算中...")
    out_races = []
    pop_counts = {"normal": 0, "shinba": 0, "mishoubi": 0}
    for r in races:
        df = r["df"]
        rid, rtype = r["race_id"], r["race_type"]
        pop_counts[rtype] += 1

        score = MS.score_race(df, r["race_name"], rtype, score_priors[rtype], weights[rtype])
        pred_rank = MS.pred_rank_of(score)

        # BOX4/BOX3予想スコア順位(通常戦のみ、priorsはnormalを流用)
        if rtype == "normal":
            score_box4 = MS.score_race(df, r["race_name"], "normal_box4", score_priors["normal"], weights["normal_box4"])
            score_box3 = MS.score_race(df, r["race_name"], "normal_box3", score_priors["normal"], weights["normal_box3"])
            pred_rank_box4 = MS.pred_rank_of(score_box4)
            pred_rank_box3 = MS.pred_rank_of(score_box3)
        else:
            pred_rank_box4 = pred_rank_box3 = None

        axis = axis_by_race.get(rid)
        radar_ranks = {}
        if axis is not None:
            for cat, slug in CATEGORY_SLUG.items():
                if cat in axis.columns:
                    radar_ranks[slug] = axis[cat].rank(method="min", ascending=False, na_option="bottom")
                    radar_ranks[slug] = radar_ranks[slug].where(axis[cat].notna())
                else:
                    radar_ranks[slug] = pd.Series(np.nan, index=df.index)
        else:
            radar_ranks = {slug: pd.Series(np.nan, index=df.index) for slug in CATEGORY_SLUG.values()}

        # 2026-09-14追加: レーダー細分化フィルタ(9本)。既存radar_ranksと全く同じ抽出
        # ロジック(rank→notnaでNaN復元)をRC.DISPLAY_SUBCATEGORY_SLUGの9カテゴリに適用するだけ。
        sub_axis = sub_axis_by_race.get(rid)
        sub_radar_ranks = {}
        if sub_axis is not None:
            for cat, slug in RC.DISPLAY_SUBCATEGORY_SLUG.items():
                if cat in sub_axis.columns:
                    sub_radar_ranks[slug] = sub_axis[cat].rank(method="min", ascending=False, na_option="bottom")
                    sub_radar_ranks[slug] = sub_radar_ranks[slug].where(sub_axis[cat].notna())
                else:
                    sub_radar_ranks[slug] = pd.Series(np.nan, index=df.index)
        else:
            sub_radar_ranks = {slug: pd.Series(np.nan, index=df.index)
                               for slug in RC.DISPLAY_SUBCATEGORY_SLUG.values()}

        flop_safety_rank = compute_flop_safety_rank(df, r["race_name"], priors_v4)

        meta = race_meta.get(rid)
        cf1_racecourse, cf1_surface, cf1_distance = meta if meta else (r["racecourse"], None, None)
        cf1 = CF1.compute_for_race(df, cf1_racecourse, cf1_distance, cf1_surface,
                                    r["kaisai_date"], cf1_hist_idx)
        cf1_kaisai_late = CF1.kaisai_late_flag(rid)
        cf2 = CF2.compute_for_race(df, r["race_name"], r["kaisai_date"], cf1_hist_idx)
        cf2_kaisai_day_num = CF2.kaisai_day_num(rid)
        cf3 = CF3.compute_for_race(df, rid, r["kaisai_date"], cf1_racecourse, cf1_surface, cf3_ctx,
                                    distance_m=cf1_distance)
        cf3_race = cf3["_race_level"]

        # 期待値(build_artifact.pyのEV計算式を踏襲、レビューでの既知バイアス1点(EVのscore
        # 比例正規化)をJSON notesに明記する)。オッズはrace_results.odds_final(確定値)を
        # 優先し、無い馬だけbias_win_oddsへフォールバックする(上記2026-09-13追記参照)。
        umaban = JS._num(JS._col(df, "umaban"))
        odds_bias = JS._num(JS._col(df, "bias_win_odds"))
        odds_final_map = odds_final_by_race.get(rid)
        if odds_final_map:
            odds_rf = umaban.map(odds_final_map)
            odds = odds_rf.where(odds_rf.notna(), odds_bias)
        else:
            odds = odds_bias
        ev_score_sum = float(score.fillna(0.0).sum())
        p_ev = (score / ev_score_sum) if ev_score_sum > 0 else pd.Series(np.nan, index=df.index)
        ev_pct = (odds * p_ev * 100).where((p_ev > 0) & odds.notna())

        ninki = JS._num(JS._col(df, "bias_ninki"))
        mark_honshi = JS._col(df, "mark_honshi")
        mark_cp = JS._col(df, "mark_cp")
        mark_other = JS._col(df, "mark_other")
        waku = JS._num(JS._col(df, "waku"))

        horses = []
        for i in df.index:
            horses.append({
                "umaban": int(umaban.at[i]) if pd.notna(umaban.at[i]) else None,
                "waku": int(waku.at[i]) if pd.notna(waku.at[i]) else None,
                "horse_name": str(df.at[i, "horse_name"]) if "horse_name" in df.columns else None,
                "race_type": rtype,
                "pred_rank": int(pred_rank.at[i]),
                "pred_rank_box4": int(pred_rank_box4.at[i]) if pred_rank_box4 is not None else None,
                "pred_rank_box3": int(pred_rank_box3.at[i]) if pred_rank_box3 is not None else None,
                "radar_rank_ability": _rank_val(radar_ranks["ability"], i),
                "radar_rank_form": _rank_val(radar_ranks["form"], i),
                "radar_rank_style": _rank_val(radar_ranks["style"], i),
                "radar_rank_aptitude": _rank_val(radar_ranks["aptitude"], i),
                "radar_rank_pedigree": _rank_val(radar_ranks["pedigree"], i),
                "radar_rank_jt": _rank_val(radar_ranks["jt"], i),
                "radar_rank_mark": _rank_val(radar_ranks["mark"], i),
                # --- レーダー細分化フィルタ(2026-09-14追加) ---
                "radar_rank_style_position": _rank_val(sub_radar_ranks["style_position"], i),
                "radar_rank_style_stamina": _rank_val(sub_radar_ranks["style_stamina"], i),
                "radar_rank_style_corner_move": _rank_val(sub_radar_ranks["style_corner_move"], i),
                "radar_rank_jt_base": _rank_val(sub_radar_ranks["jt_base"], i),
                "radar_rank_jt_change": _rank_val(sub_radar_ranks["jt_change"], i),
                "radar_rank_jt_stats": _rank_val(sub_radar_ranks["jt_stats"], i),
                "radar_rank_pedigree_pure": _rank_val(sub_radar_ranks["pedigree_pure"], i),
                "radar_rank_pedigree_training": _rank_val(sub_radar_ranks["pedigree_training"], i),
                "radar_rank_form_agari": _rank_val(sub_radar_ranks["form_agari"], i),
                "lap33_fit_rank": _rank_val(radar_ranks["lap33"], i),
                "flop_safety_rank": _rank_val(flop_safety_rank, i),
                "mark_honshi": _mark_val(mark_honshi.at[i]),
                "mark_cp": _mark_val(mark_cp.at[i]),
                "mark_other": _mark_val(mark_other.at[i]),
                "bias_ninki": int(ninki.at[i]) if pd.notna(ninki.at[i]) else None,
                "ev_pct": round(float(ev_pct.at[i]), 1) if pd.notna(ev_pct.at[i]) else None,
                # --- 新規候補ファクターv1(2026-09-01) ---
                "career_place_rate": cf1["career_place_rate"].at[i],
                "distance_band_place_rate": cf1["distance_band_place_rate"].at[i],
                "turn_apt_place_rate": cf1["turn_apt_place_rate"].at[i],
                "course_size_apt_place_rate": cf1["course_size_apt_place_rate"].at[i],
                "slope_apt_place_rate": cf1["slope_apt_place_rate"].at[i],
                "turf_type_apt_place_rate": cf1["turf_type_apt_place_rate"].at[i],
                "grade_best": cf1["grade_best"].at[i],
                "first_time_flag": cf1["first_time_flag"].at[i],
                "jockey_switch": cf1["jockey_switch"].at[i],
                "trainer_change": cf1["trainer_change"].at[i],
                "stable_multi_entry": cf1["stable_multi_entry"].at[i],
                "kaisai_late": cf1_kaisai_late,
                # --- 新規候補ファクターv2(2026-09-02) ---
                "class_form_place_rate": cf2["class_form_place_rate"].at[i],
                "class_challenge_flag": cf2["class_challenge_flag"].at[i],
                "layoff_flag": cf2["layoff_flag"].at[i],
                "second_after_layoff_flag": cf2["second_after_layoff_flag"].at[i],
                "main_jockey_return_flag": cf2["main_jockey_return_flag"].at[i],
                "bad_run_popularity_drop_flag": cf2["bad_run_popularity_drop_flag"].at[i],
                "kaisai_day_num": cf2_kaisai_day_num,
                # --- 新規候補ファクターv3(2026-09-04): 騎手/調教師/馬主/生産者プロフィール ---
                "jockey_lead_rank": cf3["jockey_lead_rank"].at[i],
                "jockey_area": cf3["jockey_area"].at[i],
                "jockey_affiliation_type": cf3["jockey_affiliation_type"].at[i],
                "jockey_career_wins_band": cf3["jockey_career_wins_band"].at[i],
                "trainer_lead_rank": cf3["trainer_lead_rank"].at[i],
                "trainer_area": cf3["trainer_area"].at[i],
                "transport_flag": cf3["transport_flag"].at[i],
                "owner_prior_win_rate": cf3["owner_prior_win_rate"].at[i],
                "breeder_group": cf3["breeder_group"].at[i],
                # --- v3: コース形態・馬場(レース単位、全馬同値) ---
                "surface": cf3_race["surface"],
                "course_turn": cf3_race["course_turn"],
                "course_turn_type": cf3_race["course_turn_type"],
                "course_straight_band": cf3_race["course_straight_band"],
                "course_elevation_band": cf3_race["course_elevation_band"],
                "course_size_band": cf3_race["course_size_band"],
                "cushion_band": cf3_race["cushion_band"],
                "moisture_band": cf3_race["moisture_band"],
                "turf_course_variant": cf3_race["turf_course_variant"],
                "race_pace_label": cf3_race["race_pace_label"],
                # --- v3: 既存df列だが未露出だった基本条件 ---
                "horse_sex": cf3["horse_sex"].at[i],
                "horse_age": cf3["horse_age"].at[i],
                "weight_carried_band": cf3["weight_carried_band"].at[i],
                "horse_weight_band": cf3["horse_weight_band"].at[i],
                "horse_weight_diff_band": cf3["horse_weight_diff_band"].at[i],
                "odds_band": cf3["odds_band"].at[i],
                "interval_band": cf3["interval_band"].at[i],
                "corner4_expected_rank": cf3["corner4_expected_rank"].at[i],
                "past1_finish_band": cf3["past1_finish_band"].at[i],
                "training_course_type": cf3["training_course_type"].at[i],
                "training_track_condition": cf3["training_track_condition"].at[i],
                "training_best_time_flag": cf3["training_best_time_flag"].at[i],
                "ca_jockey_win_rate": cf3["ca_jockey_win_rate"].at[i],
                "ca_trainer_win_rate": cf3["ca_trainer_win_rate"].at[i],
            })

        out_races.append({
            "race_id": rid, "kaisai_date": r["kaisai_date"], "racecourse": r["racecourse"],
            "race_name": r["race_name"], "race_type": rtype,
            "block_id": f'{r["kaisai_date"]}_{r["racecourse"]}',
            "offered_bet_types": offered.get(rid, []),
            "horses": horses,
        })

    print("payoutsを数値配列形式へ変換中...")
    out_payouts = {}
    for r in races:
        rid = r["race_id"]
        per_bt = actual.get(rid, {})
        row = {}
        for bt, combos in per_bt.items():
            entries = []
            for combo, payout in combos.items():
                if isinstance(combo, int):
                    arr = [combo]
                elif isinstance(combo, tuple):
                    arr = list(combo)  # 着順通り、ソートしない(馬単/3連単)
                else:  # frozenset
                    arr = sorted(combo)
                entries.append({"combo": arr, "payout": int(payout)})
            if entries:
                row[bt] = entries
        out_payouts[rid] = row

    dates = sorted({r["kaisai_date"] for r in races})
    n_blocks = len({f'{r["kaisai_date"]}_{r["racecourse"]}' for r in races})
    out = {
        "generated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
        "population": {
            "normal": {"n_races": pop_counts["normal"]},
            "shinba": {"n_races": pop_counts["shinba"]},
            "mishoubi": {"n_races": pop_counts["mishoubi"]},
            "total": {"n_races": len(races)},
            "date_range": [dates[0], dates[-1]] if dates else [],
            "n_dates": len(dates), "n_blocks": n_blocks,
        },
        "notes": {
            "odds_accuracy": "2026-09-13〜: 単勝オッズはrace_results.odds_final(確定値)を"
                            "優先し、無い馬(出走取消等)だけ取得時点値(bias_win_odds)へ"
                            "フォールバックします(旧: 全件bias_win_odds、約25%のレースで"
                            "最終確定オッズと乖離していた既知バイアスはこれで解消済み)。",
            "ev_bias": "期待値(EV)はscoreをレース内で比例正規化した簡易勝率×単勝オッズです。"
                      "scoreの散らばりは実際の勝率分布ほど急峻でないため、人気馬の勝率を"
                      "過小に、穴馬の勝率を過大に見積もる傾向があります。",
            "model_reliability": "新馬戦(54レース)・未勝利戦(195レース)の重みは通常戦"
                                "(242レース)より少ない母集団で決定されています。特に新馬戦は"
                                "単一シグナル(jt)に約89%集中する解が採用されており、"
                                "小標本への過学習の可能性が否定できません。",
            "selection_bias_warning": "ファクターは自由に組み合わせられます。良さそうに見える"
                                     "組み合わせは事後選択バイアスの産物である可能性が高く、"
                                     "95%信頼区間がブレークイーブン(回収率100%)をまたぐ場合は"
                                     "「差がある」と解釈しないでください。過去にも同種の自由探索"
                                     "(NAR300パターン・JRA BOX4/3・レーダー面積8,892パターン"
                                     "総当たり探索)が撤回・不採用となった実績があります。",
        },
        "bet_types": JD.BET_TYPES,
        "takeout_rates_pct": JE.TAKEOUT_RATES,
        "group_tiers": GROUP_TIERS,
        "factor_groups": FACTOR_GROUPS,
        "races": out_races,
        "payouts": out_payouts,
    }

    print("NaN混入チェック中(pandas Seriesの暗黙型強制でNoneがnp.nanへ化けると、JSON非準拠の"
          "NaNトークンが混入しブラウザのJSON.parseがクラッシュするため、書き出し直前に全件サニタイズする)...")
    out = _sanitize_nan(out)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, ensure_ascii=False, separators=(",", ":"), default=str)
    n_bad = text.count(":NaN,") + text.count(":NaN}")
    if n_bad:
        raise SystemExit(f"サニタイズ後もNaNトークンが{n_bad}件残っています。要調査。")
    OUT_JSON.write_text(text, encoding="utf-8")
    print(f"wrote {OUT_JSON} ({len(text):,} bytes)")


def _sanitize_nan(obj):
    """dict/list/floatを再帰的に走査し、NaN(np.nan・float('nan')いずれも)をNoneへ置換する。
    2026-09-02、jra_candidate_factors_v1追加時に発覚した実際のバグ(pandas Seriesが
    Noneを含むリストをfloat64へ暗黙型強制しNaN化 → json.dumpsが非準拠のNaNトークンを出力 →
    ブラウザのJSON.parseがクラッシュ、実測2,903件混入)の再発を防ぐ最終防波堤。"""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj


def _rank_val(rank_series: pd.Series, idx):
    v = rank_series.at[idx] if idx in rank_series.index else np.nan
    return int(v) if pd.notna(v) else None


def _mark_val(v) -> str | None:
    """予想印セルをJSON安全な文字列/Noneへ変換する。pandas 2.x系の`str`拡張dtypeでは
    Series.astype(str)が欠損値を文字列'nan'化せず生のfloat NaNのまま残すため(json.dumpsが
    そのままJSON非準拠の`NaN`トークンを出力し、ブラウザ側JSON.parseがクラッシュするバグの
    原因だった)、スカラー単位でpd.isna()判定してからstr()化する。"""
    if pd.isna(v):
        return None
    s = str(v).strip()
    return s if s not in ("", "nan") else None


if __name__ == "__main__":
    main()
