"""通常戦(pattern29)モデルで検証待ち日付の予想を生成する。

対象はJRA/NAR開催日のうち、まだレース結果(payout/finish)が確定していない日付。
netkeibaから当日のnewspaper(馬柱)ページを読み直してレース見出し情報(レース名・
競馬場・馬場・距離・発走時刻)を取得し、race_names_{date}.csvとして書き出した上で、
predictions_{date}.csvを生成する。
新馬戦・未勝利戦は対象外(専用モデルがあるが、検証待ち日付ではまだ対応していない)。

JRA分岐(既定)のシグナル計算・重み合成は scripts/jra_model/jra_signals.py(単一の真実の源)
に委譲し、重み・priorsは data/jra_pipeline/winner_v3.json から読む。scratchpad/predict.py・
predict_box4.py・predict_box3.py も同じjra_signals.py/winner_v3.jsonを使うため、探索・検証・
本番の3経路は必ず同じ値でスコアリングする(2026-08-11、NAR側で過去に発生した「探索側と
本番側でpriorsが食い違う」事故を構造的に防ぐための統合)。--circuit nar のスコアリングは
このファイル内の従来ロジック(WEIGHTS_NAR等)のまま無変更。

--circuit nar 指定時は race_names_nar_{date}.csv / predictions_nar_{date}.csv に
出力する(JRA分と衝突させないため)。重み・事前値・クラス序列はNAR専用のものに
差し替わるが、--circuit省略時(JRA)の挙動・出力は一切変更しない。

JRA(--circuit省略時)のみ、当日レポートの全頭表示用にpredictions_full_{date}.csv
(box検証に使うtop5版のpredictions_{date}.csvとは別に、全頭分をpred_rank付きで
書き出したもの)もあわせて出力する。box_return等の検証系スクリプトは従来通り
predictions_{date}.csv(top5)だけを見るため、過去の検証結果には影響しない。

前提: scripts/fetch_newspaper.py --date {date} [--circuit nar] で当日のnewspaper
データが取得済みであること(未取得のレースはmissingとして報告されるのみで、
エラー終了はしない)。
"""
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "jra_model"))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import jra_signals as JS  # noqa: E402 - JRA分岐の重み合成は単一の真実の源(jra_signals.py)に委譲
from config.settings import LOG_DIR
from src.netkeiba_pipeline.auth.session import login
from src.netkeiba_pipeline.discovery.race_calendar import list_nar_race_ids, list_race_ids
from src.netkeiba_pipeline.parsers.race_header_parser import parse_race_header
from src.netkeiba_pipeline.scrapers.newspaper import fetch_newspaper_html
from src.netkeiba_pipeline.storage.paths import newspaper_csv_path
from src.netkeiba_pipeline.utils.logging_conf import configure_logging

# このレポート生成パイプライン(build_artifact.py)専用の出力先。検証済み日付の
# predictions.csv等と同じscratchpadディレクトリに揃える(JRA分、--circuit省略時)。
OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
# --circuit nar 用の恒久的な出力先(クラウド/リモート実行環境でも参照できるよう、
# セッション固有のscratchpadではなくGit管理下のdata/nar_pipelineに書き出す)。
NAR_OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "nar_pipeline"

# JRA分岐の重み・priors・クラス序列は data/jra_pipeline/winner_v3.json を単一の真実の源とする
# (2026-08-11、NARのwinner_*.json方式に統一。探索側(scripts/jra_model/jra_search_*.py)・
# 本番側(このファイル)・scratchpad側(predict.py/predict_box4.py/predict_box3.py)が必ず
# 同じ値を読むため、NAR側で過去に起きた「探索側と本番側でpriorsが食い違う」事故を構造的に防ぐ)。
_WINNER_PATH = Path(__file__).resolve().parent.parent / "data" / "jra_pipeline" / "winner_v3.json"
_winner = json.loads(_WINNER_PATH.read_text(encoding="utf-8"))

TRAIN_RANK_MAP = {"S": 6, "A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
DNF_FINISH_PENALTY = 20
DNF_CODES = {"中止", "取消", "除外", "失格", "中", "取", "除"}
CLASS_ORDINAL = _winner["class_ordinal"]
CLASS_ADJUST_PER_LEVEL = 1.5
RUNNING_STYLE_FRONT = {"逃", "先"}
RUNNING_STYLE_CLOSE = {"差", "追"}
SHRINK_K = 12.0

# pattern29本番の重み・縮約事前値はdata/jra_pipeline/winner_v3.jsonから読む(上記参照)。
# 2026-07-27: search_patterns_v4.py(検証済み6開催日105レース、search=7/11+7/12+7/18・
# holdout=7/19+7/25+7/26)で再探索したpattern83が現行値(winner_v3.json参照)。旧pattern29
# (70レース時点)と比較し、同一holdoutで平均回収率110.0%→122.5%(全105レース)、
# 111.8%→136.4%(holdoutのみ)に改善したことを確認済み(compare_weights_v4.py、詳細は
# winner_v3.jsonのhistorical_headline参照)。2026-08-11、177レースへのデータ増加を受けて
# scripts/jra_model/jra_search_2026_08_11.pyで再検証・候補シグナル探索に着手。
WEIGHTS = _winner["weights"]
_PRIORS = _winner["priors"]
# 以下のSHRINK_SPECSは--circuit nar(WEIGHTS_NAR/_PRIORS_NAR、無変更)のscore_race()が
# 引き続き使用する。JRA分岐(既定)のスコアリングはjra_signals.score_race()に委譲しており、
# そちらは自身のSHRINK_SPECS(候補シグナル分を含む)を持つため、この辞書はJRA分岐では
# 使われない。
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
}

# --circuit nar 用。地方競馬はJRAの新馬/未勝利→1勝→...という勝ち上がり制ではなく
# C3/C2/C1・B2/B1・A2/A1・重賞というクラス別編成のため、JRAのCLASS_ORDINALをそのまま
# 使うと全レースがNaN(未分類)になる。"C1C2"のような複合クラス表記は"C1"/"C2"より
# 先に判定する必要がある(_class_ordinalは部分一致・先勝ちのため)。
CLASS_ORDINAL_NAR = {
    "新馬": 0,
    "C1C2": 2.5,
    "C3": 1, "C2": 2, "C1": 3,
    "B2": 4, "B1": 5,
    "A2": 6, "A1": 7,
    "重賞": 8,
}
# search_patterns_nar.py(scratchpad)による重み探索の結果(winner_v3_nar.json,
# pattern 24, search=20260725/holdout=20260726, 全38レース中holdout 28レースで
# floor_pass 8/8・min_sharpe -0.269で選定)。NARは現状40レース(1日+1日)しか
# 検証済みデータがなく、JRAのpattern29(70レース)より一段と薄いsearch/holdout分割
# での選定である点に注意。今後NARの検証済み日付が増えたらsearch_patterns_nar.pyを
# 再実行し、この値を差し替えること。
WEIGHTS_NAR = {
    "speed": 0.07513405757633125, "form": 0.09188612523403819, "style": 0.034868601927349,
    "jt": 0.21752914669983245, "waku": 0.11780058409195925, "apt": 0.04910816245438971,
    "train": 0.07635476476036503, "distance": 0.059845478748454556, "sire": 0.09798821415043366,
    "bms": 0.17948486435684696,
}
# apt_win: NAR newspaperにはca_speed_index_win_rate/ca_speed_index_runs列自体が
# 存在しないため常にNaN(train/comment同様、地方競馬で構造的に欠測)。
# bms_win/place3/return: 列は存在するが実データはほぼ全馬runs=0(0%)で、優勝馬の
# 母父コース適性統計が地方競馬では極めて薄い。事前値0.0はこのサンプルでの実測値。
_PRIORS_NAR = {
    "style_win": 10.562666666666667, "style_place3": 31.869333333333334,
    "jockey_win": 9.904, "trainer_win": 10.573333333333334,
    "waku_win": 9.866666666666667, "apt_win": float("nan"),
    "distance_win": 10.995466666666665, "distance_place3": 30.190133333333332,
    "distance_return": 60.64053333333332,
    "sire_win": 11.114666666666666, "sire_place3": 32.70933333333333,
    "sire_return": 70.56266666666667,
    "bms_win": 0.0, "bms_place3": 0.0, "bms_return": 0.0,
}


def _class_ordinal(text) -> float:
    if pd.isna(text):
        return np.nan
    text = str(text).strip()
    for key, val in CLASS_ORDINAL.items():
        if key in text:
            return val
    return np.nan


def _num(series: pd.Series) -> pd.Series:
    cleaned = series.where(~series.astype(str).isin(["-", "--", "nan", ""]), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def _pct(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace("%", "", regex=False)
    cleaned = cleaned.where(~cleaned.isin(["-", "--", "nan", ""]), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


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


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index)


def _drop_scratched(df: pd.DataFrame) -> pd.DataFrame:
    odds = _num(df["bias_win_odds"])
    ninki = _num(df["bias_ninki"])
    return df[odds.notna() & ninki.notna()].reset_index(drop=True)


def _shrink(df: pd.DataFrame, key: str) -> pd.Series:
    rate_col, runs_col = SHRINK_SPECS[key]
    rate = _pct(_col(df, rate_col))
    runs = _num(_col(df, runs_col)).fillna(0.0)
    prior = _PRIORS[key]
    return (rate.fillna(prior) * runs + prior * SHRINK_K) / (runs + SHRINK_K)


def score_race(df: pd.DataFrame, current_class_ordinal: float) -> pd.DataFrame:
    df = df.copy()

    speed_cols = ["speed_max_index", "speed_avg_index_5races", "speed_index_1race_ago"]
    speed_avg = df[speed_cols].apply(_num).mean(axis=1, skipna=True)
    sig_speed = _minmax(speed_avg)

    finishes = {}
    for i in (1, 2, 3):
        raw = _finish_with_dnf_penalty(df[f"past{i}_finish"])
        past_class = df[f"past{i}_race_class"].map(_class_ordinal)
        class_gap = current_class_ordinal - past_class
        adjustment = (class_gap * CLASS_ADJUST_PER_LEVEL).where(class_gap.notna(), 0.0)
        finishes[i] = raw - adjustment
    past_finish_df = pd.DataFrame(finishes)
    weights_arr = np.array([3, 2, 1])
    weighted_sum = (past_finish_df.fillna(0) * weights_arr).sum(axis=1)
    weight_total = (past_finish_df.notna() * weights_arr).sum(axis=1).replace(0, np.nan)
    recent_form = weighted_sum / weight_total
    sig_form = _minmax(-recent_form)

    style_rate = _blend_minmax(_shrink(df, "style_win"), _shrink(df, "style_place3"))
    style_label = df["ca_running_style_category_label"].astype(str).str.strip()
    front_count = style_label.isin(RUNNING_STYLE_FRONT).sum()
    field_n = max(len(df), 1)
    pace_pressure = front_count / field_n
    pace_direction = style_label.map(
        lambda s: 1.0 if s in RUNNING_STYLE_CLOSE else (-1.0 if s in RUNNING_STYLE_FRONT else 0.0)
    )
    pace_adjustment = pace_direction * (pace_pressure - 0.35)
    sig_style = _minmax(style_rate.fillna(0.5) + pace_adjustment)

    jt_rate = pd.concat([_shrink(df, "jockey_win"), _shrink(df, "trainer_win")], axis=1).mean(axis=1)
    sig_jt = _minmax(jt_rate)

    sig_waku = _minmax(_shrink(df, "waku_win"))
    sig_apt = _minmax(_shrink(df, "apt_win"))

    training = df["training_rank"].astype(str).str.strip().str.upper().map(TRAIN_RANK_MAP)
    sig_train = _minmax(training)

    sig_distance = _blend_minmax(
        _shrink(df, "distance_win"), _shrink(df, "distance_place3"), _shrink(df, "distance_return")
    )
    sig_sire = _blend_minmax(
        _shrink(df, "sire_win"), _shrink(df, "sire_place3"), _shrink(df, "sire_return")
    )
    sig_bms = _blend_minmax(
        _shrink(df, "bms_win"), _shrink(df, "bms_place3"), _shrink(df, "bms_return")
    )

    weighted_cols = {
        "speed": (sig_speed, WEIGHTS["speed"]), "form": (sig_form, WEIGHTS["form"]),
        "style": (sig_style, WEIGHTS["style"]), "jt": (sig_jt, WEIGHTS["jt"]),
        "waku": (sig_waku, WEIGHTS["waku"]), "apt": (sig_apt, WEIGHTS["apt"]),
        "train": (sig_train, WEIGHTS["train"]), "distance": (sig_distance, WEIGHTS["distance"]),
        "sire": (sig_sire, WEIGHTS["sire"]), "bms": (sig_bms, WEIGHTS["bms"]),
    }

    total_score = pd.Series(0.0, index=df.index)
    total_weight = pd.Series(0.0, index=df.index)
    for _name, (sig, w) in weighted_cols.items():
        avail = sig.notna()
        total_score = total_score + sig.fillna(0) * w
        total_weight = total_weight + avail.astype(float) * w

    df["_score"] = np.where(total_weight > 0, total_score / total_weight, np.nan)
    return df


def main() -> None:
    global CLASS_ORDINAL, WEIGHTS, _PRIORS

    parser = argparse.ArgumentParser(
        description="通常戦(pattern29)モデルで指定日(検証待ち)の予想を生成する。"
        "race_names_{date}.csv / predictions_{date}.csv をscratchpadに書き出す。"
    )
    parser.add_argument("--date", required=True, help="kaisai_date (YYYYMMDD)")
    parser.add_argument("--circuit", choices=["jra", "nar"], default="jra", help="開催区分(既定: jra)")
    args = parser.parse_args()

    if args.circuit == "nar":
        CLASS_ORDINAL = CLASS_ORDINAL_NAR
        WEIGHTS = WEIGHTS_NAR
        _PRIORS = _PRIORS_NAR

    load_dotenv()
    email = os.environ.get("NETKEIBA_EMAIL")
    password = os.environ.get("NETKEIBA_PASSWORD")
    if not email or not password:
        raise SystemExit(
            "NETKEIBA_EMAIL / NETKEIBA_PASSWORD not set. Copy .env.example to .env "
            "and fill them in yourself (never paste real credentials into chat)."
        )

    configure_logging(LOG_DIR / f"predict_pattern29_{args.date}.log")
    logger = logging.getLogger("predict_pattern29")

    session = login(email, password)
    race_ids = list_nar_race_ids(session, args.date) if args.circuit == "nar" else list_race_ids(session, args.date)
    logger.info("Found %d race_ids for %s (circuit=%s)", len(race_ids), args.date, args.circuit)

    race_name_rows = []
    targets = []
    header_errors = []
    for race_id in race_ids:
        try:
            html = fetch_newspaper_html(session, race_id)
            header = parse_race_header(html, race_id)
        except Exception:
            logger.exception("Failed to read race header for race_id=%s", race_id)
            header_errors.append(race_id)
            continue
        race_name_rows.append({"kaisai_date": args.date, **header})
        if not re.search("新馬|未勝利", header["race_name"]):
            targets.append({"kaisai_date": args.date, **header})

    if not race_name_rows:
        raise SystemExit(f"no race header could be read for {args.date} (see log for details)")

    suffix = "_nar" if args.circuit == "nar" else ""
    out_dir = NAR_OUT_DIR if args.circuit == "nar" else OUT_DIR
    race_names_df = pd.DataFrame(race_name_rows)
    race_names_out = out_dir / f"race_names{suffix}_{args.date}.csv"
    race_names_df.to_csv(race_names_out, index=False, encoding="utf-8-sig")
    logger.info("wrote %s (%d races)", race_names_out, len(race_names_df))

    all_predictions = []
    all_predictions_full = []
    missing = []
    errored = []
    for row in targets:
        race_id = row["race_id"]
        path = newspaper_csv_path(race_id)
        if not path.exists():
            missing.append(race_id)
            continue
        try:
            df = pd.read_csv(path, dtype=str, encoding="utf-8")
            if df.empty:
                missing.append(race_id)
                continue
            df = _drop_scratched(df)
            if df.empty:
                missing.append(race_id)
                continue

            current_class = _class_ordinal(row["race_name"])
            if args.circuit == "nar":
                scored = score_race(df, current_class)  # NAR分岐: 無変更(WEIGHTS_NAR等)
            else:
                scored = JS.score_race(df, current_class, weights=WEIGHTS, priors=_PRIORS,
                                       class_ordinal_map=CLASS_ORDINAL)
            scored = scored.sort_values("_score", ascending=False, kind="stable").reset_index(drop=True)
            top5 = scored.head(5).copy()
            top5["race_id"] = race_id
            top5.insert(0, "pred_rank", range(1, len(top5) + 1))
            top5.insert(0, "kaisai_date", row["kaisai_date"])
            top5.insert(1, "racecourse", row["racecourse"])
            top5.insert(2, "race_name", row["race_name"])
            all_predictions.append(
                top5[["kaisai_date", "racecourse", "race_name", "race_id", "pred_rank", "waku", "umaban",
                      "horse_name", "bias_ninki", "bias_win_odds", "bias_horse_weight", "_score"]]
            )
            if args.circuit != "nar":
                full = scored.copy()
                full["race_id"] = race_id
                full.insert(0, "pred_rank", range(1, len(full) + 1))
                full.insert(0, "kaisai_date", row["kaisai_date"])
                full.insert(1, "racecourse", row["racecourse"])
                full.insert(2, "race_name", row["race_name"])
                all_predictions_full.append(
                    full[["kaisai_date", "racecourse", "race_name", "race_id", "pred_rank", "waku", "umaban",
                          "horse_name", "bias_ninki", "bias_win_odds", "bias_horse_weight", "_score"]]
                )
        except Exception as exc:  # noqa: BLE001 - one bad race must not abort the whole batch
            errored.append((race_id, repr(exc)))
            continue

    print(f"race headers written: {len(race_name_rows)} (header read errors: {len(header_errors)} -> {header_errors})")

    if not all_predictions:
        print(f"no predictions produced (missing newspaper csv: {len(missing)} -> {missing}, "
              f"errored: {len(errored)} -> {errored})")
        return

    result = pd.concat(all_predictions, ignore_index=True)
    predictions_out = out_dir / f"predictions{suffix}_{args.date}.csv"
    result.to_csv(predictions_out, index=False, encoding="utf-8-sig")
    print(f"target races: {len(targets)}")
    print(f"predicted races: {result['race_id'].nunique()}")
    print(f"missing newspaper csv: {len(missing)} -> {missing}")
    print(f"errored races: {len(errored)} -> {errored}")
    print(f"wrote {predictions_out}")

    if args.circuit != "nar" and all_predictions_full:
        result_full = pd.concat(all_predictions_full, ignore_index=True)
        predictions_full_out = out_dir / f"predictions_full_{args.date}.csv"
        result_full.to_csv(predictions_full_out, index=False, encoding="utf-8-sig")
        print(f"wrote {predictions_full_out} (全頭, {result_full['race_id'].nunique()} races)")


if __name__ == "__main__":
    main()
