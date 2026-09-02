from pathlib import Path

import pandas as pd

from config.settings import (
    COURSE_ANALYSIS_DIR,
    COURSE_RANKING_DIR,
    HORSE_PROFILE_DIR,
    JOCKEY_PROFILE_DIR,
    LAP_TIMES_DIR,
    NEWSPAPER_DIR,
    PAYOUTS_DIR,
    PEDIGREE_DIR,
    RACE_RESULTS_DIR,
    TRAINER_PROFILE_DIR,
)
from src.netkeiba_pipeline.discovery.tracks import is_nar_race


def race_result_csv_path(kaisai_date: str, circuit: str = "jra") -> Path:
    """kaisai_date: 'YYYYMMDD'. circuit: 'jra' (default, unprefixed path - unchanged
    from before NAR support existed) or 'nar' (data/race_results/nar/...). A single
    kaisai_date can have both JRA and NAR racing, so circuit can't be derived from
    kaisai_date alone and must be passed explicitly by the caller."""
    year = kaisai_date[:4]
    base = RACE_RESULTS_DIR if circuit == "jra" else RACE_RESULTS_DIR / circuit
    return base / year / f"{kaisai_date}.csv"


def payout_csv_path(kaisai_date: str, circuit: str = "jra") -> Path:
    """kaisai_date: 'YYYYMMDD'. See race_result_csv_path for the circuit param."""
    year = kaisai_date[:4]
    base = PAYOUTS_DIR if circuit == "jra" else PAYOUTS_DIR / circuit
    return base / year / f"{kaisai_date}.csv"


def lap_times_csv_path(kaisai_date: str, circuit: str = "jra") -> Path:
    """kaisai_date: 'YYYYMMDD'. See race_result_csv_path for the circuit param."""
    year = kaisai_date[:4]
    base = LAP_TIMES_DIR if circuit == "jra" else LAP_TIMES_DIR / circuit
    return base / year / f"{kaisai_date}.csv"


def course_analysis_csv_path(race_id: str) -> Path:
    """race_id alone determines circuit (unlike kaisai_date), so it's derived here
    rather than taken as a parameter."""
    base = COURSE_ANALYSIS_DIR / "nar" if is_nar_race(race_id) else COURSE_ANALYSIS_DIR
    return base / f"{race_id}.csv"


def course_ranking_csv_path(race_id: str) -> Path:
    base = COURSE_RANKING_DIR / "nar" if is_nar_race(race_id) else COURSE_RANKING_DIR
    return base / f"{race_id}.csv"


def newspaper_csv_path(race_id: str) -> Path:
    base = NEWSPAPER_DIR / "nar" if is_nar_race(race_id) else NEWSPAPER_DIR
    return base / f"{race_id}.csv"


def pedigree_csv_path(horse_id: str) -> Path:
    """horse_id単位(circuit/race_idに非依存、JRA/NARで共有)。1ファイル1頭・1行。
    既存の「取得済みか」判定は呼び出し側で`path.exists()`を見るだけでよく(全体読込不要)、
    複数レースを跨いだ同一馬の重複取得も自然に防げる(詳細はfetch_pedigree.py参照)。"""
    return PEDIGREE_DIR / f"{horse_id}.csv"


def load_all_pedigree() -> pd.DataFrame:
    """data/pedigree/*.csv(horse_id単位ファイル群)を一括読み込みして1つのDataFrameに
    まとめる分析用ヘルパー。horse_id単位ファイルは書き込み時にO(1)のexists()チェックで
    済むよう分割している代わりに、分析側で一括読込の不便さを吸収する(全列dtype=str)。
    ディレクトリが無い/1件も無い場合はhorse_id列だけの空DataFrameを返す。"""
    if not PEDIGREE_DIR.exists():
        return pd.DataFrame(columns=["horse_id"])
    paths = sorted(PEDIGREE_DIR.glob("*.csv"))
    if not paths:
        return pd.DataFrame(columns=["horse_id"])
    return pd.concat([pd.read_csv(p, dtype=str, encoding="utf-8") for p in paths], ignore_index=True)


def horse_profile_csv_path(horse_id: str) -> Path:
    """horse_id単位(circuit/race_idに非依存)。生産者・馬主は不変性が高いため
    pedigree_csv_pathと同じ「存在すれば恒久スキップ」キャッシュ方式で運用する
    (詳細はfetch_horse_profile.py参照)。"""
    return HORSE_PROFILE_DIR / f"{horse_id}.csv"


def jockey_profile_csv_path(jockey_id: str) -> Path:
    """jockey_id単位。年度別成績等は時系列で変化するため、pedigree/horse_profileとは
    異なり毎回上書き(恒久キャッシュにしない)。詳細はfetch_person_profile.py参照。"""
    return JOCKEY_PROFILE_DIR / f"{jockey_id}.csv"


def trainer_profile_csv_path(trainer_id: str) -> Path:
    """trainer_id単位。jockey_profile_csv_pathと同じく毎回上書き。"""
    return TRAINER_PROFILE_DIR / f"{trainer_id}.csv"
