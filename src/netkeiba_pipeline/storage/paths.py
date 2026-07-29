from pathlib import Path

from config.settings import COURSE_ANALYSIS_DIR, COURSE_RANKING_DIR, NEWSPAPER_DIR, PAYOUTS_DIR, RACE_RESULTS_DIR
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
