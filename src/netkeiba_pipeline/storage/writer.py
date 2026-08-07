import csv
from datetime import datetime, timezone

import pandas as pd

from config.settings import MANIFEST_PATH
from src.netkeiba_pipeline.storage.paths import (
    course_analysis_csv_path,
    course_ranking_csv_path,
    lap_times_csv_path,
    payout_csv_path,
    race_result_csv_path,
)

MANIFEST_COLUMNS = ["race_id", "data_type", "scraped_at", "status"]


def is_already_scraped(race_id: str, data_type: str = "race_result") -> bool:
    if not MANIFEST_PATH.exists():
        return False
    df = pd.read_csv(MANIFEST_PATH, dtype=str)
    match = df[(df["race_id"] == race_id) & (df["data_type"] == data_type) & (df["status"] == "success")]
    return len(match) > 0


def write_race_result(df: pd.DataFrame, kaisai_date: str, race_id: str, circuit: str = "jra") -> None:
    """Idempotent per race_id: drops any existing rows for this race_id before
    appending, so a crash-and-retry (before the manifest is updated) never
    produces duplicate rows."""
    path = race_result_csv_path(kaisai_date, circuit)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = df.astype(str)
    if path.exists():
        existing = pd.read_csv(path, dtype=str)
        existing = existing[existing["race_id"] != race_id]
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    combined.to_csv(path, index=False, encoding="utf-8")


def write_course_analysis(df: pd.DataFrame, race_id: str, category_type: str) -> None:
    """Idempotent per (race_id, category_type): drops existing rows for this
    category_type before appending, same rationale as write_race_result."""
    path = course_analysis_csv_path(race_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = df.astype(str)
    if path.exists():
        existing = pd.read_csv(path, dtype=str)
        existing = existing[existing["category_type"] != category_type]
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    combined.to_csv(path, index=False, encoding="utf-8")


def write_course_ranking(df: pd.DataFrame, race_id: str, ranking_type: str) -> None:
    """Idempotent per (race_id, ranking_type), same rationale as write_race_result."""
    path = course_ranking_csv_path(race_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = df.astype(str)
    if path.exists():
        existing = pd.read_csv(path, dtype=str)
        existing = existing[existing["ranking_type"] != ranking_type]
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    combined.to_csv(path, index=False, encoding="utf-8")


def write_payouts(df: pd.DataFrame, kaisai_date: str, race_id: str, circuit: str = "jra") -> None:
    """Idempotent per race_id, same rationale as write_race_result."""
    path = payout_csv_path(kaisai_date, circuit)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = df.astype(str)
    if path.exists():
        existing = pd.read_csv(path, dtype=str)
        existing = existing[existing["race_id"] != race_id]
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    combined.to_csv(path, index=False, encoding="utf-8")


def write_lap_times(df: pd.DataFrame, kaisai_date: str, race_id: str, circuit: str = "jra") -> None:
    """Idempotent per race_id, same rationale as write_race_result. df may be empty
    (race has no published lap-time table, e.g. some NAR races) - an empty df for a
    race_id still clears out any stale rows for it from a previous run."""
    path = lap_times_csv_path(kaisai_date, circuit)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = df.astype(str)
    if path.exists():
        existing = pd.read_csv(path, dtype=str)
        existing = existing[existing["race_id"] != race_id]
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    combined.to_csv(path, index=False, encoding="utf-8")


def mark_scraped(race_id: str, data_type: str = "race_result", status: str = "success") -> None:
    """Call only after the corresponding write_* call has completed, so the
    manifest never claims success for data that wasn't actually written."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "race_id": race_id,
        "data_type": data_type,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
    }
    file_exists = MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
