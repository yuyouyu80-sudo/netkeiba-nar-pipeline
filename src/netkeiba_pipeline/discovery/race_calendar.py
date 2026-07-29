import re

import requests
from bs4 import BeautifulSoup

from src.netkeiba_pipeline.discovery.tracks import NAR_TRACK_CODES
from src.netkeiba_pipeline.scrapers.base import fetch
from config.settings import RACE_LIST_SUB_URL, REQUEST_DELAY_SECONDS

RACE_ID_RE = re.compile(r"race_id=(\d{12})")
NAR_RACE_LIST_SUB_URL = "https://nar.netkeiba.com/top/race_list_sub.html"


def _extract_race_ids(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    race_ids: list[str] = []
    seen = set()
    for a in soup.find_all("a", href=True):
        match = RACE_ID_RE.search(a["href"])
        if match and match.group(1) not in seen:
            seen.add(match.group(1))
            race_ids.append(match.group(1))
    return race_ids


def list_race_ids(session: requests.Session, kaisai_date: str) -> list[str]:
    """kaisai_date: 'YYYYMMDD'. Returns JRA race_ids in page order, deduplicated."""
    url = f"{RACE_LIST_SUB_URL}?kaisai_date={kaisai_date}"
    html = fetch(session, url, encoding="utf-8", delay_seconds=REQUEST_DELAY_SECONDS)
    race_ids = _extract_race_ids(html)

    if not race_ids:
        raise ValueError(
            f"No race_ids found for kaisai_date={kaisai_date}. Either there was no "
            "racing on this date, or netkeiba changed the race_list_sub.html structure."
        )
    return race_ids


def list_nar_race_ids(session: requests.Session, kaisai_date: str) -> list[str]:
    """kaisai_date: 'YYYYMMDD'. Returns race_ids for the 14 tracks in NAR_TRACK_CODES
    only (page order, deduplicated) - other nar.netkeiba.com race_ids (e.g. 帯広(ば)
    banei racing) are filtered out."""
    url = f"{NAR_RACE_LIST_SUB_URL}?kaisai_date={kaisai_date}"
    html = fetch(session, url, encoding="utf-8", delay_seconds=REQUEST_DELAY_SECONDS)
    race_ids = [rid for rid in _extract_race_ids(html) if rid[4:6] in NAR_TRACK_CODES]

    if not race_ids:
        raise ValueError(
            f"No NAR race_ids found for kaisai_date={kaisai_date} among the tracked 14 "
            "venues. Either none of them raced on this date, or netkeiba changed the "
            "race_list_sub.html structure."
        )
    return race_ids
