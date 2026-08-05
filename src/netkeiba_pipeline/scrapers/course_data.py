import requests

from src.netkeiba_pipeline.discovery.tracks import race_url
from src.netkeiba_pipeline.scrapers.base import fetch
from config.settings import REQUEST_DELAY_SECONDS

# cid values under mode=coursedata, confirmed against the "コース" tab menu
# (distinct from mode=courseanalysis's own cid numbering in course_analysis.py).
COURSEDATA_CID_LABELS = {
    1: "sire",
    4: "broodmare_sire",
}

# NOTE(2026-08-04): cid=4(broodmare_sire) returns a real table#table_sort_back for
# NAR races (no Premium_Regist_Box, no parse error) but every row's category_label
# and all stat cells are blank/"0"/"0%", for every horse regardless of actual dam
# sire - confirmed by fetching a live NAR race (202630080401) and comparing cell-by-
# cell against cid=1 (sire, which returns real per-horse data on the same race) and
# against a JRA race's cid=4 page (which returns real non-zero data). This is not a
# scraper/parser bug - netkeiba genuinely does not publish broodmare-sire aggregate
# stats for NAR horses. Same category as speed/apt/train (see fetch_newspaper.py
# and nar_signals.py KNOWN_DEAD); nar_signals.py's "bms" signal is expected to stay
# structurally dead for NAR.


def fetch_course_data_html(session: requests.Session, race_id: str, cid: int) -> str:
    url = race_url(race_id, "data_list.html", mode="coursedata", cid=cid)
    return fetch(session, url, encoding="utf-8", delay_seconds=REQUEST_DELAY_SECONDS)


def fetch_surf_summary_html(
    session: requests.Session,
    race_id: str,
    range_: int = 4,
    key1: str = "SpeedIdxScore",
    key2: str | None = None,
) -> str:
    params = {"range": range_, "key1": key1}
    if key2:
        params["key2"] = key2
    url = race_url(race_id, "surf_summary.html", **params)
    return fetch(session, url, encoding="utf-8", delay_seconds=REQUEST_DELAY_SECONDS)
