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
