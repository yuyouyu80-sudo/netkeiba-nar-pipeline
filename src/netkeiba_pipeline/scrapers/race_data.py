import requests

from src.netkeiba_pipeline.discovery.tracks import race_url
from src.netkeiba_pipeline.scrapers.base import fetch
from config.settings import REQUEST_DELAY_SECONDS

# mode=concerned cid values under data_list.html - cid=0 is the "同場同距離"
# (same venue/surface/distance as today's race) tab.
CONCERNED_CID_LABELS = {
    0: "concerned",
}

# mode= values under data.html ("項目別データ"), each showing one row per
# horse broken into a fixed number of category sub-rows (see
# race_data_parser.parse_data_breakdown for the row shapes).
DATA_BREAKDOWN_MODES = {
    "distance": 4,
    "course": 4,
    "condition": 4,
    "others": 4,
    "cushion": 5,
    "baba_water": 4,
}

# NAR's per-mode row counts differ from JRA's (confirmed against real 高知/盛岡
# race pages): "distance" shows one extra nearby-distance row, "others" one
# fewer, and "cushion"/"baba_water" collapse to a much smaller, largely
# degenerate shape - NAR racing is dirt-only, so turf-firmness/moisture
# breakdowns have little to differentiate. course/condition match JRA exactly.
NAR_DATA_BREAKDOWN_MODES = {
    "distance": 5,
    "course": 4,
    "condition": 4,
    "others": 3,
    "cushion": 2,
    "baba_water": 2,
}


def fetch_concerned_html(session: requests.Session, race_id: str, cid: int = 0) -> str:
    url = race_url(race_id, "data_list.html", mode="concerned", cid=cid)
    return fetch(session, url, encoding="utf-8", delay_seconds=REQUEST_DELAY_SECONDS)


def fetch_data_breakdown_html(session: requests.Session, race_id: str, mode: str) -> str:
    url = race_url(race_id, "data.html", mode=mode)
    return fetch(session, url, encoding="utf-8", delay_seconds=REQUEST_DELAY_SECONDS)
