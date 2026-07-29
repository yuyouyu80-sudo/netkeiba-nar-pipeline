import requests

from src.netkeiba_pipeline.discovery.tracks import race_url
from src.netkeiba_pipeline.scrapers.base import fetch
from config.settings import REQUEST_DELAY_SECONDS

# cid values under mode=courseanalysis, confirmed against the live "コース"
# tab menu on race.netkeiba.com/race/data_list.html. Note cid=1/2/3 require a
# premium ("プレミアムコース") subscription - without it, netkeiba silently
# truncates the result table to 3 rows and shows a Premium_Regist_Box
# up-sell instead of an error, so a non-premium account will get incomplete
# data rather than a clear failure.
CID_LABELS = {
    0: "waku",
    1: "running_style",
    2: "jockey",
    3: "trainer",
}


def fetch_course_analysis_html(session: requests.Session, race_id: str, cid: int) -> str:
    url = race_url(race_id, "data_list.html", mode="courseanalysis", cid=cid)
    return fetch(session, url, encoding="utf-8", delay_seconds=REQUEST_DELAY_SECONDS)
