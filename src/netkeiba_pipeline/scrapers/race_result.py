import requests

from src.netkeiba_pipeline.scrapers.base import fetch
from config.settings import RACE_RESULT_URL, REQUEST_DELAY_SECONDS


def fetch_race_result_html(session: requests.Session, race_id: str) -> str:
    url = RACE_RESULT_URL.format(race_id=race_id)
    return fetch(session, url, encoding="euc-jp", delay_seconds=REQUEST_DELAY_SECONDS)
