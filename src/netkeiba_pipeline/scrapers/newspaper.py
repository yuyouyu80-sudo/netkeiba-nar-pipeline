import requests

from src.netkeiba_pipeline.discovery.tracks import race_url
from src.netkeiba_pipeline.scrapers.base import fetch
from config.settings import REQUEST_DELAY_SECONDS


def fetch_newspaper_html(session: requests.Session, race_id: str) -> str:
    url = race_url(race_id, "newspaper.html", rf="shutuba_submenu")
    return fetch(session, url, encoding="utf-8", delay_seconds=REQUEST_DELAY_SECONDS)
