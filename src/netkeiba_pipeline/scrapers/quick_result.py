import requests

from src.netkeiba_pipeline.discovery.tracks import race_url
from src.netkeiba_pipeline.scrapers.base import fetch
from config.settings import REQUEST_DELAY_SECONDS


def fetch_quick_result_html(session: requests.Session, race_id: str) -> str:
    """nar.netkeiba.com/race/result.html: 同日中に確定する簡易結果ページ。
    db.netkeiba.com(race_result、run_pilot.pyが使う正式な確定データ)は翌日反映のため、
    発走直後の速報確認にはこちらを使う。race_urlがNAR race_idを自動判別してnar.netkeiba.com
    に切り替える。"""
    url = race_url(race_id, "result.html")
    return fetch(session, url, encoding="utf-8", delay_seconds=REQUEST_DELAY_SECONDS)
