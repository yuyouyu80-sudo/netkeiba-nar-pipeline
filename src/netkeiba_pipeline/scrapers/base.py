import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# session.get()/post()にtimeoutを渡さないと、サーバー側が接続を張ったまま無応答になった場合に
# リクエストが無期限にハングしうる(2026-08-27、3コーナー位置バックフィル中に実際に52分間
# ハングして発覚)。(接続確立, レスポンス受信)の2値タプルで指定。
DEFAULT_TIMEOUT_SECONDS = (10, 30)


def fetch(session: requests.Session, url: str, encoding: str | None = None, delay_seconds: float = 1.5) -> str:
    """Fetch a URL and return decoded HTML text.

    encoding: netkeiba serves different subdomains with different charsets
    (confirmed: race.netkeiba.com is UTF-8, db.netkeiba.com is EUC-JP, and
    db.netkeiba.com's Content-Type header omits the charset param, so
    requests can't auto-detect it). Pass it explicitly per known host.
    """
    time.sleep(delay_seconds)
    response = session.get(url, timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    if encoding:
        response.encoding = encoding
    return response.text


def fetch_json(session: requests.Session, url: str, data: dict, delay_seconds: float = 1.5) -> Any:
    """POSTs form-encoded `data` and returns the parsed JSON body. Used for
    race.netkeiba.com/race_api/ endpoints (riot.js tag pages like
    holding_time.html render client-side from these instead of server HTML)."""
    time.sleep(delay_seconds)
    response = session.post(url, data=data, timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def fetch_json_get(session: requests.Session, url: str, params: dict, delay_seconds: float = 1.5) -> Any:
    """GETs `params` as a query string and returns the parsed JSON body. Used for
    race.netkeiba.com/api/ endpoints (distinct from race_api/ above - GET+query-string
    instead of POST+form-data. Confirmed for api_get_jra_odds.html, the live odds
    widget's underlying data source)."""
    time.sleep(delay_seconds)
    response = session.get(url, params=params, timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()
