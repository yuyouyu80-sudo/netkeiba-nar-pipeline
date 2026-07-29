import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


def fetch(session: requests.Session, url: str, encoding: str | None = None, delay_seconds: float = 1.5) -> str:
    """Fetch a URL and return decoded HTML text.

    encoding: netkeiba serves different subdomains with different charsets
    (confirmed: race.netkeiba.com is UTF-8, db.netkeiba.com is EUC-JP, and
    db.netkeiba.com's Content-Type header omits the charset param, so
    requests can't auto-detect it). Pass it explicitly per known host.
    """
    time.sleep(delay_seconds)
    response = session.get(url)
    response.raise_for_status()
    if encoding:
        response.encoding = encoding
    return response.text


def fetch_json(session: requests.Session, url: str, data: dict, delay_seconds: float = 1.5) -> Any:
    """POSTs form-encoded `data` and returns the parsed JSON body. Used for
    race.netkeiba.com/race_api/ endpoints (riot.js tag pages like
    holding_time.html render client-side from these instead of server HTML)."""
    time.sleep(delay_seconds)
    response = session.post(url, data=data)
    response.raise_for_status()
    return response.json()
