import base64
import json
import zlib

import requests

from src.netkeiba_pipeline.scrapers.base import fetch_json
from config.settings import RACE_API_URL, REQUEST_DELAY_SECONDS


def fetch_holding_time_data(session: requests.Session, race_id: str) -> dict:
    """holding_time.html (持ちタイム) is a riot.js tag page: the server HTML
    has no data, it's rendered client-side from this JSON API. `class:
    AplFreqSum` is the same call the page's own JS (holdingtime.min.js)
    makes; `compress: 0` is meant to skip the zlib+base64 encoding the JS
    would otherwise have to inflate client-side, but netkeiba's server
    ignores that flag intermittently (observed on ~1 in 5 requests, no
    discernible pattern by race) and sends the compressed form anyway - so
    the compressed case is always handled as a fallback rather than trusted
    to not happen."""
    data = {
        "input": "UTF-8",
        "output": "json",
        "class": "AplFreqSum",
        "method": "get",
        "compress": "0",
        "race_id": race_id,
    }
    payload = fetch_json(session, RACE_API_URL, data, delay_seconds=REQUEST_DELAY_SECONDS)
    if payload.get("status") != "OK":
        raise ValueError(
            f"race_id={race_id}: AplFreqSum API returned status={payload.get('status')} "
            f"reason={payload.get('reason')}"
        )
    key = f"nkrace_freq_sum::{race_id}"
    if key not in payload.get("data", {}):
        raise ValueError(f"race_id={race_id}: {key} not found in AplFreqSum response - API structure may have changed")

    value = payload["data"][key]
    if isinstance(value, str):
        value = json.loads(zlib.decompress(base64.b64decode(value)))
    return value
