"""www.jra.go.jp/kouza/win5/result.html(過去のキャリーオーバー履歴)取得。
jra_official_baba.pyと同じくnetkeiba外・ログイン不要ページ。"""
import time

import requests

from config.settings import REQUEST_DELAY_SECONDS
from src.netkeiba_pipeline.scrapers.base import DEFAULT_TIMEOUT_SECONDS

_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
_URL = "https://www.jra.go.jp/kouza/win5/result.html"


def fetch_win5_carryover_history_html() -> str:
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})
    time.sleep(REQUEST_DELAY_SECONDS)
    response = session.get(_URL, timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    response.encoding = "shift_jis"
    return response.text
