import logging
import os
import pickle
import time
from pathlib import Path

import requests
import urllib3

from src.netkeiba_pipeline.utils import tether

# 2026-08-27: ネットワーク経路上のFortinetファイアウォールがTLSインスペクション(自己署名
# 証明書への差し替え)を行っており、通常の証明書検証ではSSLErrorになる環境がある。
# ユーザーの明示的な了承のもと、このプロジェクトのnetKeibaアクセスに限定して証明書検証を
# 無効化している(session.verify = False、下記login()内)。この状態では経路上の別のMITMも
# 無条件に信頼してしまうため、安全なネットワークに戻った場合は元に戻すことが望ましい。
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LOGIN_URL = "https://regist.netkeiba.com/"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
AUTH_COOKIE_NAME = "nkauth"

logger = logging.getLogger(__name__)


def login(email: str, password: str) -> requests.Session:
    session = requests.Session()
    session.verify = False  # 2026-08-27: 上記コメント参照(Fortinet TLSインスペクション対応)
    session.headers.update({"User-Agent": USER_AGENT})
    tether.attach(session)  # デザリングモード時のみ: ログイン含む全通信量を記録

    # 環境変数 NETKEIBA_SESSION_FILE を指定した時だけ、保存済みCookieがあればログインせずに使い回す
    # (スクリプトを何本も続けて回す長時間バッチで、毎回ログインしない用。未指定なら従来どおり毎回ログイン)。
    # 有効期限内のnkauthが無ければ通常どおりログインして、そのCookieを保存する。
    session_file = os.environ.get("NETKEIBA_SESSION_FILE")
    if session_file:
        path = Path(session_file)
        if load_cookies(session, path) and _auth_cookie_alive(session):
            logger.info("Reusing saved netkeiba session from %s (no login request).", path)
            return session
        session.cookies.clear()

    payload = {
        "pid": "login",
        "action": "auth",
        "rtn_url": "",
        "login_id": email,
        "pswd": password,
    }
    # timeout無しだと、回線が不安定な時にログインのPOSTが無期限に止まる(2026-10-20、調教の取得が2時間止まった)
    response = session.post(LOGIN_URL, data=payload, timeout=(10, 30))
    response.raise_for_status()

    if not is_logged_in(session):
        raise RuntimeError(
            "Login request completed but the session does not look authenticated. "
            "Check credentials in .env, and inspect the login form fields via "
            "browser DevTools in case netkeiba changed them."
        )

    logger.info("Logged in to netkeiba successfully.")
    if session_file:
        save_cookies(session, Path(session_file))
    return session


def _auth_cookie_alive(session: requests.Session) -> bool:
    """nkauthがあり、有効期限が(あれば)1時間以上先なら使い回せるとみなす。"""
    for cookie in session.cookies:
        if cookie.name == AUTH_COOKIE_NAME:
            return cookie.expires is None or cookie.expires - time.time() > 3600
    return False


def is_logged_in(session: requests.Session) -> bool:
    """The login POST redirects through netkeiba's shared SSO
    (account.sp.findfriends.jp) and, on success, sets an 'nkauth' cookie on
    the session. Confirmed by manual inspection: regist.netkeiba.com's own
    mypage_top endpoint returns an empty body regardless of auth state, so
    it can't be used as a signal; the auth cookie itself is the reliable one."""
    return any(cookie.name == AUTH_COOKIE_NAME for cookie in session.cookies)


def save_cookies(session: requests.Session, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(session.cookies, f)


def load_cookies(session: requests.Session, path: Path) -> bool:
    if not path.exists():
        return False
    with open(path, "rb") as f:
        session.cookies.update(pickle.load(f))
    return True
