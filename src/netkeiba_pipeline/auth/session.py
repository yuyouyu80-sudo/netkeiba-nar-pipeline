import logging
import pickle
from pathlib import Path

import requests

LOGIN_URL = "https://regist.netkeiba.com/"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
AUTH_COOKIE_NAME = "nkauth"

logger = logging.getLogger(__name__)


def login(email: str, password: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    payload = {
        "pid": "login",
        "action": "auth",
        "rtn_url": "",
        "login_id": email,
        "pswd": password,
    }
    response = session.post(LOGIN_URL, data=payload)
    response.raise_for_status()

    if not is_logged_in(session):
        raise RuntimeError(
            "Login request completed but the session does not look authenticated. "
            "Check credentials in .env, and inspect the login form fields via "
            "browser DevTools in case netkeiba changed them."
        )

    logger.info("Logged in to netkeiba successfully.")
    return session


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
