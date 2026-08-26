"""mark_list.html(本紙・CP・その他 予想印)の取得だけを行う。

このページの印テーブルはページ読み込み後にJavaScriptがDOM操作で描画するため、requestsでは
空のテンプレートしか取得できない(確認済み、メモリ
project_netkeiba_yosoin_marks_scraping_method_2026_08_26参照)。そのためこのプロジェクトで
唯一Playwrightを使うスクレイパーになる。既存のrequestsベースのlogin()で得たCookieを
Playwrightのブラウザコンテキストに引き継いで認証状態を再現する。

ブラウザ起動はコストが高いため(--dateでの一括実行が遅くなる)、呼び出し側が
create_authenticated_context()で1回だけ起動したcontextを使い回し、fetch_mark_list_htmlは
レースごとにpageを開いて閉じるだけにする。
"""
import time

import requests
from playwright.sync_api import Browser, BrowserContext, Playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from src.netkeiba_pipeline.discovery.tracks import race_site_domain

PAGE_LOAD_DELAY_SECONDS = 1.5


def _to_playwright_cookies(session: requests.Session) -> list[dict]:
    """requests.Session.cookies(http.cookiejar.Cookie)をPlaywrightの
    context.add_cookies()が要求する形式に変換する。"""
    cookies = []
    for cookie in session.cookies:
        cookies.append(
            {
                "name": cookie.name,
                "value": cookie.value,
                "domain": cookie.domain,
                "path": cookie.path or "/",
                "expires": cookie.expires if cookie.expires else -1,
                "secure": bool(cookie.secure),
            }
        )
    return cookies


def create_authenticated_context(pw: Playwright, session: requests.Session) -> tuple[Browser, BrowserContext]:
    """既存のrequests.Session(login()済み)のCookieを引き継いだ、認証済みの
    Playwrightブラウザコンテキストを作る。呼び出し側は使い終わったら
    browser.close()すること(contextはbrowserと一緒に閉じる)。"""
    browser = pw.chromium.launch(headless=True)
    context = browser.new_context()
    context.add_cookies(_to_playwright_cookies(session))
    return browser, context


def fetch_mark_list_html(context: BrowserContext, race_id: str) -> str:
    """race_id: 12桁。JRA/NARはrace_site_domain()で自動判別する(NAR側でも実データで
    ページが存在し印が取得できることを確認済み、2026-08-26)。

    `wait_until="networkidle"`は使わない: 実データ検証(2026-08-26、1開催日36レース中2レース)で、
    特定のrace_idのmark_list.htmlがnetworkidleに到達しないまま45秒タイムアウトすることを複数回
    (別プロセスでの単独再実行でも同じrace_idで再現)確認した。netkeiba側の広告/トラッキング関連
    と見られる継続的な通信が原因と推測され、ページ自体の取得や印テーブルの描画は正常に完了して
    いる。そのため`domcontentloaded`まで待ってから、印テーブルの出現を直接待つ方式に変更した。
    印テーブルが元々存在しないレース(専門家印が1人も掲載されていない日)ではこの待機も
    タイムアウトするが、それはパース側で空DataFrameとして正しく扱われるので許容する。"""
    domain = race_site_domain(race_id)
    url = f"https://{domain}/yoso/mark_list.html?race_id={race_id}&rf=race_submenu"

    page = context.new_page()
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        try:
            page.wait_for_selector("dl.Yosoka[id]", timeout=15000)
        except PlaywrightTimeoutError:
            pass
        time.sleep(PAGE_LOAD_DELAY_SECONDS)
        return page.content()
    finally:
        page.close()
