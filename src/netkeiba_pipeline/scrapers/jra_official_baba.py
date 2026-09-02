"""www.jra.go.jp(JRA公式サイト、netkeibaとは別ドメイン)のクッション値・含水率
アーカイブPDF取得。

**このプロジェクト唯一のnetkeiba外スクレイピング対象**(予想ファクター充足度マップ
Tier3、2026-09-02実装)。他のsrc/netkeiba_pipeline/scrapers/配下のモジュールと違い、
`src.netkeiba_pipeline.auth.session.login()`(netkeiba専用SSO・TLS検証無効化)は
一切使わない。robots.txt(全許可、2026-09-02確認済み)に従い、素の`requests`セッション
(User-Agent設定のみ)でアクセスする。

実データ確認(2026-09-02)で判明した重要な制約: このPDFは「予想時点(レース前)の速報値」
**ではない**。`/keiba/baba/`配下の他ページ(condition/cushion/moist/)は数値データを
一切持たない説明用ページ(用語解説の参考表のみ)で、実数値は開催回(「n回○○」、
複数の週にまたがるレース開催ブロック)が**完全に終了してから**公開される
(実データ確認: 進行中の開催回のPDFは403、完了済みの開催回は200)。
したがって本モジュールが取得するデータは事後の傾向分析専用であり、日次の予想
パイプラインには使えない(充足度マップ本文の想定より遅い)。

配置場所についての注記: 当初のロードマップでは「src/jra_official_pipeline/のような
別名前空間」を検討するとしていたが、実装してみると新規ファイルは本スクレイパーと
対応パーサー(jra_baba_parser.py)の2つのみで、既存のsrc/netkeiba_pipeline/scrapers/
・parsers/にファイル単位で「netkeiba外」であることをdocstringで明記する形で
配置する方が実用的と判断した(パッケージ全体を分けるほどの規模ではない)。"""
import time

import requests

from config.settings import JRA_OFFICIAL_BABA_PDF_URL, REQUEST_DELAY_SECONDS
from src.netkeiba_pipeline.scrapers.base import DEFAULT_TIMEOUT_SECONDS

_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def fetch_baba_pdf(year: str, venue_slug: str, kai: int) -> bytes:
    """クッション値・含水率アーカイブPDFの生バイト列を取得する。

    year: 'YYYY'。venue_slug: JRA_BABA_PDF_VENUE_SLUGSの値(例: "niigata")。
    kai: 開催回(整数、URL側で2桁ゼロ埋め)。

    ログイン不要ページのため`requests.Session()`を都度生成する(netkeiba用の
    ログイン済みセッションを引き回す既存パターンとは意図的に異なる)。開催回が
    まだ完全終了していない場合は403が返る(モジュールdocstring参照) - これは
    呼び出し側が`requests.exceptions.HTTPError`としてキャッチし、「まだ未公開」
    として扱うべきエラーであり、リトライしても解消しない。"""
    url = JRA_OFFICIAL_BABA_PDF_URL.format(year=year, venue=venue_slug, kai=kai)
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})

    time.sleep(REQUEST_DELAY_SECONDS)
    response = session.get(url, timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.content
