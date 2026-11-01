"""デザリングモード(2026-10-04〜): 従量・低速回線(スマホのテザリング等)で長時間バッチを回すとき、
(1)直近24時間(ローリング)の通信量を上限DAILY_LIMIT_BYTES以内に抑え、(2)低速(リクエスト間隔を
MIN_INTERVAL_SECONDS以上)で取得する。

有効化: 各スクリプトの `--tether` フラグ(fetch_pedigree.py / fetch_horse_profile.py)、または環境変数
`NETKEIBA_TETHER=1`(フラグを持たない全スクリプトに効く)。無効時はこのモジュールは何もしない。

仕組み:
- login()が作るsessionにresponseフックを付け、ログイン・リダイレクト・リトライを含む全レスポンスの
  バイト数を logs/tether_usage.tsv(epoch秒<TAB>バイト数、追記のみ)へ記録する。
- scrapers/base.pyが各リクエストの前に wait_for_budget() を呼び、直近24時間の使用量+次の1件の想定量が
  上限を超えるなら、古い記録が24時間窓から外れるまで待つ(プロセスは終了せず待機、再開不要)。
- リクエスト間隔は throttle() が MIN_INTERVAL_SECONDS 以上に引き上げる。

計上するのはこのプロジェクトのnetkeiba向けHTTPの本文+ヘッダ概算(×OVERHEAD_FACTORでTLS等を加味)のみ。
tether有効にしていない別プロセス(watch_odds等)やPC全体の通信は含まない。同じログを共有するので、
tether有効なプロセスが複数あっても合計で上限内に収まる。

確認: python -m src.netkeiba_pipeline.utils.tether  (直近24時間の使用量を表示)
"""
import logging
import os
import time
from pathlib import Path

from config.settings import LOG_DIR

logger = logging.getLogger(__name__)

DAILY_LIMIT_BYTES = 1_000_000_000  # 1GB(10進、GiBより小さい=安全側)
WINDOW_SECONDS = 24 * 3600
OVERHEAD_FACTOR = 1.05  # TLS・TCP・リクエストヘッダ分の概算
HEADER_BYTES_ESTIMATE = 800  # レスポンスヘッダ+リクエスト分の概算(本文以外)
NEXT_REQUEST_ESTIMATE_BYTES = 150_000  # 次の1件の想定量(血統約100KB・プロフィール約83KBより大きめ)
DEFAULT_MIN_INTERVAL_SECONDS = 8.0  # 通常の約1.5〜2倍以上遅い。1日約10,800件=血統でほぼ上限と釣り合う間隔
MAX_SLEEP_CHUNK_SECONDS = 300

USAGE_LOG = LOG_DIR / "tether_usage.tsv"

# 環境変数で別枠の使用量ログ・上限を指定できる(フラグを持たないスクリプトを別枠で回す用。
# 例: NETKEIBA_TETHER=1 NETKEIBA_TETHER_LOG=tether_usage_nar.tsv NETKEIBA_TETHER_LIMIT_MB=500)
if os.environ.get("NETKEIBA_TETHER_LOG"):
    USAGE_LOG = LOG_DIR / os.environ["NETKEIBA_TETHER_LOG"]
try:
    if os.environ.get("NETKEIBA_TETHER_LIMIT_MB"):
        DAILY_LIMIT_BYTES = int(float(os.environ["NETKEIBA_TETHER_LIMIT_MB"]) * 1_000_000)
except ValueError:
    pass

_enabled = False
_last_request_at = 0.0


def enable() -> None:
    global _enabled
    _enabled = True
    logger.warning(
        "TETHER MODE ON: 24h limit %.2f GB, min interval %.1fs, usage log %s",
        DAILY_LIMIT_BYTES / 1e9, min_interval(), USAGE_LOG,
    )


def is_enabled() -> bool:
    return _enabled or os.environ.get("NETKEIBA_TETHER", "") not in ("", "0")


def min_interval() -> float:
    try:
        return float(os.environ.get("NETKEIBA_TETHER_INTERVAL", DEFAULT_MIN_INTERVAL_SECONDS))
    except ValueError:
        return DEFAULT_MIN_INTERVAL_SECONDS


def _read_entries(now: float) -> list[tuple[float, int]]:
    """24時間窓内の(epoch, bytes)。壊れた行は無視する。"""
    if not USAGE_LOG.exists():
        return []
    entries = []
    with open(USAGE_LOG, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            try:
                ts, n = float(parts[0]), int(parts[1])
            except ValueError:
                continue
            if now - ts < WINDOW_SECONDS:
                entries.append((ts, n))
    return entries


def used_bytes_24h(now: float | None = None) -> int:
    now = time.time() if now is None else now
    return sum(n for _, n in _read_entries(now))


def record(nbytes: int) -> None:
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(USAGE_LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.time():.3f}\t{int(nbytes * OVERHEAD_FACTOR)}\n")


def _on_response(response, *args, **kwargs) -> None:
    try:
        length = response.headers.get("Content-Length")
        body = int(length) if length and length.isdigit() else len(response.content)
        record(body + HEADER_BYTES_ESTIMATE)
    except Exception:  # noqa: BLE001 - 計上の失敗で取得本体を止めない
        logger.debug("tether: failed to record usage", exc_info=True)


def attach(session) -> None:
    """sessionの全レスポンス(リダイレクト・リトライ含む)の通信量を記録するフックを付ける。"""
    if is_enabled():
        session.hooks["response"].append(_on_response)


def seconds_until_budget(now: float, needed: int = NEXT_REQUEST_ESTIMATE_BYTES) -> float:
    """次の1件(neededバイト)を送っても上限内に収まるまでの待ち秒数(収まるなら0)。"""
    entries = _read_entries(now)
    used = sum(n for _, n in entries)
    if used + needed <= DAILY_LIMIT_BYTES:
        return 0.0
    over = used + needed - DAILY_LIMIT_BYTES
    freed = 0
    for ts, n in sorted(entries):
        freed += n
        if freed >= over:
            return max(0.0, ts + WINDOW_SECONDS - now) + 1.0
    return WINDOW_SECONDS


def wait_for_budget() -> None:
    """上限に達していれば、24時間窓から古い記録が外れるまで待つ。無効時は何もしない。"""
    if not is_enabled():
        return
    while True:
        now = time.time()
        wait = seconds_until_budget(now)
        if wait <= 0:
            return
        logger.warning(
            "TETHER: 24h usage %.1f MB reached the %.0f MB limit - waiting %.0f min for budget to free up",
            used_bytes_24h(now) / 1e6, DAILY_LIMIT_BYTES / 1e6, wait / 60,
        )
        time.sleep(min(wait, MAX_SLEEP_CHUNK_SECONDS))


def throttle(delay_seconds: float) -> None:
    """通常の間隔待ち。デザリングモード時は MIN_INTERVAL_SECONDS 以上に引き上げる。"""
    global _last_request_at
    delay = max(delay_seconds, min_interval()) if is_enabled() else delay_seconds
    if is_enabled():
        delay = max(0.0, delay - (time.time() - _last_request_at))  # 直前リクエストからの経過を差し引く
    time.sleep(delay)
    _last_request_at = time.time()


def summary() -> str:
    used = used_bytes_24h()
    return f"tether usage (last 24h): {used / 1e6:.1f} MB / {DAILY_LIMIT_BYTES / 1e6:.0f} MB ({used / DAILY_LIMIT_BYTES:.1%})"


if __name__ == "__main__":
    print(summary())
