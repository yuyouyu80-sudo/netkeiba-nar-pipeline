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
import json
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


def _night_window() -> tuple[int, int] | None:
    """夜間だけ別の設定にする時間帯(時, 時)。環境変数 NETKEIBA_NIGHT_HOURS="0-7" は0時以上7時未満(ローカル時刻)。
    未指定なら夜間設定は使わない。"""
    try:
        start, end = os.environ.get("NETKEIBA_NIGHT_HOURS", "").split("-")
        return int(start), int(end)
    except ValueError:
        return None


def _is_night(now: float | None = None) -> bool:
    window = _night_window()
    if window is None:
        return False
    hour = time.localtime(time.time() if now is None else now).tm_hour
    start, end = window
    return start <= hour < end if start < end else hour >= start or hour < end


def min_interval() -> float:
    name = "NETKEIBA_NIGHT_INTERVAL" if _is_night() and os.environ.get("NETKEIBA_NIGHT_INTERVAL") else "NETKEIBA_TETHER_INTERVAL"
    try:
        return float(os.environ.get(name, DEFAULT_MIN_INTERVAL_SECONDS))
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


def _break_rules() -> list[tuple[float, float]]:
    """休憩の規則[(稼働秒, 休む秒), ...]。環境変数 NETKEIBA_BREAK_RULES="10:1,30:3" は
    「稼働10分ごとに1分、稼働30分ごとに3分休む」(同時に当たる時は長い方の休憩だけ取る)。
    NETKEIBA_BREAK_EVERY_MIN / NETKEIBA_BREAK_MIN の組は規則1つとして扱う。"""
    rules = []
    rules_env = os.environ.get("NETKEIBA_BREAK_RULES", "")
    if _is_night() and os.environ.get("NETKEIBA_NIGHT_BREAK_RULES"):
        rules_env = os.environ["NETKEIBA_NIGHT_BREAK_RULES"]
    try:
        for part in rules_env.split(","):
            if part.strip():
                run_m, rest_m = part.split(":")
                rules.append((float(run_m) * 60, float(rest_m) * 60))
        if not rules:
            run_m = float(os.environ.get("NETKEIBA_BREAK_EVERY_MIN", "0"))
            rest_m = float(os.environ.get("NETKEIBA_BREAK_MIN", "0"))
            if run_m > 0 and rest_m > 0:
                rules.append((run_m * 60, rest_m * 60))
    except ValueError:
        return []
    return [(r, w) for r, w in rules if r > 0 and w > 0]


def _maybe_take_break() -> None:
    """休憩の規則(_break_rules)に従い、稼働時間が規則の周期に達したら休む。稼働時間の累計は
    プロセスをまたいで logs/tether_break_state.json に持つ(スクリプトを何本も続けて回しても周期が続く)。
    前回のリクエストから10分以上空いた分(スクリプト間の処理・中断など)は稼働に数えない。"""
    rules = _break_rules()
    if not rules:
        return
    state = LOG_DIR / "tether_break_state.json"
    now = time.time()
    try:
        st = json.loads(state.read_text())
        worked, last_ts, done = float(st["worked"]), float(st["last_ts"]), list(st["done"])
        if len(done) != len(rules):
            raise ValueError
        if st.get("rules") != [r for r, _ in rules]:  # 昼夜で周期が変わった: 今の稼働時間までは休憩済みとして数える
            done = [int(worked // r) for r, _ in rules]
    except (OSError, ValueError, KeyError, TypeError):
        worked, last_ts, done = 0.0, now, [0] * len(rules)
    gap = now - last_ts
    if 0 <= gap < 600:
        worked += gap
    due_rest = 0.0
    for i, (run_s, rest_s) in enumerate(rules):
        idx = int(worked // run_s)
        if idx > done[i]:
            done[i] = idx
            due_rest = max(due_rest, rest_s)
    if due_rest > 0:
        logger.warning("BREAK: %.1f min of fetching done - resting %.1f min", worked / 60, due_rest / 60)
        time.sleep(due_rest)
        now = time.time()
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"worked": worked, "last_ts": now, "done": done, "rules": [r for r, _ in rules]}))


def throttle(delay_seconds: float) -> None:
    """通常の間隔待ち。デザリングモード時は MIN_INTERVAL_SECONDS 以上に引き上げる。"""
    global _last_request_at
    if is_enabled():
        _maybe_take_break()
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
