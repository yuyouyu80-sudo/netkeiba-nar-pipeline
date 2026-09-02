"""単勝オッズの時系列(発走前11時点+確定後1時点)を取得する常駐ウォッチャー。

各レースの発走時刻(race_names_{date}.csv、predict_pattern29.py既出力)を基準に、
1時間前/30分前/20分前/10分前/5分前/4分前/3分前/2分前/1分30秒前/1分前/30秒前、および
発走10分後(確定オッズ狙い)の計12時点でrace.netkeiba.com/race/bias.html(単勝オッズ・
人気を含む軽量な出馬表ページ。1リクエストで全頭分のオッズが取得できるため、レースごとに
馬数分リクエストする必要がない)を取得し、odds_history_{date}.csv に逐次追記する。

このスクリプトはClaude Codeのセッションに依存しない独立プロセスとして、ユーザー自身が
その日の間起動したままにしておく前提で設計している。cron/ScheduleWakeupは分単位が限界で
「30秒前」等の秒単位の指定を正確には実現できないため、自前のtime.sleepループで発走時刻から
逆算した時刻まで待つ方式にした。

使い方:
    python scripts/watch_odds.py --date 20260726                 # 本番: 1日分を監視して自動取得
    python scripts/watch_odds.py --date 20260726 --print-schedule  # 予定表を表示するだけ(取得しない)
    python scripts/watch_odds.py --test-fetch 202607020206         # 1レースだけ即時取得して動作確認

再実行時の冪等性: odds_history_{date}.csv に既に記録済みの(race_id, checkpoint_label)は
スケジュールから除外される(スクリプトを再起動しても二重取得・二重行にならない)。
発走時刻を過ぎてしまった(=起動が遅れた)チェックポイントは、値が実態とずれるため
取得を試みずスキップする。

複勝・馬連オッズ(2026-09-02〜、予想ファクター充足度マップTier2、JRAのみ): 単勝と同じ
チェックポイントで race.netkeiba.com/api/api_get_jra_odds.html (JSON API)から取得し、
data/odds_history_fuku/{year}/{date}.csv・data/odds_history_umaren/{year}/{date}.csvへ
追記する。単勝側(odds_history_{date}.csv)とは異なり、こちらはOUT_DIR(セッション固有
scratchpad)ではなくdata/配下(Git管理・永続)にのみ書く - 既存の単勝オッズ時系列は
predict_pattern29.py/build_artifact.py側の既存の予想パイプラインが読む前提のファイルの
ため、スキーマ・置き場所を変更せず(下流互換性を壊さない)、複勝・馬連は完全に別の新規
永続データとして追加する設計。冪等性は単勝側のイベントスケジュール(_already_done)に
相乗りしており、複勝・馬連専用の重複排除チェックは無い(単勝が既取得ならそのチェック
ポイントは複勝・馬連含めてまとめてスキップされる)。
"""
import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv

from config.settings import LOG_DIR
from src.netkeiba_pipeline.auth.session import login
from src.netkeiba_pipeline.parsers.bias_parser import parse_bias
from src.netkeiba_pipeline.parsers.odds_api_parser import parse_fuku_odds, parse_umaren_odds
from src.netkeiba_pipeline.scrapers.bias import fetch_bias_html
from src.netkeiba_pipeline.scrapers.odds_api import fetch_jra_odds_json
from src.netkeiba_pipeline.storage.paths import odds_history_fuku_csv_path, odds_history_umaren_csv_path
from src.netkeiba_pipeline.utils.logging_conf import configure_logging

# このレポート生成パイプライン(build_artifact.py)専用の出力先。predict_pattern29.pyの
# race_names_{date}.csvと同じscratchpadディレクトリに揃える。
OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)

# (ラベル, 発走何秒前か)。負値は発走後(例: -600 = 発走10分後)。build_artifact.py側の
# 表示順(ODDS_CHECKPOINT_ORDER)もこの並びに揃えること。
# 「確定」は単勝オッズが締切後の値で固定された後を狙うバッファとして発走10分後に設定して
# いる(審議等で確定payoutの反映が遅れることがあるが、オッズ自体は発走(=締切)時点で
# 固定されるため、10分待てば通常は反映済み)。
CHECKPOINTS: list[tuple[str, int]] = [
    ("1時間前", 3600), ("30分前", 1800), ("20分前", 1200), ("10分前", 600),
    ("5分前", 300), ("4分前", 240), ("3分前", 180), ("2分前", 120),
    ("1分30秒前", 90), ("1分前", 60), ("30秒前", 30), ("確定", -600),
]

CSV_COLUMNS = [
    "race_id", "checkpoint_label", "checkpoint_offset_sec",
    "scheduled_time", "fetched_time", "umaban", "horse_name", "win_odds", "ninki",
]

# 複勝・馬連(予想ファクター充足度マップTier2、2026-09-02〜)。単勝と違い恒久データ
# (data/配下)にのみ書く - 既存のodds_history_{date}.csv(セッション固有scratchpad、
# 予想パイプライン専用)のスキーマは変更しない(下流のbuild_artifact.py等への影響を
# 避けるため)。JRAのみ対応(NAR側の同等APIは未検証、詳細はodds_api.py参照)。
FUKU_CSV_COLUMNS = [
    "race_id", "checkpoint_label", "checkpoint_offset_sec",
    "scheduled_time", "fetched_time", "umaban", "fuku_odds_low", "fuku_odds_high", "fuku_ninki",
]
UMAREN_CSV_COLUMNS = [
    "race_id", "checkpoint_label", "checkpoint_offset_sec",
    "scheduled_time", "fetched_time", "umaban_a", "umaban_b", "umaren_odds", "umaren_ninki",
]


@dataclass
class Event:
    scheduled_time: datetime
    race_id: str
    checkpoint_label: str
    offset_sec: int


def _suffix(circuit: str) -> str:
    return "_nar" if circuit == "nar" else ""


def _load_race_schedule(date: str, circuit: str) -> pd.DataFrame:
    path = OUT_DIR / f"race_names{_suffix(circuit)}_{date}.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} が見つかりません。先に predict_pattern29.py --date {date} --circuit {circuit} を"
            f"実行して race_names{_suffix(circuit)}_{date}.csv を生成してください"
            "(発走時刻の取得元になっています)。"
        )
    df = pd.read_csv(path, dtype=str)
    df = df.dropna(subset=["start_time"])
    return df[df["start_time"].str.match(r"^\d{1,2}:\d{2}$", na=False)]


def _already_done(date: str, circuit: str) -> set[tuple[str, str]]:
    out_path = OUT_DIR / f"odds_history{_suffix(circuit)}_{date}.csv"
    if not out_path.exists():
        return set()
    df = pd.read_csv(out_path, dtype=str)
    return set(zip(df["race_id"], df["checkpoint_label"]))


def _status_path(date: str, circuit: str) -> Path:
    return OUT_DIR / f"odds_watch_status{_suffix(circuit)}_{date}.json"


def _write_status(date: str, circuit: str, status: str, started_at: datetime, logger: logging.Logger) -> None:
    """build_artifact.py側が「本当にプロセスが生きているか」を判定するためのハートビート。

    odds_history_{date}.csvは最初のチェックポイントが発火するまで作られないため、その存在
    だけでは「起動はしたがまだ何も取得していない」区間(ログイン成功〜最初のfetch)を
    「未起動」と誤判定してしまう。ここでは待機ループの毎周回(最大60秒間隔)でこのファイルを
    上書きし、last_heartbeatの新しさで生存を判定できるようにする。書き込み失敗はオッズ取得
    本体を止める理由にはならないため、警告ログのみで握りつぶす。
    """
    try:
        payload = {
            "date": date, "circuit": circuit, "pid": os.getpid(), "status": status,
            "started_at": started_at.isoformat(sep=" "),
            "last_heartbeat": datetime.now().isoformat(sep=" "),
        }
        _status_path(date, circuit).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        logger.warning("failed to write status heartbeat", exc_info=True)


def build_events(date: str, now: datetime, circuit: str = "jra") -> tuple[list[Event], list[Event]]:
    """戻り値: (今後fetchすべき未来イベント(時刻順), 発走時刻を過ぎた/取得済みのためスキップするイベント)。"""
    schedule = _load_race_schedule(date, circuit)
    done = _already_done(date, circuit)
    future: list[Event] = []
    skipped: list[Event] = []
    for _, row in schedule.iterrows():
        race_id = row["race_id"]
        start_dt = datetime.strptime(f"{date} {row['start_time']}", "%Y%m%d %H:%M")
        for label, offset in CHECKPOINTS:
            ev = Event(start_dt - timedelta(seconds=offset), race_id, label, offset)
            if (race_id, label) in done or ev.scheduled_time <= now:
                skipped.append(ev)
            else:
                future.append(ev)
    future.sort(key=lambda e: e.scheduled_time)
    return future, skipped


def fetch_odds(session, race_id: str) -> pd.DataFrame:
    html = fetch_bias_html(session, race_id)
    return parse_bias(html, race_id)


def fetch_fuku_umaren_odds(session, race_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """複勝・馬連オッズをJSON APIから取得する(JRAのみ、呼び出し側で保証すること)。
    type=1で単勝+複勝がセットで返るが、単勝は既存のfetch_odds(bias.html)経由で
    別途取得済みのため複勝のみ使う。"""
    tanfuku_payload = fetch_jra_odds_json(session, race_id, bet_type=1)
    umaren_payload = fetch_jra_odds_json(session, race_id, bet_type=4)
    return parse_fuku_odds(tanfuku_payload, race_id), parse_umaren_odds(umaren_payload, race_id)


def _append_rows(date: str, circuit: str, race_id: str, label: str, offset: int, scheduled: datetime,
                  fetched: datetime, bias_df: pd.DataFrame) -> int:
    out_path = OUT_DIR / f"odds_history{_suffix(circuit)}_{date}.csv"
    is_new = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        n = 0
        for _, r in bias_df.iterrows():
            writer.writerow({
                "race_id": race_id, "checkpoint_label": label, "checkpoint_offset_sec": offset,
                "scheduled_time": scheduled.isoformat(sep=" "), "fetched_time": fetched.isoformat(sep=" "),
                "umaban": r["umaban"], "horse_name": r["bias_horse_name"],
                "win_odds": r["bias_win_odds"], "ninki": r["bias_ninki"],
            })
            n += 1
    return n


def _append_fuku_umaren_rows(date: str, race_id: str, label: str, offset: int, scheduled: datetime,
                              fetched: datetime, fuku_df: pd.DataFrame, umaren_df: pd.DataFrame) -> tuple[int, int]:
    """複勝・馬連を data/odds_history_fuku|umaren/{year}/{date}.csv (恒久データ)へ追記する。
    OUT_DIR(scratchpad)側には一切書かない - 既存のodds_history_{date}.csvのスキーマ・
    置き場所を変更しない方針(詳細はFUKU_CSV_COLUMNS付近のコメント参照)。"""
    fuku_path = odds_history_fuku_csv_path(date)
    umaren_path = odds_history_umaren_csv_path(date)
    fuku_path.parent.mkdir(parents=True, exist_ok=True)
    umaren_path.parent.mkdir(parents=True, exist_ok=True)

    fuku_is_new = not fuku_path.exists()
    with open(fuku_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FUKU_CSV_COLUMNS)
        if fuku_is_new:
            writer.writeheader()
        for _, r in fuku_df.iterrows():
            writer.writerow({
                "race_id": race_id, "checkpoint_label": label, "checkpoint_offset_sec": offset,
                "scheduled_time": scheduled.isoformat(sep=" "), "fetched_time": fetched.isoformat(sep=" "),
                "umaban": r["umaban"], "fuku_odds_low": r["fuku_odds_low"],
                "fuku_odds_high": r["fuku_odds_high"], "fuku_ninki": r["fuku_ninki"],
            })

    umaren_is_new = not umaren_path.exists()
    with open(umaren_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=UMAREN_CSV_COLUMNS)
        if umaren_is_new:
            writer.writeheader()
        for _, r in umaren_df.iterrows():
            writer.writerow({
                "race_id": race_id, "checkpoint_label": label, "checkpoint_offset_sec": offset,
                "scheduled_time": scheduled.isoformat(sep=" "), "fetched_time": fetched.isoformat(sep=" "),
                "umaban_a": r["umaban_a"], "umaban_b": r["umaban_b"],
                "umaren_odds": r["umaren_odds"], "umaren_ninki": r["umaren_ninki"],
            })

    return len(fuku_df), len(umaren_df)


def _require_credentials() -> tuple[str, str]:
    load_dotenv()
    email = os.environ.get("NETKEIBA_EMAIL")
    password = os.environ.get("NETKEIBA_PASSWORD")
    if not email or not password:
        raise SystemExit(
            "NETKEIBA_EMAIL / NETKEIBA_PASSWORD not set. Copy .env.example to .env "
            "and fill them in yourself (never paste real credentials into chat)."
        )
    return email, password


def run_watch(date: str, circuit: str = "jra") -> None:
    email, password = _require_credentials()

    configure_logging(LOG_DIR / f"watch_odds_{date}.log")
    logger = logging.getLogger("watch_odds")

    now = datetime.now()
    future, skipped = build_events(date, now, circuit)
    if skipped:
        logger.warning(
            "%d checkpoint(s) already past or already recorded at startup, skipping: %s",
            len(skipped),
            [(e.race_id, e.checkpoint_label) for e in skipped[:20]],
        )
    if not future:
        logger.info("no future checkpoints to watch for %s - nothing to do", date)
        return
    logger.info(
        "watching %d checkpoint(s) across %d race(s), from %s to %s",
        len(future), len({e.race_id for e in future}),
        future[0].scheduled_time, future[-1].scheduled_time,
    )

    session = login(email, password)
    logger.info("logged in to netkeiba")
    started_at = datetime.now()
    _write_status(date, circuit, "running", started_at, logger)

    errored = 0
    fetched_count = 0
    interrupted = False
    try:
        while future:
            now = datetime.now()
            _write_status(date, circuit, "running", started_at, logger)
            wait = (future[0].scheduled_time - now).total_seconds()
            if wait > 0:
                time.sleep(min(wait, 60))
                continue

            ev = future.pop(0)
            try:
                bias_df = fetch_odds(session, ev.race_id)
                if bias_df.empty:
                    logger.warning(
                        "empty bias table for race_id=%s (%s) - skipping", ev.race_id, ev.checkpoint_label
                    )
                    continue
                n = _append_rows(
                    date, circuit, ev.race_id, ev.checkpoint_label, ev.offset_sec,
                    ev.scheduled_time, datetime.now(), bias_df,
                )
                fetched_count += 1
                logger.info("fetched race_id=%s checkpoint=%s (%d horses)", ev.race_id, ev.checkpoint_label, n)

                # 複勝・馬連(Tier2、JRAのみ)。失敗しても単勝側の取得成功は既に確定して
                # いるので、監視ループ全体を止めずログのみに留める。
                if circuit == "jra":
                    try:
                        fuku_df, umaren_df = fetch_fuku_umaren_odds(session, ev.race_id)
                        n_fuku, n_umaren = _append_fuku_umaren_rows(
                            date, ev.race_id, ev.checkpoint_label, ev.offset_sec,
                            ev.scheduled_time, datetime.now(), fuku_df, umaren_df,
                        )
                        logger.info(
                            "fetched fuku/umaren race_id=%s checkpoint=%s (%d/%d rows)",
                            ev.race_id, ev.checkpoint_label, n_fuku, n_umaren,
                        )
                    except Exception:  # noqa: BLE001 - 単勝側の成功は既に確定済み、複勝・馬連は付随データ
                        logger.exception(
                            "failed to fetch fuku/umaren odds for race_id=%s checkpoint=%s",
                            ev.race_id, ev.checkpoint_label,
                        )
            except Exception:  # noqa: BLE001 - 1件の失敗で監視全体を止めない
                errored += 1
                logger.exception("failed to fetch race_id=%s checkpoint=%s", ev.race_id, ev.checkpoint_label)
    except KeyboardInterrupt:
        interrupted = True
        logger.info("interrupted by user - %d remaining checkpoint(s) not fetched", len(future))

    _write_status(date, circuit, "interrupted" if interrupted else "done", started_at, logger)
    logger.info("done. fetched=%d errored=%d remaining=%d", fetched_count, errored, len(future))


def print_schedule(date: str, circuit: str = "jra") -> None:
    now = datetime.now()
    future, skipped = build_events(date, now, circuit)
    print(f"now = {now.isoformat(sep=' ')}")
    print(f"future checkpoints: {len(future)} / already past-or-done: {len(skipped)}")
    for e in future:
        print(f"  {e.scheduled_time.isoformat(sep=' ')}  {e.race_id}  {e.checkpoint_label}")


def test_fetch(race_id: str) -> None:
    email, password = _require_credentials()
    session = login(email, password)
    df = fetch_odds(session, race_id)
    print(f"race_id={race_id}: {len(df)} horses")
    print(df[["umaban", "bias_horse_name", "bias_win_odds", "bias_ninki"]].to_string(index=False))

    fuku_df, umaren_df = fetch_fuku_umaren_odds(session, race_id)
    print(f"\nfuku: {len(fuku_df)} rows")
    print(fuku_df.to_string(index=False))
    print(f"\numaren: {len(umaren_df)} rows (showing first 10)")
    print(umaren_df.head(10).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="単勝オッズの時系列を発走前11時点で自動取得する(常駐・単独プロセス)。"
    )
    parser.add_argument("--date", help="kaisai_date (YYYYMMDD)")
    parser.add_argument("--circuit", choices=["jra", "nar"], default="jra", help="開催区分(既定: jra)")
    parser.add_argument("--print-schedule", action="store_true", help="取得は行わず予定表だけ表示して終了する")
    parser.add_argument("--test-fetch", metavar="RACE_ID", help="指定race_idを1回だけ即時取得して動作確認する")
    args = parser.parse_args()

    if args.test_fetch:
        test_fetch(args.test_fetch)
        return

    if not args.date:
        raise SystemExit("--date YYYYMMDD が必要です(--test-fetchのみの場合を除く)")

    if args.print_schedule:
        print_schedule(args.date, args.circuit)
        return

    run_watch(args.date, args.circuit)


if __name__ == "__main__":
    main()
