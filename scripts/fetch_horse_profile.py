"""db.netkeiba.com/horse/{horse_id}/ (競走馬プロフィール本体、生産者・馬主欄)を取得し、
data/horse_profile/{horse_id}.csv(horse_id単位、1ファイル1行)へ保存するスクリプト。

対象horse_idは、既存の data/newspaper/{jra,nar}/{race_id}.csv の horse_id 列から
--date/--race-id(--circuit)で指定したレース群を横断して集める(fetch_pedigree.pyと同じ)。

生産者・馬主は不変性が高いデータのため、fetch_pedigree.pyと同じ「存在すれば恒久スキップ」
キャッシュ方式(data/horse_profile/{horse_id}.csvが既に存在すれば再取得しない、--forceで
強制再取得)を採用する。年度別成績のような時系列データはこのスクリプトの対象外
(scripts/fetch_person_profile.pyを参照)。

前提: 対象race_idについて data/newspaper/{jra,nar}/{race_id}.csv が
scripts/fetch_newspaper.py で既に生成済みであること。

注意: db.netkeiba.com/horse/{horse_id}/ 自体はログイン不要の公開ページだが、--date指定時に
使う list_race_ids/list_nar_race_ids(開催日→race_id列挙)はログイン必須のnetkeiba
APIを使うため、他のfetch_*.pyスクリプトと同様に事前ログインが必要。
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv

from config.settings import LOG_DIR
from src.netkeiba_pipeline.auth.session import login
from src.netkeiba_pipeline.discovery.race_calendar import list_nar_race_ids, list_race_ids
from src.netkeiba_pipeline.parsers.horse_profile_parser import parse_horse_profile
from src.netkeiba_pipeline.scrapers.profile import fetch_horse_profile_html
from src.netkeiba_pipeline.storage.paths import horse_profile_csv_path, newspaper_csv_path
from src.netkeiba_pipeline.utils.logging_conf import configure_logging


def collect_horse_ids(race_ids: list[str], logger: logging.Logger) -> list[str]:
    """対象race_id群のnewspaper CSVからユニークなhorse_idを集める(ページ順、重複排除)。
    newspaper CSV未生成のrace_idはログを出してスキップする(例外にしない)。
    fetch_pedigree.pyのcollect_horse_idsと同一実装(意図的な重複、この単純なヘルパー
    関数のためだけに共有モジュールを新設するのは過剰と判断)。"""
    seen: dict[str, None] = {}
    for race_id in race_ids:
        path = newspaper_csv_path(race_id)
        if not path.exists():
            logger.warning("race_id=%s: %s が存在しない(先にfetch_newspaper.pyが必要) - skip", race_id, path.name)
            continue
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
        for horse_id in df.get("horse_id", pd.Series(dtype=str)).dropna().unique():
            seen.setdefault(str(horse_id), None)
    return list(seen.keys())


def fetch_horse_profile_for_horse(session, horse_id: str, force: bool, logger: logging.Logger) -> str:
    """戻り値: "fetched" / "skipped"(既存キャッシュあり、force未指定) / 呼び出し元でraiseされる例外。"""
    path = horse_profile_csv_path(horse_id)
    if path.exists() and not force:
        return "skipped"

    html = fetch_horse_profile_html(session, horse_id)
    df = parse_horse_profile(html, horse_id)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("horse_id=%s: saved %s", horse_id, path)
    return "fetched"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="db.netkeiba.com/horse/{horse_id}/ の生産者・馬主情報を取得し、"
        "data/horse_profile/{horse_id}.csv へhorse_id単位で保存する。"
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--race-id", help="単一race_idの出走馬だけを対象にする")
    target.add_argument("--date", help="この開催日(YYYYMMDD)の全race_idの出走馬を対象にする")
    parser.add_argument("--circuit", choices=["jra", "nar"], default="jra", help="開催区分(既定: jra)")
    parser.add_argument(
        "--force", action="store_true", help="既存のdata/horse_profile/{horse_id}.csvがあっても再取得する"
    )
    args = parser.parse_args()

    load_dotenv()
    email = os.environ.get("NETKEIBA_EMAIL")
    password = os.environ.get("NETKEIBA_PASSWORD")
    if not email or not password:
        raise SystemExit(
            "NETKEIBA_EMAIL / NETKEIBA_PASSWORD not set. Copy .env.example to .env "
            "and fill them in yourself (never paste real credentials into chat)."
        )

    log_name = args.date if args.date else args.race_id
    configure_logging(LOG_DIR / f"fetch_horse_profile_{log_name}.log")
    logger = logging.getLogger("fetch_horse_profile")

    session = login(email, password)

    if args.race_id:
        race_ids = [args.race_id]
    else:
        list_ids = list_nar_race_ids if args.circuit == "nar" else list_race_ids
        race_ids = list_ids(session, args.date)
        logger.info("Found %d race_ids for %s (circuit=%s)", len(race_ids), args.date, args.circuit)

    horse_ids = collect_horse_ids(race_ids, logger)
    logger.info("Collected %d unique horse_ids across %d race_ids", len(horse_ids), len(race_ids))

    fetched, skipped, failed = [], [], []
    for horse_id in horse_ids:
        try:
            result = fetch_horse_profile_for_horse(session, horse_id, args.force, logger)
        except Exception:
            logger.exception("horse_id=%s: failed to fetch/parse horse profile", horse_id)
            failed.append(horse_id)
            continue
        (fetched if result == "fetched" else skipped).append(horse_id)

    print(
        f"horse_profile: fetched {len(fetched)}, skipped (already cached) {len(skipped)}, "
        f"failed {len(failed)} / {len(horse_ids)} horse_ids across {len(race_ids)} race_ids"
    )
    if failed:
        print(f"Failed ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
