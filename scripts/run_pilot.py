import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from config.settings import LOG_DIR
from src.netkeiba_pipeline.auth.session import login
from src.netkeiba_pipeline.discovery.race_calendar import list_nar_race_ids, list_race_ids
from src.netkeiba_pipeline.parsers.race_result_parser import parse_payouts, parse_race_result
from src.netkeiba_pipeline.scrapers.race_result import fetch_race_result_html
from src.netkeiba_pipeline.storage.writer import is_already_scraped, mark_scraped, write_payouts, write_race_result
from src.netkeiba_pipeline.utils.logging_conf import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot: scrape race results for one kaisai_date")
    parser.add_argument("--date", required=True, help="kaisai_date in YYYYMMDD format")
    parser.add_argument(
        "--circuit",
        choices=["jra", "nar"],
        default="jra",
        help="jra (default, central racing, unchanged behavior) or nar (14 tracked "
        "regional tracks, written under a separate data/.../nar/ path)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-fetch and overwrite even race_ids already marked as scraped in the manifest",
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

    configure_logging(LOG_DIR / f"pilot_{args.date}_{args.circuit}.log")
    logger = logging.getLogger("run_pilot")

    session = login(email, password)

    list_ids = list_nar_race_ids if args.circuit == "nar" else list_race_ids
    race_ids = list_ids(session, args.date)
    logger.info("Found %d race_ids for %s (circuit=%s)", len(race_ids), args.date, args.circuit)

    success, skipped, failed = 0, 0, 0
    for race_id in race_ids:
        if not args.force and is_already_scraped(race_id):
            logger.info("Skipping %s (already scraped)", race_id)
            skipped += 1
            continue
        try:
            html = fetch_race_result_html(session, race_id)
            df = parse_race_result(html, race_id)
            write_race_result(df, args.date, race_id, circuit=args.circuit)
            payouts_df = parse_payouts(html, race_id)
            write_payouts(payouts_df, args.date, race_id, circuit=args.circuit)
            mark_scraped(race_id, status="success")
            logger.info("Scraped %s (%d result rows, %d payout rows)", race_id, len(df), len(payouts_df))
            success += 1
        except Exception:
            logger.exception("Failed to scrape %s", race_id)
            mark_scraped(race_id, status="failed")
            failed += 1

    logger.info("Done. success=%d skipped=%d failed=%d", success, skipped, failed)


if __name__ == "__main__":
    main()
