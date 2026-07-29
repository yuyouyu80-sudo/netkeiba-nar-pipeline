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
from src.netkeiba_pipeline.discovery.tracks import is_nar_race
from src.netkeiba_pipeline.parsers.course_analysis_parser import parse_course_analysis, parse_horse_stat_table
from src.netkeiba_pipeline.parsers.ranking_parser import parse_ranking
from src.netkeiba_pipeline.scrapers.course_analysis import CID_LABELS, fetch_course_analysis_html
from src.netkeiba_pipeline.scrapers.course_data import (
    COURSEDATA_CID_LABELS,
    fetch_course_data_html,
    fetch_surf_summary_html,
)
from src.netkeiba_pipeline.scrapers.ranking import fetch_ranking_html
from src.netkeiba_pipeline.storage.writer import (
    is_already_scraped,
    mark_scraped,
    write_course_analysis,
    write_course_ranking,
)
from src.netkeiba_pipeline.utils.logging_conf import configure_logging


def _run_courseanalysis(session, race_id: str, cids: list[int], logger: logging.Logger) -> None:
    for cid in cids:
        category_type = CID_LABELS.get(cid, str(cid))
        data_type = f"course_analysis_{category_type}"
        if is_already_scraped(race_id, data_type=data_type):
            logger.info("Skipping courseanalysis race_id=%s cid=%s (already scraped)", race_id, cid)
            continue
        try:
            html = fetch_course_analysis_html(session, race_id, cid)
            df = parse_course_analysis(html, race_id, cid)
            write_course_analysis(df, race_id, category_type)
            mark_scraped(race_id, data_type=data_type, status="success")
            logger.info("Scraped courseanalysis race_id=%s cid=%s (%d rows)", race_id, cid, len(df))
        except Exception:
            logger.exception("Failed courseanalysis race_id=%s cid=%s", race_id, cid)
            mark_scraped(race_id, data_type=data_type, status="failed")


def _run_coursedata(session, race_id: str, cids: list[int], logger: logging.Logger) -> None:
    for cid in cids:
        category_type = COURSEDATA_CID_LABELS.get(cid, f"coursedata_{cid}")
        data_type = f"course_analysis_{category_type}"
        if is_already_scraped(race_id, data_type=data_type):
            logger.info("Skipping coursedata race_id=%s cid=%s (already scraped)", race_id, cid)
            continue
        try:
            html = fetch_course_data_html(session, race_id, cid)
            df = parse_horse_stat_table(html, race_id, category_type, source=f"coursedata cid={cid}")
            write_course_analysis(df, race_id, category_type)
            mark_scraped(race_id, data_type=data_type, status="success")
            logger.info("Scraped coursedata race_id=%s cid=%s (%d rows)", race_id, cid, len(df))
        except Exception:
            logger.exception("Failed coursedata race_id=%s cid=%s", race_id, cid)
            mark_scraped(race_id, data_type=data_type, status="failed")


def _run_surf_summary(session, race_id: str, logger: logging.Logger) -> None:
    data_type = "course_analysis_speed_index"
    if is_already_scraped(race_id, data_type=data_type):
        logger.info("Skipping surf_summary race_id=%s (already scraped)", race_id)
        return
    try:
        html = fetch_surf_summary_html(session, race_id)
        df = parse_horse_stat_table(html, race_id, "speed_index", source="surf_summary")
        write_course_analysis(df, race_id, "speed_index")
        mark_scraped(race_id, data_type=data_type, status="success")
        logger.info("Scraped surf_summary race_id=%s (%d rows)", race_id, len(df))
    except ValueError:
        if not is_nar_race(race_id):
            logger.exception("Failed surf_summary race_id=%s", race_id)
            mark_scraped(race_id, data_type=data_type, status="failed")
            return
        # Confirmed genuinely absent for NAR (no table#table_sort_back at all,
        # checked on multiple tracks/dates) - not a transient scrape failure,
        # so mark success-with-nothing-to-write rather than "failed" (which
        # would just get pointlessly retried on every future run).
        logger.info("race_id=%s: surf_summary speed_index data not published for NAR - skipping", race_id)
        mark_scraped(race_id, data_type=data_type, status="success")
    except Exception:
        logger.exception("Failed surf_summary race_id=%s", race_id)
        mark_scraped(race_id, data_type=data_type, status="failed")


def _run_ranking(session, race_id: str, logger: logging.Logger) -> None:
    data_type = "course_ranking"
    if is_already_scraped(race_id, data_type=data_type):
        logger.info("Skipping ranking race_id=%s (already scraped)", race_id)
        return
    try:
        html = fetch_ranking_html(session, race_id)
        df = parse_ranking(html, race_id)
        for ranking_type, group in df.groupby("ranking_type"):
            write_course_ranking(group, race_id, ranking_type)
        mark_scraped(race_id, data_type=data_type, status="success")
        logger.info("Scraped ranking race_id=%s (%d rows)", race_id, len(df))
    except Exception:
        logger.exception("Failed ranking race_id=%s", race_id)
        mark_scraped(race_id, data_type=data_type, status="failed")


def _run_all_for_race(session, race_id: str, cids: list[int], coursedata_cids: list[int], args, logger) -> None:
    _run_courseanalysis(session, race_id, cids, logger)
    _run_coursedata(session, race_id, coursedata_cids, logger)
    if not args.skip_surf_summary:
        _run_surf_summary(session, race_id, logger)
    if not args.skip_ranking:
        _run_ranking(session, race_id, logger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch per-race course/data-analysis pages")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--race-id", help="fetch for a single race_id")
    target.add_argument("--date", help="fetch for every race_id on this kaisai_date (YYYYMMDD)")
    parser.add_argument(
        "--circuit",
        choices=["jra", "nar"],
        default="jra",
        help="jra (default, unchanged behavior) or nar (14 tracked regional tracks). "
        "Only affects --date discovery; --race-id is unaffected (circuit is inferred from the race_id itself).",
    )
    parser.add_argument("--cids", default="0,1,2,3", help="courseanalysis cid values (default: 0,1,2,3)")
    parser.add_argument(
        "--coursedata-cids",
        default="1,4",
        help="coursedata cid values: 1=sire, 4=broodmare_sire (default: 1,4)",
    )
    parser.add_argument("--skip-surf-summary", action="store_true")
    parser.add_argument("--skip-ranking", action="store_true")
    args = parser.parse_args()
    cids = [int(c) for c in args.cids.split(",")] if args.cids else []
    coursedata_cids = [int(c) for c in args.coursedata_cids.split(",")] if args.coursedata_cids else []

    load_dotenv()
    email = os.environ.get("NETKEIBA_EMAIL")
    password = os.environ.get("NETKEIBA_PASSWORD")
    if not email or not password:
        raise SystemExit(
            "NETKEIBA_EMAIL / NETKEIBA_PASSWORD not set. Copy .env.example to .env "
            "and fill them in yourself (never paste real credentials into chat)."
        )

    log_name = args.date if args.date else args.race_id
    configure_logging(LOG_DIR / f"course_analysis_{log_name}.log")
    logger = logging.getLogger("fetch_course_analysis")

    session = login(email, password)

    if args.race_id:
        _run_all_for_race(session, args.race_id, cids, coursedata_cids, args, logger)
    else:
        list_ids = list_nar_race_ids if args.circuit == "nar" else list_race_ids
        race_ids = list_ids(session, args.date)
        logger.info("Found %d race_ids for %s (circuit=%s)", len(race_ids), args.date, args.circuit)
        for race_id in race_ids:
            _run_all_for_race(session, race_id, cids, coursedata_cids, args, logger)


if __name__ == "__main__":
    main()
