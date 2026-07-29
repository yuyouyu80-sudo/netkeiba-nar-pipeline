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
from src.netkeiba_pipeline.discovery.race_calendar import list_race_ids
from src.netkeiba_pipeline.parsers.bias_parser import parse_bias
from src.netkeiba_pipeline.scrapers.bias import fetch_bias_html
from src.netkeiba_pipeline.storage.paths import newspaper_csv_path
from src.netkeiba_pipeline.utils.logging_conf import configure_logging


def update_weight_for_race(session, race_id: str, logger: logging.Logger) -> str:
    """既存のnewspaper CSVのうち馬体重(bias_horse_weight)が未取得の行があれば
    table.Biasページだけを再取得し、馬体重列だけを上書き更新する。他の列・他の
    ページは一切再取得しない(通常のfetch_newspaper.pyよりずっと軽い)。
    馬体重は発表され次第table.Biasに列が追加される仕様(bias_parser.py参照)
    のため、未発表の間は同じページを何度取り直しても空のままになる。
    """
    path = newspaper_csv_path(race_id)
    if not path.exists():
        return "no_file"

    df = pd.read_csv(path, dtype=str, encoding="utf-8")
    if df.empty or "bias_horse_weight" not in df.columns or "umaban" not in df.columns:
        return "no_file"

    missing_before = df["bias_horse_weight"].isna() | (df["bias_horse_weight"] == "")
    if not missing_before.any():
        return "already_complete"

    bias_html = fetch_bias_html(session, race_id)
    bias_df = parse_bias(bias_html, race_id)
    bias_df["umaban"] = bias_df["umaban"].astype(str)
    weight_map = dict(zip(bias_df["umaban"], bias_df["bias_horse_weight"]))

    df["umaban"] = df["umaban"].astype(str)
    fetched = df["umaban"].map(weight_map)
    df["bias_horse_weight"] = fetched.where(fetched.notna() & (fetched != ""), df["bias_horse_weight"])

    df.to_csv(path, index=False, encoding="utf-8")

    still_missing = int((df["bias_horse_weight"].isna() | (df["bias_horse_weight"] == "")).sum())
    if still_missing == 0:
        logger.info("race_id=%s: filled all missing horse weights", race_id)
        return "filled"
    logger.info(
        "race_id=%s: still missing weight for %d horse(s) (not yet announced by netkeiba)",
        race_id,
        still_missing,
    )
    return "still_missing"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="馬体重(bias_horse_weight)が未取得のレースについて、table.Biasページだけを"
        "再取得して馬体重列を補完する追加取得スクリプト。既に馬体重が揃っているレースは"
        "通信せずスキップする。"
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--race-id", help="対象race_idを1件指定")
    target.add_argument("--date", help="kaisai_date(YYYYMMDD)を指定し、既存newspaperデータがある全レースを対象にする")
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
    configure_logging(LOG_DIR / f"weight_{log_name}.log")
    logger = logging.getLogger("fetch_horse_weight")

    session = login(email, password)

    if args.race_id:
        race_ids = [args.race_id]
    else:
        race_ids = list_race_ids(session, args.date)
        logger.info("Found %d race_ids for %s", len(race_ids), args.date)

    counts: dict[str, int] = {}
    for race_id in race_ids:
        try:
            status = update_weight_for_race(session, race_id, logger)
        except Exception:
            logger.exception("Failed race_id=%s", race_id)
            status = "error"
        counts[status] = counts.get(status, 0) + 1

    print(
        "Done. filled={filled} still_missing={still_missing} already_complete={already_complete} "
        "no_file={no_file} error={error}".format(
            filled=counts.get("filled", 0),
            still_missing=counts.get("still_missing", 0),
            already_complete=counts.get("already_complete", 0),
            no_file=counts.get("no_file", 0),
            error=counts.get("error", 0),
        )
    )


if __name__ == "__main__":
    main()
