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
from src.netkeiba_pipeline.discovery.tracks import is_nar_race
from src.netkeiba_pipeline.parsers.bias_parser import parse_bias
from src.netkeiba_pipeline.parsers.course_analysis_parser import parse_course_analysis, parse_horse_stat_table
from src.netkeiba_pipeline.parsers.holding_time_parser import parse_holding_time
from src.netkeiba_pipeline.parsers.newspaper_parser import has_predictrap_paywall, parse_newspaper
from src.netkeiba_pipeline.parsers.race_data_parser import parse_data_breakdown, parse_horse_category_table
from src.netkeiba_pipeline.parsers.shutuba_past_parser import parse_shutuba_past
from src.netkeiba_pipeline.parsers.speed_index_parser import parse_speed_index
from src.netkeiba_pipeline.scrapers.bias import fetch_bias_html
from src.netkeiba_pipeline.scrapers.course_analysis import CID_LABELS, fetch_course_analysis_html
from src.netkeiba_pipeline.scrapers.course_data import (
    COURSEDATA_CID_LABELS,
    fetch_course_data_html,
    fetch_surf_summary_html,
)
from src.netkeiba_pipeline.scrapers.holding_time import fetch_holding_time_data
from src.netkeiba_pipeline.scrapers.newspaper import fetch_newspaper_html
from src.netkeiba_pipeline.scrapers.race_data import (
    DATA_BREAKDOWN_MODES,
    NAR_DATA_BREAKDOWN_MODES,
    fetch_concerned_html,
    fetch_data_breakdown_html,
)
from src.netkeiba_pipeline.scrapers.shutuba_past import fetch_shutuba_past_html
from src.netkeiba_pipeline.scrapers.speed_index import fetch_speed_index_html
from src.netkeiba_pipeline.storage.paths import newspaper_csv_path
from src.netkeiba_pipeline.storage.writer import mark_scraped
from src.netkeiba_pipeline.utils.logging_conf import configure_logging

# surf_summary.html key1+key2 combos: pairs a horse's own attribute (jockey /
# odds zone / pedigree number) with a second attribute, showing that specific
# combination's win/place record. range=5 pages compare against a coarser
# bucket (all horses sharing the sire line), range=4 against exact-code
# matches (this exact jockey/trainer/owner pair etc).
SURF_SUMMARY_COMBOS = [
    ("surf_ketto_training", 5, "KettoNum", "TrainingValue"),
    ("surf_ketto_comment", 5, "KettoNum", "CommentValue"),
    ("surf_odds_jockey", 4, "OddsZone", "JockeyCode"),
    ("surf_jockey_trainer", 4, "JockeyCode", "TrainerCode"),
    ("surf_jockey_owner", 4, "JockeyCode", "OwnerCode"),
    ("surf_jockey_prevjockey", 4, "JockeyCode", "PrevJockeyCode"),
]


def _merge_stat_table(df: pd.DataFrame, source_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Merges a long-format (race_id, category_type, umaban, category_label,
    ...stats..., horse_id, horse_name) table onto df, keyed on umaban, with
    every non-key column renamed under `prefix` so distinct sources never
    collide."""
    source_df = source_df.copy()
    source_df["umaban"] = source_df["umaban"].astype(str)
    source_df = source_df.drop(columns=["race_id", "category_type"])
    source_df = source_df.rename(columns={c: f"{prefix}_{c}" for c in source_df.columns if c != "umaban"})
    return df.merge(source_df, on="umaban", how="left", validate="one_to_one")


def _merge_breakdown(df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    """parse_data_breakdown already prefixes every column except umaban."""
    source_df = source_df.copy()
    source_df["umaban"] = source_df["umaban"].astype(str)
    return df.merge(source_df, on="umaban", how="left", validate="one_to_one")


_WRITEUP_COLUMNS = [
    "waku",
    "stable_comment",
    "stable_comment_reporter",
    "stable_comment_rating_code",
    "training_review",
    "training_date",
    "training_course",
    "training_track_condition",
    "training_rider",
    "training_times",
    "training_partner_comment",
    "training_position",
    "training_load",
    "training_critic",
    "training_rank",
]


def fetch_one_race(session, race_id: str, logger: logging.Logger, circuit: str = "jra") -> Path:
    is_nar = circuit == "nar"
    html = fetch_newspaper_html(session, race_id)

    if has_predictrap_paywall(html):
        logger.warning(
            "race_id=%s: predicted-lap table (PredictRap_Table) is paywalled beyond this "
            "account's access level - omitting it from the output.",
            race_id,
        )

    # Fetched before parse_newspaper() below (rather than in its usual spot further
    # down) because NAR needs it as a fallback horse-identity source.
    bias_html = fetch_bias_html(session, race_id)
    bias_df = parse_bias(bias_html, race_id)
    bias_df["umaban"] = bias_df["umaban"].astype(str)

    df = parse_newspaper(html, race_id, require_writeup=not is_nar)
    if df.empty and is_nar:
        # Confirmed against multiple real races (including a named/featured
        # one) that NAR does not publish 厩舎コメント/調教タイム content at all -
        # bias.html's horse identity becomes the base frame instead, with the
        # writeup columns left blank for every horse (the honest NAR reality,
        # not a missing-data bug).
        logger.info(
            "race_id=%s: no 厩舎コメント/調教タイム content (expected for NAR) - "
            "using bias.html horse identity as the base frame",
            race_id,
        )
        df = bias_df[["umaban", "bias_horse_id", "bias_horse_name"]].rename(
            columns={"bias_horse_id": "horse_id", "bias_horse_name": "horse_name"}
        )
        for col in _WRITEUP_COLUMNS:
            df[col] = pd.NA
        df.insert(0, "race_id", race_id)

    past_html = fetch_shutuba_past_html(session, race_id)
    past_df = parse_shutuba_past(past_html, race_id)

    speed_html = fetch_speed_index_html(session, race_id)
    speed_df = parse_speed_index(speed_html, race_id)

    holding_time_payload = fetch_holding_time_data(session, race_id)
    holding_time_df = parse_holding_time(holding_time_payload, race_id)

    df["umaban"] = df["umaban"].astype(str)
    past_df["umaban"] = past_df["umaban"].astype(str)
    speed_df["umaban"] = speed_df["umaban"].astype(str)
    bias_df["umaban"] = bias_df["umaban"].astype(str)
    df["horse_id"] = df["horse_id"].astype(str)
    holding_time_df["horse_id"] = holding_time_df["horse_id"].astype(str)
    df = df.merge(past_df, on="umaban", how="left", validate="one_to_one")
    df = df.merge(speed_df, on="umaban", how="left", validate="one_to_one")
    df = df.merge(holding_time_df, on="horse_id", how="left", validate="one_to_one")
    df = df.merge(bias_df, on="umaban", how="left", validate="one_to_one")

    # mode=courseanalysis (waku/running_style/jockey/trainer)
    for cid, cat in CID_LABELS.items():
        html_ca = fetch_course_analysis_html(session, race_id, cid)
        df_ca = parse_course_analysis(html_ca, race_id, cid)
        df = _merge_stat_table(df, df_ca, prefix=f"ca_{cat}")

    # mode=coursedata (sire/broodmare_sire)
    for cid, cat in COURSEDATA_CID_LABELS.items():
        html_cd = fetch_course_data_html(session, race_id, cid)
        df_cd = parse_horse_stat_table(html_cd, race_id, cat, source=f"coursedata cid={cid}")
        df = _merge_stat_table(df, df_cd, prefix=f"ca_{cat}")

    # surf_summary.html default (speed_index aptitude ranking) - confirmed genuinely
    # absent for NAR (no table#table_sort_back at all, checked on 2 different tracks/
    # dates), unlike course_analysis/coursedata which do exist there. Skip rather than
    # fail the whole race over one missing sub-metric.
    try:
        html_surf = fetch_surf_summary_html(session, race_id)
        df_surf = parse_horse_stat_table(html_surf, race_id, "speed_index", source="surf_summary")
        df = _merge_stat_table(df, df_surf, prefix="ca_speed_index")
    except ValueError:
        if not is_nar:
            raise
        logger.info("race_id=%s: surf_summary speed_index data not published for NAR - skipping", race_id)

    # mode=concerned cid=0 (同場同距離 - same venue/surface/distance record)
    html_concerned = fetch_concerned_html(session, race_id, cid=0)
    df_concerned = parse_horse_category_table(html_concerned, race_id, "concerned", source="concerned cid=0")
    df = _merge_stat_table(df, df_concerned, prefix="concerned")

    # surf_summary.html key1+key2 combos (jockey/trainer/owner/odds/pedigree cross-stats)
    for prefix, range_, key1, key2 in SURF_SUMMARY_COMBOS:
        html_combo = fetch_surf_summary_html(session, race_id, range_=range_, key1=key1, key2=key2)
        df_combo = parse_horse_category_table(html_combo, race_id, prefix, source=prefix)
        df = _merge_stat_table(df, df_combo, prefix=prefix)

    # data.html mode=distance/course/condition/others/cushion/baba_water
    breakdown_modes = NAR_DATA_BREAKDOWN_MODES if is_nar else DATA_BREAKDOWN_MODES
    for mode, num_slots in breakdown_modes.items():
        html_breakdown = fetch_data_breakdown_html(session, race_id, mode)
        # 2026-08-13: NAR's mode=distance row count is not actually fixed at 5 -
        # it's "however many nearby-distance buckets this race/horse has comparison
        # data for" (observed range: 1-4) followed by a fixed final "全成績" row.
        # num_slots=5 stays as an upper-bound sanity check; acceptance is instead
        # gated on the last row being labeled "全成績". Other modes are unaffected
        # (not observed to vary) and keep the strict exact-count check.
        terminal_label = "全成績" if (is_nar and mode == "distance") else None
        df_breakdown = parse_data_breakdown(html_breakdown, race_id, f"data_{mode}", num_slots, terminal_label=terminal_label)
        df = _merge_breakdown(df, df_breakdown)

    df["umaban"] = pd.to_numeric(df["umaban"], errors="coerce")
    df = df.sort_values("umaban").reset_index(drop=True)

    path = newspaper_csv_path(race_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Wrote %d rows to %s", len(df), path)
    # Phase B(2026-08-04, NAR回収率改善計画): このタイムスタンプが、モデル検証で
    # 使う馬柱データの取得時刻を初めて記録する。従来はfetch_newspaper.pyがmanifestに
    # 一切書き込んでおらず、「発走前取得か発走後取得か」を事後に判別する手段が
    # 無かった(2026-08-01発覚のdata_provenance_caveat参照)。過去分は遡って復元
    # できないが、今後の取得はここから記録され、nar_model/verify_provenance.pyで
    # 発走時刻と突き合わせて使えるようになる。
    mark_scraped(race_id, data_type="newspaper")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the full per-horse newspaper dataset (newspaper/shutuba_past/speed/holding_time/"
        "bias/course_analysis/course_data/concerned/surf_summary/data breakdowns), one row per horse"
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--race-id", help="fetch a single race_id")
    target.add_argument("--date", help="fetch every race_id on this kaisai_date (YYYYMMDD)")
    parser.add_argument(
        "--circuit",
        choices=["jra", "nar"],
        default="jra",
        help="jra (default, unchanged behavior) or nar (14 tracked regional tracks). "
        "Ignored when --race-id is given directly (circuit is inferred from the race_id itself).",
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
    configure_logging(LOG_DIR / f"newspaper_{log_name}.log")
    logger = logging.getLogger("fetch_newspaper")

    session = login(email, password)

    if args.race_id:
        circuit = "nar" if is_nar_race(args.race_id) else "jra"
        path = fetch_one_race(session, args.race_id, logger, circuit=circuit)
        print(f"Wrote {path}")
        return

    list_ids = list_nar_race_ids if args.circuit == "nar" else list_race_ids
    race_ids = list_ids(session, args.date)
    logger.info("Found %d race_ids for %s (circuit=%s)", len(race_ids), args.date, args.circuit)
    succeeded = []
    failed = []
    for race_id in race_ids:
        try:
            path = fetch_one_race(session, race_id, logger, circuit=args.circuit)
            succeeded.append((race_id, path))
        except Exception:
            logger.exception("Failed race_id=%s", race_id)
            failed.append(race_id)

    print(f"{len(succeeded)}/{len(race_ids)} races written for {args.date}")
    for race_id, path in succeeded:
        print(f"  {race_id}: {path}")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
