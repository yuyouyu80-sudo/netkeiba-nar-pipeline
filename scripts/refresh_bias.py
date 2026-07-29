"""発走前レースの人気・オッズ・馬体重(bias.html由来の3列)だけを再取得し、既存の
newspaper CSV(data/newspaper/{jra,nar}/{race_id}.csv)を上書き更新する軽量スクリプト。

fetch_newspaper.pyの完全版は1レースあたり20以上のページを取得し約1分かかるが、
発走が近づくにつれて変動するのは実質この3列だけ(馬体重は発走約1時間前に発表、
オッズ・人気は締切直前まで変動)。bias.html 1ページのみを取得するため数秒/レースで
済み、直前の再取得に向く。取消・除外馬が出た場合、bias.html側は当該馬のオッズ・人気を
空にするため、この3列を上書きするだけで predict_pattern29.py 等の _drop_scratched
(オッズ・人気が両方空の馬を除外する処理)が正しく最新の出走馬だけを対象にする。

前提: 対象race_idについて data/newspaper/{jra,nar}/{race_id}.csv が
scripts/fetch_newspaper.py で既に生成済みであること。このスクリプトは新規行・
新規列を作らず、既存行のbias_win_odds/bias_ninki/bias_horse_weight列だけを
上書きする(未生成のrace_idはスキップし、fetch_newspaper.pyの実行を促す)。
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
from src.netkeiba_pipeline.parsers.bias_parser import parse_bias
from src.netkeiba_pipeline.scrapers.bias import fetch_bias_html
from src.netkeiba_pipeline.storage.paths import newspaper_csv_path
from src.netkeiba_pipeline.utils.logging_conf import configure_logging

REFRESH_COLUMNS = ["bias_win_odds", "bias_ninki", "bias_horse_weight"]


def refresh_one_race(session, race_id: str, logger: logging.Logger) -> tuple[int, int]:
    """戻り値: (値が変化した馬数, CSV上の総馬数)。CSV未生成なら(0, 0)。"""
    path = newspaper_csv_path(race_id)
    if not path.exists():
        logger.warning("race_id=%s: %s が存在しない(先にfetch_newspaper.pyが必要) - skip", race_id, path.name)
        return 0, 0

    df = pd.read_csv(path, dtype=str, encoding="utf-8")
    if df.empty:
        return 0, 0

    html = fetch_bias_html(session, race_id)
    bias_df = parse_bias(html, race_id)
    if bias_df.empty:
        logger.warning("race_id=%s: bias.html取得結果が空 - skip", race_id)
        return 0, len(df)

    bias_df["umaban"] = bias_df["umaban"].astype(str)
    bias_df = bias_df.set_index("umaban")
    df["umaban"] = df["umaban"].astype(str)

    updated = 0
    for col in REFRESH_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
        if col not in bias_df.columns:
            continue
        new_vals = df["umaban"].map(bias_df[col])
        changed = new_vals.notna() & (df[col].fillna("").astype(str) != new_vals.fillna("").astype(str))
        updated = max(updated, int(changed.sum()))
        df[col] = new_vals.where(new_vals.notna(), df[col])

    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("race_id=%s: refreshed odds/ninki/weight for %d/%d horses", race_id, updated, len(df))
    return updated, len(df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="発走前レースの人気・オッズ・馬体重だけをbias.html 1ページから再取得し、"
        "既存のnewspaper CSVを上書き更新する(全項目の再取得はしない軽量版)。"
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--race-id", help="単一race_idのみ更新する")
    target.add_argument("--date", help="この開催日(YYYYMMDD)の全race_idを更新する")
    parser.add_argument("--circuit", choices=["jra", "nar"], default="jra", help="開催区分(既定: jra)")
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
    configure_logging(LOG_DIR / f"refresh_bias_{log_name}.log")
    logger = logging.getLogger("refresh_bias")

    session = login(email, password)

    if args.race_id:
        n, total = refresh_one_race(session, args.race_id, logger)
        print(f"{args.race_id}: updated {n}/{total} horses")
        return

    list_ids = list_nar_race_ids if args.circuit == "nar" else list_race_ids
    race_ids = list_ids(session, args.date)
    logger.info("Found %d race_ids for %s (circuit=%s)", len(race_ids), args.date, args.circuit)

    total_updated, total_horses, missing = 0, 0, []
    for race_id in race_ids:
        n, total = refresh_one_race(session, race_id, logger)
        if total == 0:
            missing.append(race_id)
        total_updated += n
        total_horses += total

    print(
        f"refreshed odds/ninki/weight: {total_updated}/{total_horses} horses across "
        f"{len(race_ids) - len(missing)}/{len(race_ids)} races for {args.date}"
    )
    if missing:
        print(f"skipped (no existing newspaper csv): {len(missing)} -> {missing}")


if __name__ == "__main__":
    main()
