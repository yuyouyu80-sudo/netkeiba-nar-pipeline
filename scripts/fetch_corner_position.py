"""newspaper.htmlの「AI展開」4コーナー位置取り予想を取得し、既存のnewspaper CSV
(data/newspaper/{jra,nar}/{race_id}.csv)へ列を追加・上書き更新するスクリプト。

対象データはnewspaper.html自体に(JS実行不要で)既に含まれているため、mark_list.htmlと違い
Playwrightは不要。既存のfetch_newspaper_html()(requestsベース)を再利用する軽量版
(refresh_bias.pyと同じ設計)。

前提: 対象race_idについて data/newspaper/{jra,nar}/{race_id}.csv が
scripts/fetch_newspaper.py で既に生成済みであること。このスクリプトは新規行を作らず、
既存行に以下の列を追加・上書きする:
- corner4_rank: 4コーナー時点の推定順位(1=先頭、同着は同順位)
- corner4_gap_pct: 先頭からの差(netkeiba側の座標スケール、0=先頭)
- corner4_gap_lengths: 上記を馬身換算した近似値(柵の間隔≈1馬身という換算、
  詳細はsrc/netkeiba_pipeline/parsers/corner_position_parser.py参照)
- corner4_speedup: 4コーナーでの加速マーク(▶)の数(0〜3)
- corner3_rank / corner3_gap_pct / corner3_gap_lengths: 3コーナー時点の同種データ
  (2026-08-27追加。加速マークは4コーナー時点のみ描画される仕様のためcorner3_speedupは無い)

netkeiba側にAI展開データが無いレース(未成立・データ欠落等)はスキップする。
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
from src.netkeiba_pipeline.parsers.corner_position_parser import parse_corner3_position, parse_corner4_position
from src.netkeiba_pipeline.scrapers.newspaper import fetch_newspaper_html
from src.netkeiba_pipeline.storage.paths import newspaper_csv_path
from src.netkeiba_pipeline.storage.writer import mark_scraped
from src.netkeiba_pipeline.utils.logging_conf import configure_logging


def refresh_corner_position_for_race(session, race_id: str, logger: logging.Logger) -> tuple[int, int]:
    """戻り値: (値が更新された馬数, CSV上の総馬数)。CSV未生成、またはAI展開データが
    無いレースなら(0, 総馬数)。"""
    path = newspaper_csv_path(race_id)
    if not path.exists():
        logger.warning("race_id=%s: %s が存在しない(先にfetch_newspaper.pyが必要) - skip", race_id, path.name)
        return 0, 0

    df = pd.read_csv(path, dtype=str, encoding="utf-8")
    if df.empty:
        return 0, 0

    html = fetch_newspaper_html(session, race_id)
    corner4_df = parse_corner4_position(html, race_id)
    corner3_df = parse_corner3_position(html, race_id)
    if corner4_df.empty and corner3_df.empty:
        logger.warning("race_id=%s: AI展開データが無い - skip", race_id)
        return 0, len(df)

    corner4_df = corner4_df.drop(columns=["horse_id", "horse_name"], errors="ignore")
    corner3_df = corner3_df.drop(columns=["horse_id", "horse_name"], errors="ignore")
    for d in (corner4_df, corner3_df):
        if not d.empty:
            d["umaban"] = d["umaban"].astype(str)

    if corner4_df.empty:
        corner_df = corner3_df.set_index("umaban")
    elif corner3_df.empty:
        corner_df = corner4_df.set_index("umaban")
    else:
        corner_df = corner4_df.set_index("umaban").join(corner3_df.set_index("umaban"), how="outer")

    df["umaban"] = df["umaban"].astype(str)

    updated = 0
    for col in corner_df.columns:
        if col not in df.columns:
            df[col] = pd.NA
        new_vals = df["umaban"].map(corner_df[col])
        changed = new_vals.notna() & (df[col].fillna("").astype(str) != new_vals.fillna("").astype(str))
        updated = max(updated, int(changed.sum()))
        df[col] = new_vals.where(new_vals.notna(), df[col])

    df.to_csv(path, index=False, encoding="utf-8")
    mark_scraped(race_id, data_type="corner_position")
    logger.info("race_id=%s: refreshed corner3/corner4 position for %d/%d horses", race_id, updated, len(df))
    return updated, len(df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="newspaper.htmlの3・4コーナー位置取り(AI展開)を取得し、既存のnewspaper CSVへ"
        "corner{3,4}_rank/corner{3,4}_gap_pct/corner{3,4}_gap_lengths/corner4_speedupを追加・上書き更新する。"
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
    configure_logging(LOG_DIR / f"fetch_corner_position_{log_name}.log")
    logger = logging.getLogger("fetch_corner_position")

    session = login(email, password)

    if args.race_id:
        n, total = refresh_corner_position_for_race(session, args.race_id, logger)
        print(f"{args.race_id}: updated {n}/{total} horses")
        return

    list_ids = list_nar_race_ids if args.circuit == "nar" else list_race_ids
    race_ids = list_ids(session, args.date)
    logger.info("Found %d race_ids for %s (circuit=%s)", len(race_ids), args.date, args.circuit)

    total_updated, total_horses, missing, failed = 0, 0, [], []
    for race_id in race_ids:
        try:
            n, total = refresh_corner_position_for_race(session, race_id, logger)
        except Exception:
            logger.exception("race_id=%s: failed to fetch/parse corner position", race_id)
            failed.append(race_id)
            continue
        if total == 0:
            missing.append(race_id)
        total_updated += n
        total_horses += total

    print(
        f"refreshed corner3/corner4 position: {total_updated}/{total_horses} horses across "
        f"{len(race_ids) - len(missing) - len(failed)}/{len(race_ids)} races for {args.date}"
    )
    if missing:
        print(f"skipped (no existing newspaper csv or no AI-tenkai data): {len(missing)} -> {missing}")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
