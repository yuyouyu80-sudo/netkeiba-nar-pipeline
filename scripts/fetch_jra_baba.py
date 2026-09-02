"""JRA公式サイト(www.jra.go.jp、netkeiba外)のクッション値・含水率アーカイブPDFを取得し、
data/jra_baba/{year}/{venue_code}_{kai}.csv(開催回単位、1ファイル1開催回分の全日程)へ
保存するスクリプト。

**事後データである点に注意**(予想ファクター充足度マップTier3、詳細は
src/netkeiba_pipeline/scrapers/jra_official_baba.pyのdocstring参照): 対象の開催回が
完全に終了してから公開されるため、進行中の開催回を指定すると失敗する(403、
「まだ未公開」として扱いskip対象、エラー扱いにしない)。日次の予想パイプラインには
使えず、あくまで事後の傾向分析用データ。

対象(year, venue_code, kai)は、既存の data/race_results/{year}/*.csv (--date指定時は
その日付のみ、--race-id指定時はそのrace_idのみ)からrace_idを集め、
race_id[4:6]=venue_code・race_id[6:8]=kaiを取り出して重複排除する
(NAR race_idは対象外、is_nar_raceで除外)。

冪等性: data/jra_baba/{year}/{venue_code}_{kai}.csv が既に存在すれば再取得しない
(開催回終了後のデータは不変なので、fetch_pedigree.pyと同じ「存在すれば恒久スキップ」
方式でよい、--forceで強制再取得)。
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

from config.settings import LOG_DIR, RACE_RESULTS_DIR
from src.netkeiba_pipeline.discovery.tracks import (
    JRA_BABA_PDF_VENUE_SLUGS,
    JRA_TRACK_CODES,
    is_nar_race,
    jra_venue_and_kai_from_race_id,
)
from src.netkeiba_pipeline.parsers.jra_baba_parser import parse_baba_pdf
from src.netkeiba_pipeline.scrapers.jra_official_baba import fetch_baba_pdf
from src.netkeiba_pipeline.storage.paths import jra_baba_csv_path
from src.netkeiba_pipeline.utils.logging_conf import configure_logging


def collect_race_ids_from_race_results(date: str | None, race_id: str | None, logger: logging.Logger) -> list[str]:
    """--date指定時はdata/race_results/{year}/{date}.csvから、--race-id指定時はその1件を
    そのまま対象にする(fetch_pedigree.py等と違いnewspaper CSVではなくrace_resultsを見る
    - year/venue/kaiの導出にhorse_idは不要でrace_id自体があれば足りるため、より単純な
    race_results側を使う。ただしrace_results CSV自体はレース確定後にしか生成されない
    ことに注意)。"""
    if race_id:
        return [race_id]

    year = date[:4]
    path = RACE_RESULTS_DIR / year / f"{date}.csv"
    if not path.exists():
        logger.warning("%s が存在しない(先にrun_pilot.pyが必要) - skip", path.name)
        return []
    df = pd.read_csv(path, dtype=str, encoding="utf-8")
    return list(df["race_id"].dropna().astype(str).unique())


def collect_venue_kai_targets(race_ids: list[str], logger: logging.Logger) -> list[tuple[str, str, str]]:
    """race_id群から(year, venue_code, kai)のユニークな組を集める(JRAのみ、NARは除外)。"""
    seen: dict[tuple[str, str, str], None] = {}
    for race_id in race_ids:
        if is_nar_race(race_id):
            logger.info("race_id=%s: NAR race - skip (このスクリプトはJRA専用)", race_id)
            continue
        year = race_id[:4]
        venue_code, kai = jra_venue_and_kai_from_race_id(race_id)
        seen.setdefault((year, venue_code, kai), None)
    return list(seen.keys())


def fetch_baba_for_target(year: str, venue_code: str, kai: str, force: bool, logger: logging.Logger) -> str:
    """戻り値: "fetched" / "skipped"(既存キャッシュあり) / "not_yet_published"(403、
    開催回未終了) / "unparseable"(PDFは取得できたが0行、後述) / 呼び出し元でraiseされる
    その他の例外。

    2026-09-02判明の重要な注意: parse_baba_pdfが対応する表レイアウト("開催日次"+
    "測定月日"の単一の幅広テーブル)は2025年1月以降のPDFのみで、2024年以前は
    「週末ブロック単位」の別レイアウトのため0行を返す(例外にはならない)。この場合を
    "fetched"扱いで空CSVを保存すると、"存在すれば恒久スキップ"キャッシュにより誤って
    永続的に空データがキャッシュされてしまうため、0行の結果は"unparseable"として
    ファイルを保存せずスキップする(2024年以前向けの別パーサー実装は将来課題)。"""
    path = jra_baba_csv_path(year, venue_code, kai)
    if path.exists() and not force:
        return "skipped"

    venue_name = JRA_TRACK_CODES[venue_code]
    venue_slug = JRA_BABA_PDF_VENUE_SLUGS[venue_code]
    try:
        pdf_bytes = fetch_baba_pdf(year, venue_slug, int(kai))
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 403:
            return "not_yet_published"
        raise

    df = parse_baba_pdf(pdf_bytes, year=year, venue=venue_name, kai=kai)
    if df.empty:
        logger.warning(
            "year=%s venue=%s(%s) kai=%s: PDF取得できたが0行(未対応レイアウトの可能性、"
            "2024年以前で既知) - 空CSVはキャッシュせずskip", year, venue_code, venue_name, kai,
        )
        return "unparseable"

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("year=%s venue=%s(%s) kai=%s: saved %s (%d days)", year, venue_code, venue_name, kai, path, len(df))
    return "fetched"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JRA公式サイトのクッション値・含水率アーカイブPDFを取得し、"
        "data/jra_baba/{year}/{venue_code}_{kai}.csv へ開催回単位で保存する(事後データ、JRAのみ)。"
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--race-id", help="単一race_idが属する開催回を対象にする")
    target.add_argument("--date", help="この開催日(YYYYMMDD)の全race_idが属する開催回を対象にする")
    parser.add_argument(
        "--force", action="store_true", help="既存のdata/jra_baba/{year}/{venue_code}_{kai}.csvがあっても再取得する"
    )
    args = parser.parse_args()

    log_name = args.date if args.date else args.race_id
    configure_logging(LOG_DIR / f"fetch_jra_baba_{log_name}.log")
    logger = logging.getLogger("fetch_jra_baba")

    race_ids = collect_race_ids_from_race_results(args.date, args.race_id, logger)
    targets = collect_venue_kai_targets(race_ids, logger)
    logger.info("Collected %d unique (year, venue, kai) target(s) across %d race_ids", len(targets), len(race_ids))

    fetched, skipped, not_yet, unparseable, failed = [], [], [], [], []
    for year, venue_code, kai in targets:
        try:
            result = fetch_baba_for_target(year, venue_code, kai, args.force, logger)
        except Exception:
            logger.exception("year=%s venue=%s kai=%s: failed to fetch/parse baba PDF", year, venue_code, kai)
            failed.append((year, venue_code, kai))
            continue
        if result == "fetched":
            fetched.append((year, venue_code, kai))
        elif result == "skipped":
            skipped.append((year, venue_code, kai))
        elif result == "unparseable":
            unparseable.append((year, venue_code, kai))
        else:
            not_yet.append((year, venue_code, kai))

    print(
        f"jra_baba: fetched {len(fetched)}, skipped (already cached) {len(skipped)}, "
        f"not yet published (開催回未終了) {len(not_yet)}, unparseable (未対応レイアウト、"
        f"2024年以前で既知) {len(unparseable)}, failed {len(failed)} / {len(targets)} targets"
    )
    if not_yet:
        print(f"Not yet published ({len(not_yet)}): {not_yet}")
    if unparseable:
        print(f"Unparseable ({len(unparseable)}): {unparseable}")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
