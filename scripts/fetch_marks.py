"""mark_list.html(本紙・CP・その他の予想印)を取得し、既存のnewspaper CSV
(data/newspaper/{jra,nar}/{race_id}.csv)へ列を追加・上書き更新するスクリプト。

mark_list.htmlの印テーブルはページ読み込み後にJavaScriptがDOM操作で描画するため、このプロジェクト
で唯一Playwrightを使う(要: `pip install -r requirements.txt` 後に一度だけ
`python -m playwright install chromium`)。既存のrequestsベースのlogin()で得たCookieを
Playwrightのブラウザコンテキストに引き継いで認証状態を再現する
(src/netkeiba_pipeline/scrapers/mark_list.py)。

前提: 対象race_idについて data/newspaper/{jra,nar}/{race_id}.csv が
scripts/fetch_newspaper.py で既に生成済みであること(refresh_bias.pyと同じ前提)。このスクリプトは
新規行を作らず、既存行に以下の列を追加・上書きする:
- mark_raw_{専門家名}: 専門家ごとの生の印(◎○▲△☆、無印は空文字)。専門家の顔ぶれはレースごとに
  変動するため、このレースのCSVに実際に載っている専門家の分だけ列が追加される。
- mark_honshi / mark_cp: 「本紙」「CP予想」という名前の専門家の印(該当専門家がそのレースに
  掲載されていなければ空)。
- mark_other: 「本紙」「CP予想」を除く全専門家の印を、馬ごとに◎=6/○=4/▲=3/△=2/☆=0.5点で
  スコアリングして合計し、レース内で順位付け(同点は同順位)した結果。1〜4位に◎○▲△、
  5位以下は無印、0点(=本紙・CP予想以外の誰からも印をもらっていない馬)は★
  (詳細: src/netkeiba_pipeline/parsers/mark_list_parser.py の summarize_marks())。
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

from config.settings import LOG_DIR
from src.netkeiba_pipeline.auth.session import login
from src.netkeiba_pipeline.discovery.race_calendar import list_nar_race_ids, list_race_ids
from src.netkeiba_pipeline.parsers.mark_list_parser import parse_mark_list, summarize_marks
from src.netkeiba_pipeline.scrapers.mark_list import create_authenticated_context, fetch_mark_list_html
from src.netkeiba_pipeline.storage.paths import newspaper_csv_path
from src.netkeiba_pipeline.storage.writer import mark_scraped
from src.netkeiba_pipeline.utils.logging_conf import configure_logging


def refresh_marks_for_race(context, race_id: str, logger: logging.Logger) -> tuple[int, int]:
    """戻り値: (mark_otherが更新された馬数, CSV上の総馬数)。CSV未生成、または
    mark_list.html取得結果が空(専門家が1人も掲載されていない)なら(0, 総馬数)。"""
    path = newspaper_csv_path(race_id)
    if not path.exists():
        logger.warning("race_id=%s: %s が存在しない(先にfetch_newspaper.pyが必要) - skip", race_id, path.name)
        return 0, 0

    df = pd.read_csv(path, dtype=str, encoding="utf-8")
    if df.empty:
        return 0, 0

    html = fetch_mark_list_html(context, race_id)
    raw = parse_mark_list(html, race_id)
    if raw.empty:
        logger.warning("race_id=%s: mark_list.html取得結果が空(専門家印なし) - skip", race_id)
        return 0, len(df)

    summary = summarize_marks(raw)
    marks_df = raw.merge(summary, on="umaban", how="outer", validate="one_to_one")
    marks_df["umaban"] = marks_df["umaban"].astype(str)
    marks_df = marks_df.set_index("umaban")

    df["umaban"] = df["umaban"].astype(str)

    updated = 0
    for col in marks_df.columns:
        if col not in df.columns:
            df[col] = pd.NA
        new_vals = df["umaban"].map(marks_df[col])
        changed = new_vals.notna() & (df[col].fillna("").astype(str) != new_vals.fillna("").astype(str))
        updated = max(updated, int(changed.sum()))
        df[col] = new_vals.where(new_vals.notna(), df[col])

    df.to_csv(path, index=False, encoding="utf-8")
    mark_scraped(race_id, data_type="marks")
    logger.info("race_id=%s: refreshed marks for %d/%d horses", race_id, updated, len(df))
    return updated, len(df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mark_list.html(本紙・CP・その他の予想印)を取得し、既存のnewspaper CSVへ"
        "mark_raw_*/mark_honshi/mark_cp/mark_otherを追加・上書き更新する(Playwright使用)。"
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
    configure_logging(LOG_DIR / f"fetch_marks_{log_name}.log")
    logger = logging.getLogger("fetch_marks")

    session = login(email, password)

    with sync_playwright() as pw:
        browser, context = create_authenticated_context(pw, session)
        try:
            if args.race_id:
                n, total = refresh_marks_for_race(context, args.race_id, logger)
                print(f"{args.race_id}: updated {n}/{total} horses")
                return

            list_ids = list_nar_race_ids if args.circuit == "nar" else list_race_ids
            race_ids = list_ids(session, args.date)
            logger.info("Found %d race_ids for %s (circuit=%s)", len(race_ids), args.date, args.circuit)

            total_updated, total_horses, missing, failed = 0, 0, [], []
            for race_id in race_ids:
                try:
                    n, total = refresh_marks_for_race(context, race_id, logger)
                except Exception:
                    # 1レースの取得失敗(タイムアウト等の一時的なネットワークエラーを含む)で
                    # --date一括実行全体を止めない。fetch_newspaper.pyの日付一括モードと同じ方針
                    # (失敗race_idを集めて最後に報告、再取得は--race-idで個別に行う)。
                    logger.exception("race_id=%s: failed to fetch/parse marks", race_id)
                    failed.append(race_id)
                    continue
                if total == 0:
                    missing.append(race_id)
                total_updated += n
                total_horses += total

            print(
                f"refreshed marks: {total_updated}/{total_horses} horses across "
                f"{len(race_ids) - len(missing) - len(failed)}/{len(race_ids)} races for {args.date}"
            )
            if missing:
                print(f"skipped (no existing newspaper csv or no marks): {len(missing)} -> {missing}")
            if failed:
                print(f"Failed ({len(failed)}): {failed}")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
