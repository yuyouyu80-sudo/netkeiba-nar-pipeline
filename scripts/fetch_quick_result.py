"""race.netkeiba.com(JRA)/nar.netkeiba.com(NAR)のresult.htmlから、同日中に確定する簡易結果
(1〜5着・単勝払戻)だけを取得する軽量スクリプト。

db.netkeiba.com(run_pilot.pyが使う正式な確定データ)は翌日にならないと反映されないため、
発走直後にその日のうちに「どのレースが終わっていて、何が来たか」を確認したい場合はこちらを使う。
正式な回収率検証(box_return_nar.py等)には使わず、あくまで速報表示専用。

まだ発走していない・結果ページが未掲載のレースは空扱いでスキップする(エラーではない)。

2026-08-15、JRA/NAR両対応の--circuitフラグを追加(旧名 fetch_quick_result_nar.py から改称)。
race_url()がrace_idからドメインを自動判別するため(race.netkeiba.com/nar.netkeiba.com)、
取得ロジック自体は元々circuit非依存だった。出力先だけがJRA/NARで異なる:
- --circuit省略時(JRA、既定): JRAレポート生成パイプライン(build_artifact.py等)と同じ
  セッション固有scratchpad(predict_pattern29.pyのOUT_DIRと同一)にquick_result_{date}.csvを書く。
- --circuit nar: 従来通りdata/nar_pipeline/quick_result_nar_{date}.csv(git管理下)に書く。
  ファイル名・出力先はこのスクリプトへ統合される前と完全互換。

出力: 1レース1行(finish1〜finish5 + tansho_payout/tansho_ninki)
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
from src.netkeiba_pipeline.parsers.quick_result_parser import parse_quick_result
from src.netkeiba_pipeline.scrapers.quick_result import fetch_quick_result_html
from src.netkeiba_pipeline.utils.logging_conf import configure_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NAR_DATA_DIR = PROJECT_ROOT / "data" / "nar_pipeline"
# JRAレポート生成パイプライン(build_artifact.py等)はセッション固有scratchpadにしか存在しない
# (predict_pattern29.pyのOUT_DIRと同じ場所に揃える)。
JRA_SCRATCHPAD = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="race.netkeiba.com/nar.netkeiba.comのresult.htmlから簡易結果(1〜5着・"
        "単勝払戻)を取得する。"
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--race-id", help="単一race_idのみ取得する")
    target.add_argument("--date", help="この開催日(YYYYMMDD)の全race_idを取得する")
    parser.add_argument(
        "--circuit", choices=["jra", "nar"], default="jra",
        help="出力先の切り替え(既定: jra)。--race-id指定時は表示のみ・書き込み無しのため無視される。",
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
    configure_logging(LOG_DIR / f"fetch_quick_result_{log_name}.log")
    logger = logging.getLogger("fetch_quick_result")

    session = login(email, password)
    if args.race_id:
        race_ids = [args.race_id]
    else:
        list_ids = list_nar_race_ids if args.circuit == "nar" else list_race_ids
        race_ids = list_ids(session, args.date)
        logger.info("Found %d race_ids for %s (circuit=%s)", len(race_ids), args.date, args.circuit)

    rows = []
    not_yet = []
    for race_id in race_ids:
        html = fetch_quick_result_html(session, race_id)
        df = parse_quick_result(html, race_id)
        if df.empty:
            not_yet.append(race_id)
            continue
        rows.append(df)
        logger.info(
            "race_id=%s: 1着 %s(%s) 単勝%s",
            race_id, df.iloc[0]["finish1_horse"], df.iloc[0]["finish1_umaban"],
            df.iloc[0]["tansho_payout"],
        )

    if args.race_id:
        if rows:
            print(rows[0].iloc[0].to_dict())
        else:
            print(f"{args.race_id}: まだ結果ページが無い(未発走、または未掲載)")
        return

    out_path = (
        NAR_DATA_DIR / f"quick_result_nar_{args.date}.csv" if args.circuit == "nar"
        else JRA_SCRATCHPAD / f"quick_result_{args.date}.csv"
    )
    if rows:
        result = pd.concat(rows, ignore_index=True)
        result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(
        f"quick results: {len(rows)}/{len(race_ids)} races written to {out_path.name} "
        f"for {args.date} (circuit={args.circuit})"
    )
    if not_yet:
        print(f"未発走・未掲載 ({len(not_yet)}): {not_yet}")


if __name__ == "__main__":
    main()
