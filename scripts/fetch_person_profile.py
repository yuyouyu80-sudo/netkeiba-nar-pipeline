"""db.netkeiba.com/{jockey,trainer}/{person_id}/ (年度別成績・所属地・所属形態)を取得し、
data/{jockey,trainer}_profile/{person_id}.csv(person_id単位、1ファイル1行)へ保存する
スクリプト。--kind jockey / --kind trainer で対象を切り替える。

対象person_idは、既存の data/newspaper/{jra,nar}/{race_id}.csv の
bias_jockey_id / bias_trainer_id 列(2026-09-02のbias_parser.py拡張で追加、Tier0)から
--date/--race-id(--circuit)で指定したレース群を横断して集める。これはレース結果
(data/race_results、レース後にしか生成されない)ではなく、レース前に生成されるnewspaper
CSV由来なので、デビュー戦の騎手・初出走の管理馬でも対象に含められる。

--ids-file(2026-09-02追記、予想ファクター充足度マップの勝率検証用に追加): 1行1IDの
テキストファイルからperson_idを直接指定する第三の対象指定方法。過去のnewspaper CSVは
Tier0以前に取得されたものが大半でbias_jockey_id列を持たないため、既存576レース分を
新カラム目当てに再取得するのは非現実的(1レース約1分×576=長時間)。代わりに
data/race_results(レース後生成、jockey_id/trainer_id列は本スキーマ拡張以前から存在)
側から集めたユニークIDをこの方法で渡せば、過去分もまとめて年度別成績・所属情報の
バックフィルができる(その場合、取得できるのは「現在時点(取得日)の」年度別成績・
リーディング順位であり、過去の各レース時点のものではない近似値である点に注意)。

fetch_pedigree.py/fetch_horse_profile.pyとの決定的な違い: 年度別成績・リーディング順位は
レースのたびに変化する時系列データなので、**「存在すれば恒久スキップ」キャッシュにはしない**
(データ収集ロードマップのレビューで指摘済み)。同じperson_idを再度指定した場合は常に
再取得して上書きする(--forceオプション自体が無い、常にforce相当の動作)。
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
from src.netkeiba_pipeline.parsers.person_profile_parser import parse_person_profile
from src.netkeiba_pipeline.scrapers.profile import fetch_jockey_profile_html, fetch_trainer_profile_html
from src.netkeiba_pipeline.storage.paths import (
    jockey_profile_csv_path,
    newspaper_csv_path,
    trainer_profile_csv_path,
)
from src.netkeiba_pipeline.utils.logging_conf import configure_logging

_KIND_CONFIG = {
    "jockey": {
        "id_column": "bias_jockey_id",
        "fetch_html": fetch_jockey_profile_html,
        "csv_path": jockey_profile_csv_path,
    },
    "trainer": {
        "id_column": "bias_trainer_id",
        "fetch_html": fetch_trainer_profile_html,
        "csv_path": trainer_profile_csv_path,
    },
}


def collect_person_ids(race_ids: list[str], id_column: str, logger: logging.Logger) -> list[str]:
    """対象race_id群のnewspaper CSVからユニークなperson_id(bias_jockey_id/bias_trainer_id)を
    集める(ページ順、重複排除)。newspaper CSV未生成、またはid_column自体が無い
    (bias_parser.py拡張前に取得された古いnewspaper CSV)のrace_idはログを出してスキップする。"""
    seen: dict[str, None] = {}
    for race_id in race_ids:
        path = newspaper_csv_path(race_id)
        if not path.exists():
            logger.warning("race_id=%s: %s が存在しない(先にfetch_newspaper.pyが必要) - skip", race_id, path.name)
            continue
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
        if id_column not in df.columns:
            logger.warning(
                "race_id=%s: %s に%s列が無い(古いnewspaper CSVの可能性、fetch_newspaper.pyの再実行が必要) - skip",
                race_id,
                path.name,
                id_column,
            )
            continue
        for person_id in df[id_column].dropna().astype(str).unique():
            if person_id and person_id != "nan":
                seen.setdefault(person_id, None)
    return list(seen.keys())


def fetch_profile_for_person(session, person_id: str, kind: str, logger: logging.Logger) -> None:
    config = _KIND_CONFIG[kind]
    html = config["fetch_html"](session, person_id)
    df = parse_person_profile(html, person_id, kind)

    path = config["csv_path"](person_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("%s_id=%s: saved %s", kind, person_id, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="db.netkeiba.com/{jockey,trainer}/{person_id}/ の年度別成績・所属情報を取得し、"
        "data/{jockey,trainer}_profile/{person_id}.csv へperson_id単位で保存する(常に上書き)。"
    )
    parser.add_argument("--kind", choices=["jockey", "trainer"], required=True, help="対象種別")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--race-id", help="単一race_idの出走馬の騎手・調教師だけを対象にする")
    target.add_argument("--date", help="この開催日(YYYYMMDD)の全race_idの出走馬の騎手・調教師を対象にする")
    target.add_argument("--ids-file", help="1行1IDのテキストファイルからperson_idを直接指定する(race_results等からのバックフィル用)")
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

    log_name = args.date if args.date else (args.race_id if args.race_id else Path(args.ids_file).stem)
    configure_logging(LOG_DIR / f"fetch_person_profile_{args.kind}_{log_name}.log")
    logger = logging.getLogger("fetch_person_profile")

    session = login(email, password)

    if args.ids_file:
        with open(args.ids_file, encoding="utf-8") as f:
            person_ids = [line.strip() for line in f if line.strip()]
        race_ids = []
        logger.info("Loaded %d %s_ids from %s", len(person_ids), args.kind, args.ids_file)
    else:
        if args.race_id:
            race_ids = [args.race_id]
        else:
            list_ids = list_nar_race_ids if args.circuit == "nar" else list_race_ids
            race_ids = list_ids(session, args.date)
            logger.info("Found %d race_ids for %s (circuit=%s)", len(race_ids), args.date, args.circuit)

        id_column = _KIND_CONFIG[args.kind]["id_column"]
        person_ids = collect_person_ids(race_ids, id_column, logger)
        logger.info("Collected %d unique %s_ids across %d race_ids", len(person_ids), args.kind, len(race_ids))

    fetched, failed = [], []
    for person_id in person_ids:
        try:
            fetch_profile_for_person(session, person_id, args.kind, logger)
        except Exception:
            logger.exception("%s_id=%s: failed to fetch/parse profile", args.kind, person_id)
            failed.append(person_id)
            continue
        fetched.append(person_id)

    print(
        f"{args.kind}_profile: fetched {len(fetched)}, failed {len(failed)} / "
        f"{len(person_ids)} {args.kind}_ids across {len(race_ids)} race_ids"
    )
    if failed:
        print(f"Failed ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
