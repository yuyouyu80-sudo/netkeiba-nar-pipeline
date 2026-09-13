# -*- coding: utf-8 -*-
"""point-in-time provenance監査(2026-09-06、改善計画Phase 0-B)。

`data/jra_pipeline/factor_database.json`の母集団(検証済みレース)について、
`data/_manifest/scraped_race_ids.csv`のnewspaper取得記録(scraped_at)を突き合わせ、
「開催日当日の発走後に取得された疑いのあるレース」「取得記録自体が無いレース」を
洗い出す。取得記録が無いレース・発走後取得レースは、near-real-timeで確定していく
netkeiba側の集計列(ca_*/data_*/surf_*等、累積成績)に当該レース自身の結果が
混入している(リークしている)可能性を排除できないため、モデル母集団の信頼性を
判断する材料として記録する。

JRAのレースは開催日16:30頃までに全レースが終わるため、`LAST_RACE_CUTOFF_JST`を
それより余裕を持たせた17:00 JSTとし、それ以降のnewspaper取得は「発走後取得の疑い」
とする(netkeibaのnewspaper.htmlは前日〜当日早朝に取得するのが通常の運用フロー)。

出力: `data/jra_pipeline/provenance_audit_report.json`
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FACTOR_DB_PATH = PROJECT_ROOT / "data" / "jra_pipeline" / "factor_database.json"
MANIFEST_PATH = PROJECT_ROOT / "data" / "_manifest" / "scraped_race_ids.csv"
OUT_PATH = PROJECT_ROOT / "data" / "jra_pipeline" / "provenance_audit_report.json"

JST = timezone(timedelta(hours=9))
LAST_RACE_CUTOFF_JST_HOUR = 17  # この時刻(JST)以降の取得は「発走後取得の疑い」


def _load_population(factor_db_path: Path) -> pd.DataFrame:
    data = json.loads(factor_db_path.read_text(encoding="utf-8"))
    rows = [
        {"race_id": r["race_id"], "kaisai_date": r["kaisai_date"], "race_type": r.get("race_type")}
        for r in data["races"]
    ]
    return pd.DataFrame(rows)


def _load_manifest_newspaper(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path, dtype={"race_id": str})
    df = df[(df["data_type"] == "newspaper") & (df["status"] == "success")].copy()
    df["scraped_at"] = pd.to_datetime(df["scraped_at"], utc=True, errors="coerce")
    # 同一race_idが複数回成功記録されている場合は最初の成功取得を採用
    df = df.sort_values("scraped_at").drop_duplicates("race_id", keep="first")
    return df[["race_id", "scraped_at"]]


def audit(factor_db_path: Path = FACTOR_DB_PATH, manifest_path: Path = MANIFEST_PATH) -> dict:
    pop = _load_population(factor_db_path)
    manifest = _load_manifest_newspaper(manifest_path)

    merged = pop.merge(manifest, on="race_id", how="left")
    merged["kaisai_date_dt"] = pd.to_datetime(merged["kaisai_date"], format="%Y%m%d")

    def _classify(row) -> str:
        if pd.isna(row["scraped_at"]):
            return "no_record"
        scraped_jst = row["scraped_at"].tz_convert(JST)
        kaisai = row["kaisai_date_dt"]
        cutoff = pd.Timestamp(
            year=kaisai.year, month=kaisai.month, day=kaisai.day,
            hour=LAST_RACE_CUTOFF_JST_HOUR, tz=JST,
        )
        if scraped_jst.normalize() > kaisai.tz_localize(JST if kaisai.tzinfo is None else None):
            return "fetched_after_kaisai_date"
        if scraped_jst > cutoff:
            return "fetched_same_day_after_cutoff"
        return "fetched_before_cutoff"

    merged["provenance_class"] = merged.apply(_classify, axis=1)

    counts = merged["provenance_class"].value_counts().to_dict()
    total = len(merged)
    suspect = merged[merged["provenance_class"].isin(
        ["fetched_after_kaisai_date", "fetched_same_day_after_cutoff"]
    )]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "population_source": str(factor_db_path.relative_to(PROJECT_ROOT)),
        "manifest_source": str(manifest_path.relative_to(PROJECT_ROOT)),
        "total_races": total,
        "counts_by_class": counts,
        "coverage_pct": round(100.0 * (total - counts.get("no_record", 0)) / total, 1) if total else None,
        "suspect_leak_race_ids": sorted(suspect["race_id"].tolist()),
        "no_record_race_ids": sorted(
            merged[merged["provenance_class"] == "no_record"]["race_id"].tolist()
        ),
        "notes": (
            "fetched_after_kaisai_date/fetched_same_day_after_cutoffは、newspaper取得時点で"
            "当該レースの一部集計列(ca_*/data_*/surf_*等)に自レースの結果が混入している"
            "(リーク)可能性を排除できないレース。no_recordはmanifestに取得記録が無く、"
            "取得タイミングの検証自体ができないレース。感度分析(これらを除いた母集団での"
            "再評価)を推奨。"
        ),
    }
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-db", type=Path, default=FACTOR_DB_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    report = audit(args.factor_db, args.manifest)
    print(f"total_races={report['total_races']}  coverage_pct={report['coverage_pct']}")
    for cls, n in sorted(report["counts_by_class"].items()):
        print(f"  {cls}: {n}")
    print(f"suspect_leak_race_ids: {len(report['suspect_leak_race_ids'])}")
    print(f"wrote {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
