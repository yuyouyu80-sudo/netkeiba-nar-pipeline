# -*- coding: utf-8 -*-
"""NAR馬柱(newspaper)データが発走前に取得されたか判定する(Phase B、2026-08-04)。

背景: 2026-08-01のsearch300再検証で、当時の検証用253レースの77%が発走後に馬柱CSVを
取得したものだったと判明した(data/nar_pipeline/winner_box4_nar.json の
data_provenance_caveat_2026_08_01)。原因はfetch_newspaper.pyがこれまでmanifest
(data/_manifest/scraped_race_ids.csv)に一切書き込んでおらず、「いつ取得したか」を
記録していなかったこと。fetch_newspaper.py側の対応(mark_scraped呼び出し追加)は
別途済み。このスクリプトは、その記録が溜まった後に「発走前取得(検証に使ってよい)」
と「発走後取得(除外/警告)」を機械的に判定するためのもの。

既存253レース分は取得時刻が記録されていないため遡って判定できない(過去分は
"unknown_not_recorded"のまま)。今後fetch_newspaper.pyで取得したレースから
判定が効くようになる。

使い方:
    python scripts/nar_model/verify_provenance.py
    python scripts/nar_model/verify_provenance.py --out custom_path.csv
"""
import argparse
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "_manifest" / "scraped_race_ids.csv"
RESULTS_DIR = PROJECT_ROOT / "data" / "race_results" / "nar" / "2026"
DEFAULT_OUT = PROJECT_ROOT / "data" / "nar_pipeline" / "newspaper_provenance.csv"
JST = ZoneInfo("Asia/Tokyo")


def _load_race_start_times() -> pd.DataFrame:
    """data/race_results/nar/2026/*.csv から race_id ごとの発走時刻(JST)を集める。"""
    frames = []
    for path in sorted(RESULTS_DIR.glob("*.csv")):
        df = pd.read_csv(path, dtype=str, usecols=["race_id", "race_date", "start_time"])
        frames.append(df.drop_duplicates(subset="race_id"))
    if not frames:
        return pd.DataFrame(columns=["race_id", "race_start_jst"])
    all_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="race_id")
    all_df["race_start_jst"] = pd.to_datetime(
        all_df["race_date"] + " " + all_df["start_time"], errors="coerce"
    ).dt.tz_localize(JST)
    return all_df[["race_id", "race_start_jst"]]


def _load_newspaper_scraped_at() -> pd.DataFrame:
    """manifestからdata_type=newspaperの成功記録を読み、race_idごとに最新の
    scraped_at(=現在ディスク上のCSVを実際に書いた取得)だけを残す。
    fetch_newspaper.pyのwrite先は毎回上書きなので、再取得された場合は最後の
    成功が現在のファイル内容に対応する。"""
    if not MANIFEST_PATH.exists():
        return pd.DataFrame(columns=["race_id", "scraped_at_jst"])
    df = pd.read_csv(MANIFEST_PATH, dtype=str)
    df = df[(df["data_type"] == "newspaper") & (df["status"] == "success")]
    if df.empty:
        return pd.DataFrame(columns=["race_id", "scraped_at_jst"])
    df["scraped_at_utc"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df = df.sort_values("scraped_at_utc").drop_duplicates(subset="race_id", keep="last")
    df["scraped_at_jst"] = df["scraped_at_utc"].dt.tz_convert(JST)
    return df[["race_id", "scraped_at_jst"]]


def build_provenance_table() -> pd.DataFrame:
    starts = _load_race_start_times()
    scraped = _load_newspaper_scraped_at()

    if scraped.empty:
        return pd.DataFrame(
            columns=["race_id", "scraped_at_jst", "race_start_jst", "verdict", "minutes_after_start"]
        )

    merged = scraped.merge(starts, on="race_id", how="left")

    def _verdict(row) -> str:
        if pd.isna(row["race_start_jst"]):
            return "unknown_no_result_yet"
        return "pre_race" if row["scraped_at_jst"] < row["race_start_jst"] else "post_race"

    merged["verdict"] = merged.apply(_verdict, axis=1)
    merged["minutes_after_start"] = (
        (merged["scraped_at_jst"] - merged["race_start_jst"]).dt.total_seconds() / 60.0
    )
    return merged.sort_values("race_id")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NAR馬柱データの取得タイミング(発走前/発走後)を判定してCSVに出力する"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    table = build_provenance_table()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, encoding="utf-8")

    counts = table["verdict"].value_counts()
    print(f"races with recorded newspaper scraped_at: {len(table)}")
    for verdict, n in counts.items():
        print(f"  {verdict}: {n}")
    print(f"wrote {args.out}")

    n_total_manifest = len(_load_newspaper_scraped_at())
    print(
        f"\n注記: manifestにnewspaper取得記録があるレースは{n_total_manifest}件のみ"
        "(2026-08-04のfetch_newspaper.py修正より前の取得は記録が無く、遡って判定不可)。"
        "データが今後蓄積されるまではpre_race件数が少ないのが正常。"
    )


if __name__ == "__main__":
    main()
