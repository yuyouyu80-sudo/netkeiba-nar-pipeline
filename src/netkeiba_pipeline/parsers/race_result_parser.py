import re

import pandas as pd
from bs4 import BeautifulSoup

# netkeiba wraps some result-table header/data cells in a non-standard
# <diary_snap_cut> tag. lxml's HTML parser mishandles it and silently drops
# the <th>/<td> elements nested inside (confirmed by comparing parsed output
# with/without stripping it), so it must be removed before parsing.
_DIARY_SNAP_CUT_RE = re.compile(r"</?diary_snap_cut>")

_ID_HREF_RE = {
    "horse": re.compile(r"/horse/(\d+)/"),
    # jockey/trainer/owner use \w+ (not \d+): NAR-licensed jockeys carry alphanumeric
    # IDs (e.g. "a05df"), confirmed against real race.netkeiba.com/nar.netkeiba.com
    # pages. \d+ is a strict subset of \w+, so this is a no-op for JRA's numeric IDs.
    "jockey": re.compile(r"/jockey/result/recent/(\w+)/"),
    "trainer": re.compile(r"/trainer/result/recent/(\w+)/"),
    "owner": re.compile(r"/owner/result/recent/(\w+)/"),
}

# Column layout of table.race_table_01, confirmed against a real race page.
# Indices 9-13 (5 time-index cells, premium-only) and 19-21 (training time /
# stable comment / remarks, premium-only icons) are intentionally skipped.
_COL = {
    "finish_pos": 0,
    "waku": 1,
    "umaban": 2,
    "horse": 3,
    "sex_age": 4,
    "kinryo": 5,
    "jockey": 6,
    "time": 7,
    "margin": 8,
    "passing_order": 14,
    "last_3f": 15,
    "odds_final": 16,
    "popularity": 17,
    "weight": 18,
    "trainer": 22,
    "owner": 23,
    "prize": 24,
}
_EXPECTED_TD_COUNT = 25


def _extract_id(cell, kind: str) -> str | None:
    a = cell.find("a", href=True)
    if not a:
        return None
    match = _ID_HREF_RE[kind].search(a["href"])
    return match.group(1) if match else None


def _parse_metadata(soup: BeautifulSoup, race_id: str) -> dict:
    racedata = soup.select_one("dl.racedata")
    if racedata is None:
        raise ValueError(f"race_id={race_id}: dl.racedata metadata block not found")

    dt = racedata.select_one("dt")
    race_number_match = re.search(r"(\d+)", dt.get_text(strip=True)) if dt else None

    h1 = racedata.select_one("h1")
    race_name = h1.get_text(strip=True) if h1 else None

    span = racedata.select_one("p span")
    span_text = span.get_text(" ", strip=True) if span else ""
    surface_match = re.search(r"(芝|ダ)\D*?(\d+)m", span_text)
    weather_match = re.search(r"天候\s*:\s*(\S+)", span_text)
    going_match = re.search(r"(芝|ダート)\s*:\s*(\S+)", span_text)
    start_time_match = re.search(r"発走\s*:\s*(\S+)", span_text)

    smalltxt = soup.select_one("p.smalltxt")
    smalltxt_text = smalltxt.get_text(strip=True) if smalltxt else ""
    date_match = re.search(r"(\d{4})年(\d{2})月(\d{2})日", smalltxt_text)
    racecourse_match = re.search(r"\d+回(\D+?)\d+日目", smalltxt_text)

    return {
        "race_number": int(race_number_match.group(1)) if race_number_match else None,
        "race_name": race_name,
        "surface": surface_match.group(1) if surface_match else None,
        "distance_m": int(surface_match.group(2)) if surface_match else None,
        "weather": weather_match.group(1) if weather_match else None,
        "going": going_match.group(2) if going_match else None,
        "start_time": start_time_match.group(1) if start_time_match else None,
        "race_date": (
            f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
            if date_match
            else None
        ),
        "racecourse": racecourse_match.group(1) if racecourse_match else None,
    }


def parse_race_result(html: str, race_id: str) -> pd.DataFrame:
    cleaned = _DIARY_SNAP_CUT_RE.sub("", html)
    soup = BeautifulSoup(cleaned, "lxml")

    metadata = _parse_metadata(soup, race_id)

    table = soup.select_one("table.race_table_01")
    if table is None:
        raise ValueError(f"race_id={race_id}: table.race_table_01 not found - page structure may have changed")

    rows = table.find_all("tr")[1:]  # skip header row
    if not rows:
        raise ValueError(f"race_id={race_id}: result table has no data rows")

    records = []
    for tr in rows:
        tds = tr.find_all("td", recursive=False)
        if len(tds) != _EXPECTED_TD_COUNT:
            raise ValueError(
                f"race_id={race_id}: expected {_EXPECTED_TD_COUNT} <td> per row, "
                f"got {len(tds)} - page structure may have changed"
            )

        horse_cell = tds[_COL["horse"]]
        jockey_cell = tds[_COL["jockey"]]
        trainer_cell = tds[_COL["trainer"]]
        owner_cell = tds[_COL["owner"]]

        horse_id = _extract_id(horse_cell, "horse")
        jockey_id = _extract_id(jockey_cell, "jockey")
        trainer_id = _extract_id(trainer_cell, "trainer")
        owner_id = _extract_id(owner_cell, "owner")
        if horse_id is None or jockey_id is None or trainer_id is None or owner_id is None:
            raise ValueError(
                f"race_id={race_id}: could not extract horse_id/jockey_id/trainer_id/"
                "owner_id from a row"
            )

        records.append(
            {
                "race_id": race_id,
                **metadata,
                "finish_pos": tds[_COL["finish_pos"]].get_text(strip=True),
                "waku": tds[_COL["waku"]].get_text(strip=True),
                "umaban": tds[_COL["umaban"]].get_text(strip=True),
                "horse_id": horse_id,
                "horse_name": horse_cell.get_text(strip=True),
                "sex_age": tds[_COL["sex_age"]].get_text(strip=True),
                "kinryo": tds[_COL["kinryo"]].get_text(strip=True),
                "jockey_id": jockey_id,
                "jockey_name": jockey_cell.get_text(strip=True),
                "time": tds[_COL["time"]].get_text(strip=True),
                "margin": tds[_COL["margin"]].get_text(strip=True),
                "passing_order": tds[_COL["passing_order"]].get_text(strip=True),
                "last_3f": tds[_COL["last_3f"]].get_text(strip=True),
                "odds_final": tds[_COL["odds_final"]].get_text(strip=True),
                "popularity": tds[_COL["popularity"]].get_text(strip=True),
                "weight": tds[_COL["weight"]].get_text(strip=True),
                "trainer_id": trainer_id,
                "trainer_name": trainer_cell.get_text(strip=True),
                "owner_id": owner_id,
                "owner_name": owner_cell.get_text(strip=True),
                "prize": tds[_COL["prize"]].get_text(strip=True),
            }
        )

    return pd.DataFrame.from_records(records)


_LAP_SEGMENT_RE = re.compile(r"\s*-\s*")


def parse_lap_times(html: str, race_id: str) -> pd.DataFrame:
    """Parses table.result_table_02[summary="ラップタイム"]'s "ラップ" row (先頭馬基準の
    200mごとの区間タイム、race_lap_cellの"11.8 - 10.7 - ..."形式)。

    一部レース(主にNAR)はこの表自体は存在するが中身が空(<tr>が1つも無い、
    <caption>ラップタイム</caption>だけ)というケースが実際のfixtureで確認されている
    (掲載自体が無いレースがある、という仕様上の欠測)。この場合は例外にせず空の
    DataFrameを返す(呼び出し側はrace_result/payoutsの成功を妨げない)。表自体が
    見つからない場合はページ構造が変わった可能性が高いため例外にする。"""
    cleaned = _DIARY_SNAP_CUT_RE.sub("", html)
    soup = BeautifulSoup(cleaned, "lxml")

    empty = pd.DataFrame(columns=["race_id", "segment_index", "lap_time_sec"])

    table = soup.select_one('table.result_table_02[summary="ラップタイム"]')
    if table is None:
        raise ValueError(f"race_id={race_id}: ラップタイム table not found - page structure may have changed")

    lap_cell = None
    for tr in table.find_all("tr"):
        th = tr.find("th")
        if th is not None and th.get_text(strip=True) == "ラップ":
            lap_cell = tr.find("td", class_="race_lap_cell")
            break
    if lap_cell is None:
        return empty  # このレースはラップタイム未掲載(欠測として許容)

    segments = [s for s in _LAP_SEGMENT_RE.split(lap_cell.get_text(strip=True)) if s]
    if not segments:
        return empty

    records = [
        {"race_id": race_id, "segment_index": i, "lap_time_sec": float(seg)}
        for i, seg in enumerate(segments, start=1)
    ]
    return pd.DataFrame.from_records(records)


# th class -> canonical bet-type label, per dl.pay_block > table.pay_table_01.
# 複勝/ワイド rows carry 2-3 <br>-separated combinations (fewer combos in
# small fields); every other bet type always carries exactly one.
_PAYOUT_BET_LABELS = {
    "tan": "単勝",
    "fuku": "複勝",
    "waku": "枠連",
    "uren": "馬連",
    "wide": "ワイド",
    "utan": "馬単",
    "sanfuku": "3連複",
    "santan": "3連単",
}


def parse_payouts(html: str, race_id: str) -> pd.DataFrame:
    """Parses dl.pay_block's two table.pay_table_01 tables (払い戻し): one row
    per (race_id, bet_type, rank) - rank distinguishes the 2-3 combinations a
    複勝/ワイド row can carry, and is always 1 for the single-combination bet
    types."""
    cleaned = _DIARY_SNAP_CUT_RE.sub("", html)
    soup = BeautifulSoup(cleaned, "lxml")

    tables = soup.select("dl.pay_block table.pay_table_01")
    if not tables:
        raise ValueError(f"race_id={race_id}: pay_table_01 not found - page structure may have changed")

    records = []
    for table in tables:
        for tr in table.find_all("tr"):
            th = tr.find("th")
            tds = tr.find_all("td", recursive=False)
            if th is None or len(tds) != 3:
                raise ValueError(
                    f"race_id={race_id}: expected <th> + 3 <td> per payout row, got "
                    f"th={'present' if th else 'missing'} tds={len(tds)} - page structure may have changed"
                )

            bet_code = (th.get("class") or [None])[0]
            bet_type = _PAYOUT_BET_LABELS.get(bet_code)
            if bet_type is None:
                raise ValueError(f"race_id={race_id}: unrecognized payout row class {bet_code!r}")

            combos = [c.strip() for c in tds[0].get_text("\n").split("\n") if c.strip()]
            payouts = [c.strip().replace(",", "") for c in tds[1].get_text("\n").split("\n") if c.strip()]
            ninkis = [c.strip() for c in tds[2].get_text("\n").split("\n") if c.strip()]
            if not (len(combos) == len(payouts) == len(ninkis)):
                raise ValueError(
                    f"race_id={race_id}: {bet_type} row has mismatched combination/payout/popularity "
                    f"counts ({len(combos)}/{len(payouts)}/{len(ninkis)})"
                )

            for rank, (combo, payout, ninki) in enumerate(zip(combos, payouts, ninkis), start=1):
                records.append(
                    {
                        "race_id": race_id,
                        "bet_type": bet_type,
                        "rank": rank,
                        "combination": combo,
                        "payout": int(payout),
                        "popularity": int(ninki),
                    }
                )

    return pd.DataFrame.from_records(records)
