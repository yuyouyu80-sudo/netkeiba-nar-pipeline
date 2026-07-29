import re

import pandas as pd
from bs4 import BeautifulSoup

HORSE_ID_RE = re.compile(r"/horse/(\d+)")

_STAT_FIELDS = [
    "win_1st",
    "win_2nd",
    "win_3rd",
    "win_out",
    "runs",
    "win_rate",
    "place2_rate",
    "place3_rate",
    "win_return_rate",
    "place_return_rate",
]


_CATEGORY_TABLE_COLUMNS = [
    "race_id",
    "category_type",
    "umaban",
    "category_label",
    "horse_id",
    "horse_name",
    *_STAT_FIELDS,
]


def parse_horse_category_table(html: str, race_id: str, category_type: str, source: str = "") -> pd.DataFrame:
    """table#table_sort_back, one row per horse for a single stat category
    (mode=concerned, and surf_summary.html's key1+key2 combo pages). Unlike
    course_analysis_parser.parse_horse_stat_table, this resolves cells by
    CSS class rather than raw <td> position: some of these pages put
    td.Horse_Info right after the checkmark cell, others put it last, so a
    fixed-index read silently grabs the wrong cell on half of them.

    Some race/combo pairings legitimately have nothing to show (e.g. a
    newcomer race has no pedigree-training combo history) - the table is
    then entirely absent or empty, which returns an empty frame rather than
    raising."""
    soup = BeautifulSoup(html, "lxml")
    label = f"race_id={race_id} source={source or category_type}"

    paywall = soup.select_one(".Premium_Regist_Box")
    if paywall is not None:
        raise ValueError(
            f"{label}: Premium_Regist_Box paywall present - this session does not have "
            "full premium access (only a partial, truncated table would be captured)."
        )

    table = soup.select_one("table#table_sort_back")
    if table is None:
        return pd.DataFrame(columns=_CATEGORY_TABLE_COLUMNS)

    rows = table.select("tbody > tr")
    if not rows:
        return pd.DataFrame(columns=_CATEGORY_TABLE_COLUMNS)

    records = []
    for tr in rows:
        cells = tr.find_all("td", recursive=False)
        umaban_td = cells[0] if cells else None
        category_td = tr.find("td", class_="DataTitle_Cell")
        horse_td = tr.find("td", class_=lambda c: bool(c) and "Horse_Info" in c.split())
        if umaban_td is None or category_td is None or horse_td is None:
            raise ValueError(f"{label}: could not find umaban/category/horse cell in a row")

        horse_a = horse_td.find("a", href=True)
        horse_id_match = HORSE_ID_RE.search(horse_a["href"]) if horse_a else None

        excluded = {id(umaban_td), id(category_td), id(horse_td)}
        checkmark_td = tr.find("td", class_="CheckMark")
        if checkmark_td is not None:
            excluded.add(id(checkmark_td))
        # A scratched (取消) horse replaces the checkmark cell's "CheckMark"
        # class with "Cancel_Txt" - exclude that too so it isn't miscounted
        # as one of the 10 stat cells.
        cancel_td = tr.find("td", class_="Cancel_Txt")
        if cancel_td is not None:
            excluded.add(id(cancel_td))
        stat_tds = [td for td in cells if id(td) not in excluded]
        if len(stat_tds) != len(_STAT_FIELDS):
            raise ValueError(
                f"{label}: expected {len(_STAT_FIELDS)} stat <td> per row, got {len(stat_tds)} - "
                "page structure may have changed"
            )

        record = {
            "race_id": race_id,
            "category_type": category_type,
            "umaban": umaban_td.get_text(strip=True),
            "category_label": category_td.get_text(strip=True),
            "horse_id": horse_id_match.group(1) if horse_id_match else "",
            "horse_name": horse_a.get_text(strip=True) if horse_a else "",
        }
        record.update(zip(_STAT_FIELDS, (td.get_text(strip=True) for td in stat_tds)))
        records.append(record)

    return pd.DataFrame.from_records(records)


def parse_data_breakdown(html: str, race_id: str, prefix: str, num_slots: int) -> pd.DataFrame:
    """table.Course_Result_All (race/data.html?mode=distance|course|condition|
    others|cushion|baba_water): each horse has one tr.HorseList row (rowspan
    across umaban/checkmark/horse-info) followed by exactly `num_slots`
    sibling rows. Each sibling row is either a category-stat row
    (td.Data_Title + 10 stat cells) or, for mode=others' final slot, a
    free-text note row (plain first <td> + a single colspan <td>, e.g.
    "馬体重" / "連対時馬体重452kg～464kg") - handled generically here so one
    parser covers all six modes without knowing their category semantics.

    Some breakdown modes don't apply to every race (e.g. mode=cushion is
    turf-only, so a dirt race has no cushion-value table at all) - that is
    real, not a scrape failure, so an empty (umaban-only) frame is returned
    rather than raising."""
    soup = BeautifulSoup(html, "lxml")
    label = f"race_id={race_id} source={prefix}"

    table = soup.select_one("table.Course_Result_All")
    if table is None:
        return pd.DataFrame(columns=["umaban"])

    horse_rows = table.select("tbody > tr.HorseList")
    if not horse_rows:
        return pd.DataFrame(columns=["umaban"])

    records = []
    for hr in horse_rows:
        num_td = hr.find("td", class_="Num")
        info_td = hr.find("td", class_="Horse_Info")
        if num_td is None or info_td is None:
            raise ValueError(f"{label}: could not find umaban/horse-info cell in a horse row")

        horse_a = info_td.find("a", href=True)
        horse_id_match = HORSE_ID_RE.search(horse_a["href"]) if horse_a else None

        record = {
            "umaban": num_td.get_text(strip=True),
            f"{prefix}_horse_id": horse_id_match.group(1) if horse_id_match else "",
            f"{prefix}_horse_name": horse_a.get_text(strip=True) if horse_a else "",
        }

        slot_rows = []
        sib = hr.find_next_sibling("tr")
        while sib is not None and "HorseList" not in (sib.get("class") or []):
            slot_rows.append(sib)
            sib = sib.find_next_sibling("tr")

        if len(slot_rows) != num_slots:
            raise ValueError(
                f"{label}: expected {num_slots} category rows per horse, got {len(slot_rows)} - "
                "page structure may have changed"
            )

        for i, row in enumerate(slot_rows, start=1):
            slot_prefix = f"{prefix}_slot{i}"
            title_td = row.find("td", class_="Data_Title")
            cells = row.find_all("td", recursive=False)
            if title_td is not None:
                stat_tds = [td for td in cells if td is not title_td]
                if len(stat_tds) != len(_STAT_FIELDS):
                    raise ValueError(
                        f"{label}: expected {len(_STAT_FIELDS)} stat <td> in slot {i}, got {len(stat_tds)}"
                    )
                record[f"{slot_prefix}_label"] = title_td.get_text(strip=True)
                record[f"{slot_prefix}_note"] = ""
                record.update(
                    {f"{slot_prefix}_{field}": td.get_text(strip=True) for field, td in zip(_STAT_FIELDS, stat_tds)}
                )
            else:
                if len(cells) < 2:
                    raise ValueError(f"{label}: unrecognized category row shape in slot {i}")
                record[f"{slot_prefix}_label"] = cells[0].get_text(strip=True)
                record[f"{slot_prefix}_note"] = cells[1].get_text(strip=True)
                for field in _STAT_FIELDS:
                    record[f"{slot_prefix}_{field}"] = ""

        records.append(record)

    return pd.DataFrame.from_records(records)
