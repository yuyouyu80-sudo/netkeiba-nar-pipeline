import re

import pandas as pd
from bs4 import BeautifulSoup

from src.netkeiba_pipeline.scrapers.course_analysis import CID_LABELS

_EXPECTED_TD_COUNT = 14


def parse_horse_stat_table(html: str, race_id: str, category_type: str, source: str = "") -> pd.DataFrame:
    """Shared parser for netkeiba's table#table_sort_back layout, confirmed
    identical across mode=courseanalysis (waku/running_style/jockey/trainer),
    mode=coursedata (sire/broodmare_sire), and surf_summary.html (speed_index).
    `source` is only used to make error messages identify which page failed."""
    soup = BeautifulSoup(html, "lxml")
    label = f"race_id={race_id} source={source or category_type}"

    # netkeiba truncates this table to 3 rows and shows a Premium_Regist_Box
    # up-sell instead of erroring when the session isn't a premium member -
    # fail loudly rather than silently persisting a partial field.
    paywall = soup.select_one(".Premium_Regist_Box")
    if paywall is not None:
        raise ValueError(
            f"{label}: Premium_Regist_Box paywall present - this session does not have "
            "full premium access (only a partial, truncated table would be captured)."
        )

    table = soup.select_one("table#table_sort_back")
    if table is None:
        raise ValueError(f"{label}: table#table_sort_back not found - page structure may have changed")

    rows = table.select("tbody tr")
    if not rows:
        raise ValueError(f"{label}: table has no data rows")

    records = []
    for tr in rows:
        tds = tr.find_all("td", recursive=False)
        if len(tds) != _EXPECTED_TD_COUNT:
            raise ValueError(
                f"{label}: expected {_EXPECTED_TD_COUNT} <td> per row, got {len(tds)} - "
                "page structure may have changed"
            )

        horse_cell = tds[13]
        horse_a = horse_cell.find("a", href=True)
        if horse_a is None:
            raise ValueError(f"{label}: could not find horse link in a row")
        horse_id_match = re.search(r"/horse/(\d+)", horse_a["href"])
        horse_id = horse_id_match.group(1) if horse_id_match else None

        records.append(
            {
                "race_id": race_id,
                "category_type": category_type,
                "umaban": tds[0].get_text(strip=True),
                "category_label": tds[2].get_text(strip=True),
                "win_1st": tds[3].get_text(strip=True),
                "win_2nd": tds[4].get_text(strip=True),
                "win_3rd": tds[5].get_text(strip=True),
                "win_out": tds[6].get_text(strip=True),
                "runs": tds[7].get_text(strip=True),
                "win_rate": tds[8].get_text(strip=True),
                "place2_rate": tds[9].get_text(strip=True),
                "place3_rate": tds[10].get_text(strip=True),
                "win_return_rate": tds[11].get_text(strip=True),
                "place_return_rate": tds[12].get_text(strip=True),
                "horse_id": horse_id,
                "horse_name": horse_a.get_text(strip=True),
            }
        )

    return pd.DataFrame.from_records(records)


def parse_course_analysis(html: str, race_id: str, cid: int) -> pd.DataFrame:
    category_type = CID_LABELS.get(cid, str(cid))
    return parse_horse_stat_table(html, race_id, category_type, source=f"courseanalysis cid={cid}")
