import re

import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag

HORSE_ID_RE = re.compile(r"/horse/(\d+)")

_INDEX_TD_CLASSES = {
    "sk__max_index": "speed_max_index",
    "sk__average_index": "speed_avg_index_5races",
    "sk__max_distance_index": "speed_max_distance_index",
    "sk__max_course_index": "speed_max_course_index",
    "sk__index3": "speed_index_3races_ago",
    "sk__index2": "speed_index_2races_ago",
    "sk__index1": "speed_index_1race_ago",
    "sk__odds": "speed_odds",
    "sk__ninki": "speed_ninki",
    "sk__load_weight": "speed_weight_carried",
}


def _visible_value(td: Tag) -> str:
    """Each index cell hides a zero-padded sort key in a
    span.Sort_Function_Data_Hidden ahead of the actual displayed value (a
    link for indices tied to a specific past race, plain text otherwise, or
    "未" when there's no data) - only the displayed value is wanted here.
    A trailing "*" (netkeiba's marker that the average is based on fewer
    than 5 past races) is stripped so numeric values stay purely numeric;
    "未" has no "*" so it passes through unchanged."""
    hidden = td.find("span", class_="Sort_Function_Data_Hidden")
    if hidden is None:
        return td.get_text(strip=True)
    sib = hidden.next_sibling
    while isinstance(sib, NavigableString) and not sib.strip():
        sib = sib.next_sibling
    if sib is None:
        return ""
    if isinstance(sib, Tag):
        value = sib.get_text(strip=True)
    else:
        value = str(sib).strip()
    return value.rstrip("*")


_SPEED_INDEX_COLUMNS = [
    "umaban",
    "speed_horse_id",
    "speed_horse_name",
    "speed_sex_age",
    "speed_jockey",
    *_INDEX_TD_CLASSES.values(),
]


def parse_speed_index(html: str, race_id: str) -> pd.DataFrame:
    """table.SpeedIndex_Table (タイム指数出馬表): one row per horse. A race
    where every runner is making their debut (no past races to index, e.g.
    a 新馬 race) legitimately has no index table - an empty frame is
    returned rather than raising."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.SpeedIndex_Table")
    if table is None:
        return pd.DataFrame(columns=_SPEED_INDEX_COLUMNS)

    rows = table.select("tbody tr.HorseList")
    if not rows:
        return pd.DataFrame(columns=_SPEED_INDEX_COLUMNS)

    records = []
    for tr in rows:
        umaban_td = tr.find("td", class_="UmaBan")
        horse_td = tr.find("td", class_="Horse_Name")
        if umaban_td is None or horse_td is None:
            raise ValueError(f"race_id={race_id}: could not find umaban/horse cell in SpeedIndex_Table row")

        horse_a = horse_td.find("a", href=True)
        horse_id_match = HORSE_ID_RE.search(horse_a["href"]) if horse_a else None

        record = {
            "umaban": umaban_td.get_text(strip=True),
            "speed_horse_id": horse_id_match.group(1) if horse_id_match else "",
            "speed_horse_name": horse_a.get_text(strip=True) if horse_a else "",
        }

        sex_age_td = tr.find("td", class_="Txt_C")
        record["speed_sex_age"] = sex_age_td.get_text(strip=True) if sex_age_td else ""

        jockey_td = tr.find("td", class_="Jockey")
        jockey_a = jockey_td.find("a") if jockey_td else None
        record["speed_jockey"] = jockey_a.get_text(strip=True) if jockey_a else (
            jockey_td.get_text(strip=True) if jockey_td else ""
        )

        for td_class, field in _INDEX_TD_CLASSES.items():
            td = tr.find("td", class_=td_class)
            record[field] = _visible_value(td) if td is not None else ""

        records.append(record)

    return pd.DataFrame.from_records(records)
