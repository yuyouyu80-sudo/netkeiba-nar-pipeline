import re

import pandas as pd
from bs4 import BeautifulSoup, Tag

_WEIGHT_DIFF_RE = re.compile(r"^(?P<weight>\S+?)\((?P<diff>[+-]?\d+|\D*)\)$")
_BIRTHDAY_RE = re.compile(r"\d+月\d+日生まれ")
_PAST_FIELDS = [
    "date",
    "venue",
    "finish",
    "race_name",
    "race_class",
    "surface_distance",
    "time",
    "track_condition",
    "field_size",
    "umaban_in_race",
    "ninki",
    "jockey",
    "weight_carried",
    "corner_positions",
    "agari_3f",
    "horse_weight",
    "horse_weight_diff",
    "beaten_by",
    "birthday",
]


def _first_text(tag: Tag) -> str:
    for content in tag.contents:
        if isinstance(content, str) and content.strip():
            return content.strip()
    return tag.get_text(strip=True)


def _parse_past_cell(td: Tag) -> dict:
    """One <td class="Past"> = one past race. Format is a display grid, not
    structured data, so several sub-fields are extracted with light,
    best-effort parsing (space-token splits) that falls back to leaving a
    field blank rather than raising - historical rows legitimately vary
    (foreign races, scratches, dead heats).

    Debut/maiden horses with too little race history have netkeiba replace
    one slot with <td class="Rest"> (siblings + birthday) instead of a race -
    only the birthday is extracted from it, into that same slot's "birthday"
    field, per the site's own note that this appears "in place of" a past race."""
    empty = {field: "" for field in _PAST_FIELDS}

    if "Rest" in (td.get("class") or []):
        result = dict(empty)
        text = td.get_text(" ", strip=True)
        match = _BIRTHDAY_RE.search(text)
        if match:
            result["birthday"] = match.group(0)
        return result

    item = td.find("div", class_="Data_Item")
    if item is None:
        return empty

    result = dict(empty)

    data01 = item.find("div", class_="Data01")
    if data01:
        spans = data01.find_all("span")
        if spans:
            parts = spans[0].get_text(strip=True).split()
            result["date"] = parts[0] if parts else ""
            result["venue"] = parts[1] if len(parts) > 1 else ""
        if len(spans) > 1:
            result["finish"] = spans[1].get_text(strip=True)

    data02 = item.find("div", class_="Data02")
    if data02:
        a = data02.find("a")
        if a is not None:
            result["race_name"] = _first_text(a)
            grade_span = a.find("span")
            result["race_class"] = grade_span.get_text(strip=True) if grade_span else ""
        else:
            result["race_name"] = data02.get_text(strip=True)

    data05 = item.find("div", class_="Data05")
    if data05:
        tokens = data05.get_text(" ", strip=True).split()
        if len(tokens) == 3:
            result["surface_distance"], result["time"], result["track_condition"] = tokens
        else:
            result["surface_distance"] = " ".join(tokens)

    data03 = item.find("div", class_="Data03")
    if data03:
        tokens = data03.get_text(" ", strip=True).split()
        if len(tokens) >= 5:
            result["field_size"] = tokens[0].rstrip("頭")
            result["umaban_in_race"] = tokens[1].rstrip("番")
            result["ninki"] = tokens[2].rstrip("人")
            result["weight_carried"] = tokens[-1]
            result["jockey"] = "".join(tokens[3:-1])
        else:
            result["jockey"] = " ".join(tokens)

    data06 = item.find("div", class_="Data06")
    if data06:
        tokens = data06.get_text(" ", strip=True).split()
        if len(tokens) >= 1:
            result["corner_positions"] = tokens[0]
        if len(tokens) >= 2:
            result["agari_3f"] = tokens[1].strip("()")
        if len(tokens) >= 3:
            match = _WEIGHT_DIFF_RE.match(tokens[2])
            if match:
                result["horse_weight"] = match.group("weight")
                result["horse_weight_diff"] = match.group("diff")
            else:
                result["horse_weight"] = tokens[2]

    data07 = item.find("div", class_="Data07")
    if data07:
        result["beaten_by"] = data07.get_text(strip=True)

    return result


def parse_shutuba_past(html: str, race_id: str) -> pd.DataFrame:
    """table.Shutuba_Past5_Table: one row per horse, with up to 5 td.Past
    cells (前走..5走前, in that order). Returns one row per horse with
    columns past1_* (前走) .. past5_* (5走前)."""
    soup = BeautifulSoup(html, "lxml")
    # id="sort_table" pins this to the actual racecard table - the page also
    # embeds several same-classed table02..table08 elements inside a "how to
    # read this page" tutorial block elsewhere in the DOM, with sample/example
    # content (not real horse data) that must not be picked up here.
    table = soup.select_one("table.Shutuba_Past5_Table#sort_table")
    if table is None:
        raise ValueError(f"race_id={race_id}: table.Shutuba_Past5_Table#sort_table not found - page structure may have changed")

    rows = table.select("tbody tr.HorseList")
    if not rows:
        raise ValueError(f"race_id={race_id}: Shutuba_Past5_Table has no horse rows")

    records = []
    for tr in rows:
        umaban_td = tr.find("td", class_="Waku")
        if umaban_td is None:
            raise ValueError(f"race_id={race_id}: could not find umaban cell in Shutuba_Past5_Table row")

        record = {"umaban": umaban_td.get_text(strip=True)}
        past_tds = tr.select("td.Past, td.Rest")
        for i in range(5):
            prefix = f"past{i + 1}"
            fields = _parse_past_cell(past_tds[i]) if i < len(past_tds) else {f: "" for f in _PAST_FIELDS}
            for field, value in fields.items():
                record[f"{prefix}_{field}"] = value
        records.append(record)

    return pd.DataFrame.from_records(records)
