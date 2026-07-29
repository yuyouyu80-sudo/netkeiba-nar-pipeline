import pandas as pd
from bs4 import BeautifulSoup

_EXPECTED_TD_COUNT = 12

# Maps a substring of the table's caption (e.g. "芝2000m 騎手ランキング　TOP20")
# to our category label. Matched by substring since the course/surface/distance
# prefix varies per race.
_CAPTION_TYPE_MAP = {
    "騎手": "jockey",
    "種牡馬": "sire",
    "調教師": "trainer",
}


def parse_ranking(html: str, race_id: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")

    # All three ranking tables (jockey/sire/trainer) live inside a single
    # shared div.Race_Ranking_Data wrapper, so each caption's own table must
    # be located via document-order traversal (find_next), not by scoping to
    # a per-table wrapper div (there isn't one).
    captions = soup.select("h3.DataTable_Caption")
    if not captions:
        raise ValueError(f"race_id={race_id}: no DataTable_Caption headings found - page structure may have changed")

    records = []
    for caption in captions:
        caption_text = caption.get_text(strip=True)
        ranking_type = next((v for k, v in _CAPTION_TYPE_MAP.items() if k in caption_text), None)
        if ranking_type is None:
            raise ValueError(f"race_id={race_id}: unrecognized ranking caption '{caption_text}'")

        table = caption.find_next("table")
        if table is None:
            raise ValueError(f"race_id={race_id}: no table found after ranking caption '{caption_text}'")

        rows = table.select("tbody tr")
        if not rows:
            raise ValueError(f"race_id={race_id}: ranking table '{caption_text}' has no data rows")

        for tr in rows:
            tds = tr.find_all("td", recursive=False)
            if len(tds) != _EXPECTED_TD_COUNT:
                raise ValueError(
                    f"race_id={race_id}: expected {_EXPECTED_TD_COUNT} <td> per ranking row, "
                    f"got {len(tds)} - page structure may have changed"
                )
            records.append(
                {
                    "race_id": race_id,
                    "ranking_type": ranking_type,
                    "rank": tds[0].get_text(strip=True),
                    "name": tds[1].get_text(strip=True),
                    "win_1st": tds[2].get_text(strip=True),
                    "win_2nd": tds[3].get_text(strip=True),
                    "win_3rd": tds[4].get_text(strip=True),
                    "win_out": tds[5].get_text(strip=True),
                    "runs": tds[6].get_text(strip=True),
                    "win_rate": tds[7].get_text(strip=True),
                    "place2_rate": tds[8].get_text(strip=True),
                    "place3_rate": tds[9].get_text(strip=True),
                    "win_return_rate": tds[10].get_text(strip=True),
                    "place_return_rate": tds[11].get_text(strip=True),
                }
            )

    return pd.DataFrame.from_records(records)
