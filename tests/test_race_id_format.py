from pathlib import Path

from bs4 import BeautifulSoup

from src.netkeiba_pipeline.discovery.race_calendar import RACE_ID_RE

FIXTURES = Path(__file__).parent / "fixtures"


def test_race_id_extraction_from_race_list_sub():
    html = (FIXTURES / "race_list_sub_20240106.html").read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    race_ids = []
    seen = set()
    for a in soup.find_all("a", href=True):
        match = RACE_ID_RE.search(a["href"])
        if match and match.group(1) not in seen:
            seen.add(match.group(1))
            race_ids.append(match.group(1))

    assert len(race_ids) > 0
    assert all(len(rid) == 12 for rid in race_ids)
    assert "202406010101" in race_ids
