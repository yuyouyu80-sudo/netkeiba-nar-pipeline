import re

from bs4 import BeautifulSoup

# netkeiba's newspaper page (Shutuba_Table / 馬柱) header - the same page already
# fetched by fetch_newspaper.py for the per-horse tables, but the race-level
# metadata (race name / course / surface / distance / start time) in the page
# header was never extracted since fetch_newspaper.py only needed per-horse rows.
# race_number is NOT scraped here - it is the last 2 digits of race_id itself
# (netkeiba race_id = YYYY + venue(2) + kai(2) + nichime(2) + race_number(2)).
SURFACE_DISTANCE_RE = re.compile(r"(芝|ダ)\D*?(\d+)m")
START_TIME_RE = re.compile(r"(\d{1,2}:\d{2})発走")
FIELD_SIZE_RE = re.compile(r"^(\d+)頭$")


def parse_race_header(html: str, race_id: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    # JRAはh1.RaceName、NARは同じクラスがdiv.RaceNameに付与されるため、タグ指定なしで
    # クラス名のみに絞る(実データで両方1件ヒットのみを確認済み)。
    name_el = soup.select_one(".RaceName")
    race_name = name_el.get_text(strip=True) if name_el else None
    if not race_name:
        raise ValueError(f"race_id={race_id}: h1.RaceName not found - page structure may have changed")

    data01 = soup.select_one(".RaceData01")
    text01 = data01.get_text(" ", strip=True) if data01 is not None else ""
    surface_match = SURFACE_DISTANCE_RE.search(text01)
    start_time_match = START_TIME_RE.search(text01)

    data02 = soup.select_one(".RaceData02")
    spans = [s.get_text(strip=True) for s in data02.find_all("span")] if data02 is not None else []
    # netkeiba's standard "N回 開催地 N日目 ..." order - racecourse is always the
    # 2nd span (same convention as race_result_parser._parse_metadata's
    # r"\d+回(\D+?)\d+日目" regex, just read via the DOM here since this page's
    # markup already separates each field into its own <span>).
    racecourse = spans[1] if len(spans) > 1 else None
    field_size = None
    for s in spans:
        m = FIELD_SIZE_RE.match(s)
        if m:
            field_size = int(m.group(1))
            break

    return {
        "race_id": race_id,
        "race_name": race_name,
        "racecourse": racecourse,
        "race_number": int(race_id[-2:]),
        "surface": surface_match.group(1) if surface_match else None,
        "distance_m": int(surface_match.group(2)) if surface_match else None,
        "start_time": start_time_match.group(1) if start_time_match else None,
        "field_size": field_size,
    }
