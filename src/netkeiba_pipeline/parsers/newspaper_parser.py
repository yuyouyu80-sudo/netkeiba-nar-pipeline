import re

import pandas as pd
from bs4 import BeautifulSoup

HORSE_ID_RE = re.compile(r"/horse/(\d+)")
MARK_CODE_RE = re.compile(r"Icon_Mark_(\d+)")


def _horse_id(a_tag) -> str | None:
    if a_tag is None:
        return None
    match = HORSE_ID_RE.search(a_tag.get("href", ""))
    return match.group(1) if match else None


def has_predictrap_paywall(html: str) -> bool:
    """The per-horse predicted-lap table (Shutuba_Table.PredictRap_Table) is
    gated behind a subscription tier beyond regular premium membership: netkeiba
    truncates it to the first horse and fills the rest with table_dummy_*.png
    placeholder images instead of a Premium_Regist_Box, so it needs its own
    detection instead of the usual paywall check."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.PredictRap_Table")
    if table is None:
        return False
    return "table_dummy" in str(table)


_STABLE_COMMENT_COLUMNS = [
    "umaban",
    "waku",
    "horse_id",
    "horse_name",
    "stable_comment",
    "stable_comment_reporter",
    "stable_comment_rating_code",
]


def parse_stable_comments(html: str, race_id: str) -> pd.DataFrame:
    """Parses table.Stable_Comment (厩舎コメント). One row per horse. netkeiba
    only writes stable comments for a subset of races each day (typically
    the featured ones) - when the table is entirely absent that's real, not
    a scrape failure, so an empty frame is returned instead of raising."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.Stable_Comment")
    if table is None:
        return pd.DataFrame(columns=_STABLE_COMMENT_COLUMNS)

    rows = table.select("tbody tr") or table.find_all("tr")
    rows = [r for r in rows if r.find("td")]
    if not rows:
        raise ValueError(f"race_id={race_id}: Stable_Comment table has no data rows")

    records = []
    for tr in rows:
        tds = tr.find_all("td", recursive=False)
        if len(tds) != 5:
            raise ValueError(
                f"race_id={race_id}: expected 5 <td> per Stable_Comment row, got {len(tds)} - "
                "page structure may have changed"
            )
        waku_td, umaban_td, horse_td, comment_td, hyoka_td = tds

        horse_a = horse_td.find("a", href=True)
        dd = comment_td.find("dd")
        reporter_span = dd.find("span") if dd else None
        reporter = reporter_span.get_text(strip=True) if reporter_span else ""
        comment_full = dd.get_text(" ", strip=True) if dd else ""
        comment_text = comment_full.replace(reporter, "").strip() if reporter else comment_full

        mark_span = hyoka_td.find("span")
        mark_match = MARK_CODE_RE.search(" ".join(mark_span.get("class", []))) if mark_span else None

        records.append(
            {
                "umaban": umaban_td.get_text(strip=True),
                "waku": waku_td.get_text(strip=True),
                "horse_id": _horse_id(horse_a),
                "horse_name": horse_a.get_text(strip=True) if horse_a else "",
                "stable_comment": comment_text,
                "stable_comment_reporter": reporter,
                "stable_comment_rating_code": mark_match.group(1) if mark_match else "",
            }
        )

    return pd.DataFrame.from_records(records)


_OIKIRI_COLUMNS = [
    "umaban",
    "training_review",
    "training_date",
    "training_course",
    "training_track_condition",
    "training_rider",
    "training_times",
    "training_partner_comment",
    "training_position",
    "training_load",
    "training_critic",
    "training_rank",
    "_waku_check",
    "_horse_id_check",
    "_horse_name_check",
]


def _parse_oikiri_review_format(data_rows: list, race_id: str) -> pd.DataFrame:
    """Review-row + detail-row pairs (only used for the day's featured
    races): a rowspan=2 review row (waku/umaban/horse cells + a colspan
    written review) followed by a detail row (date/course/track/rider/
    times/etc)."""
    if len(data_rows) % 2 != 0:
        raise ValueError(
            f"race_id={race_id}: OikiriTable has {len(data_rows)} data rows, expected an even "
            "number (review row + detail row per horse) - page structure may have changed"
        )

    records = []
    for review_tr, detail_tr in zip(data_rows[0::2], data_rows[1::2]):
        review_tds = review_tr.find_all("td", recursive=False)
        if len(review_tds) != 4:
            raise ValueError(
                f"race_id={race_id}: expected 4 <td> in OikiriTable review row, got {len(review_tds)} - "
                "page structure may have changed"
            )
        waku_td, umaban_td, horse_td, review_td = review_tds
        horse_a = horse_td.find("a", href=True)

        detail_tds = detail_tr.find_all("td", recursive=False)
        if len(detail_tds) < 9:
            raise ValueError(
                f"race_id={race_id}: expected at least 9 <td> in OikiriTable detail row, got "
                f"{len(detail_tds)} - page structure may have changed"
            )
        # Some race classes add a trailing 映像 (video link) column after
        # rank (10 <td> instead of 9) - only the first 9 are meaningful here.
        date_td, course_td, track_td, rider_td, time_td, pos_td, load_td, critic_td, rank_td = detail_tds[:9]

        times = [li.get_text(strip=True) for li in time_td.select("ul.TrainingTimeDataList li")]
        partner_comment = time_td.find("div", class_="Comment_Cell")

        records.append(
            {
                "umaban": umaban_td.get_text(strip=True),
                "training_review": review_td.get_text(" ", strip=True),
                "training_date": date_td.get_text(strip=True),
                "training_course": course_td.get_text(strip=True),
                "training_track_condition": track_td.get_text(strip=True),
                "training_rider": rider_td.get_text(strip=True),
                "training_times": ";".join(t for t in times if t and t != "-"),
                "training_partner_comment": partner_comment.get_text(" ", strip=True) if partner_comment else "",
                "training_position": pos_td.get_text(strip=True),
                "training_load": load_td.get_text(strip=True),
                "training_critic": critic_td.get_text(strip=True),
                "training_rank": rank_td.get_text(strip=True),
                "_waku_check": waku_td.get_text(strip=True),
                "_horse_id_check": _horse_id(horse_a),
                "_horse_name_check": horse_a.get_text(strip=True) if horse_a else "",
            }
        )

    return pd.DataFrame.from_records(records)


def _parse_oikiri_flat_format(data_rows: list, race_id: str) -> pd.DataFrame:
    """One flat <tr> per horse (the "Stable_Time" layout used for most
    non-featured races): 枠/馬番/馬名/日付/コース/馬場/乗り役/調教タイム/位置/
    負荷/評価/ランク, with no written review or training-partner comment."""
    records = []
    for tr in data_rows:
        tds = tr.find_all("td", recursive=False)
        if len(tds) != 12:
            raise ValueError(
                f"race_id={race_id}: expected 12 <td> in flat OikiriTable row, got {len(tds)} - "
                "page structure may have changed"
            )
        (
            waku_td,
            umaban_td,
            horse_td,
            date_td,
            course_td,
            track_td,
            rider_td,
            time_td,
            pos_td,
            load_td,
            critic_td,
            rank_td,
        ) = tds
        horse_a = horse_td.find("a", href=True)
        times = [li.get_text(strip=True) for li in time_td.select("ul.TrainingTimeDataList li")]

        records.append(
            {
                "umaban": umaban_td.get_text(strip=True),
                "training_review": "",
                "training_date": date_td.get_text(strip=True),
                "training_course": course_td.get_text(strip=True),
                "training_track_condition": track_td.get_text(strip=True),
                "training_rider": rider_td.get_text(strip=True),
                "training_times": ";".join(t for t in times if t and t != "-"),
                "training_partner_comment": "",
                "training_position": pos_td.get_text(strip=True),
                "training_load": load_td.get_text(strip=True),
                "training_critic": critic_td.get_text(strip=True),
                "training_rank": rank_td.get_text(strip=True),
                "_waku_check": waku_td.get_text(strip=True),
                "_horse_id_check": _horse_id(horse_a),
                "_horse_name_check": horse_a.get_text(strip=True) if horse_a else "",
            }
        )

    return pd.DataFrame.from_records(records)


def parse_oikiri(html: str, race_id: str) -> pd.DataFrame:
    """Parses table.OikiriTable (調教タイム). netkeiba renders it in two
    different layouts depending on whether the race got the full featured
    write-up (see _parse_oikiri_review_format) or not (see
    _parse_oikiri_flat_format) - both are detected from the same table, and
    a race with no training data published at all returns an empty frame
    rather than erroring."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.OikiriTable")
    if table is None:
        return pd.DataFrame(columns=_OIKIRI_COLUMNS)

    data_rows = [r for r in table.find_all("tr") if r.find("td")]
    if not data_rows:
        return pd.DataFrame(columns=_OIKIRI_COLUMNS)

    first_td_count = len(data_rows[0].find_all("td", recursive=False))
    if first_td_count == 4:
        return _parse_oikiri_review_format(data_rows, race_id)
    if first_td_count == 12:
        return _parse_oikiri_flat_format(data_rows, race_id)
    raise ValueError(
        f"race_id={race_id}: OikiriTable's first data row has {first_td_count} <td>, expected 4 "
        "(review format) or 12 (flat format) - page structure may have changed"
    )


def parse_newspaper(html: str, race_id: str, require_writeup: bool = True) -> pd.DataFrame:
    """One row per horse, merging 厩舎コメント and 調教タイム. Excludes the
    predicted-lap (前半3F/後半3F) table: it is gated behind a subscription
    tier beyond regular premium and would otherwise silently produce a
    partial column (real value for horse 1, blank for the rest).

    require_writeup=True (default, JRA): raises if both source tables are
    absent, since for JRA that combination has only ever been observed when
    netkeiba's own page structure changed. require_writeup=False (NAR):
    confirmed against real races - including a named/featured one - that NAR
    simply does not publish 厩舎コメント/調教タイム content at all, so both
    being empty is the normal case there, not a scrape failure."""
    comments = parse_stable_comments(html, race_id)
    oikiri = parse_oikiri(html, race_id)

    if comments.empty and oikiri.empty and require_writeup:
        raise ValueError(
            f"race_id={race_id}: neither Stable_Comment nor OikiriTable present - page structure may have changed"
        )

    merged = comments.merge(oikiri, on="umaban", how="outer", validate="one_to_one")
    # When Stable_Comment is absent (no stable-comment race today), waku/
    # horse_id/horse_name only came from OikiriTable's own cells - fall back
    # to those instead of leaving the identifying columns blank.
    merged["waku"] = merged["waku"].fillna(merged["_waku_check"])
    merged["horse_id"] = merged["horse_id"].fillna(merged["_horse_id_check"])
    merged["horse_name"] = merged["horse_name"].fillna(merged["_horse_name_check"])
    merged = merged.drop(columns=["_waku_check", "_horse_id_check", "_horse_name_check"])
    merged.insert(0, "race_id", race_id)
    merged["umaban"] = pd.to_numeric(merged["umaban"], errors="coerce")
    merged = merged.sort_values("umaban").reset_index(drop=True)
    return merged
