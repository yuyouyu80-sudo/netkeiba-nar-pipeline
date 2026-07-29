"""Confirms the JRA parsers work against real NAR (regional racing) pages once
pointed at nar.netkeiba.com, and that the small number of genuine NAR-specific
behaviors (missing 厩舎コメント/調教タイム writeup, missing surf_summary
speed_index, different data-breakdown row counts) are handled correctly.
Fixtures were captured from real races: 202654072501 (高知 2026-07-25 R1),
202635072701 (盛岡 2026-07-27 R1)."""

from pathlib import Path

import pytest

from src.netkeiba_pipeline.parsers.bias_parser import parse_bias
from src.netkeiba_pipeline.parsers.course_analysis_parser import parse_course_analysis, parse_horse_stat_table
from src.netkeiba_pipeline.parsers.newspaper_parser import parse_newspaper
from src.netkeiba_pipeline.parsers.race_data_parser import parse_data_breakdown, parse_horse_category_table
from src.netkeiba_pipeline.parsers.ranking_parser import parse_ranking
from src.netkeiba_pipeline.parsers.shutuba_past_parser import parse_shutuba_past
from src.netkeiba_pipeline.parsers.speed_index_parser import parse_speed_index

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_course_analysis_works_unmodified_for_nar():
    html = (FIXTURES / "course_analysis_202654072501_cid0_nar.html").read_text(encoding="utf-8")
    df = parse_course_analysis(html, race_id="202654072501", cid=0)
    assert len(df) == 9


def test_parse_horse_stat_table_coursedata_works_unmodified_for_nar():
    html = (FIXTURES / "coursedata_202654072501_cid1_sire_nar.html").read_text(encoding="utf-8")
    df = parse_horse_stat_table(html, race_id="202654072501", category_type="sire", source="coursedata cid=1")
    assert len(df) == 9


def test_parse_horse_stat_table_surf_summary_raises_when_genuinely_absent_for_nar():
    """Confirmed genuinely absent (no table#table_sort_back at all) on every NAR
    race checked - the callers (fetch_newspaper.py / fetch_course_analysis.py)
    catch this specific ValueError and skip rather than fail the whole race."""
    html = (FIXTURES / "surf_summary_202654072501_default_nar_absent.html").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="table_sort_back"):
        parse_horse_stat_table(html, race_id="202654072501", category_type="speed_index", source="surf_summary")


def test_parse_ranking_works_for_nar_within_publish_window():
    """リーディング data is only published race-eve through 2 days after (netkeiba's
    own stated window) - this fixture was captured for a same-day race to stay
    safely inside it. A race fetched right at/after the window's edge legitimately
    returns no data; that's expected, not a parser bug."""
    html = (FIXTURES / "ranking_202635072701_nar.html").read_text(encoding="utf-8")
    df = parse_ranking(html, race_id="202635072701")
    assert len(df) > 0
    assert set(df["ranking_type"]) <= {"jockey", "sire", "trainer"}


def test_parse_horse_category_table_concerned_works_unmodified_for_nar():
    html = (FIXTURES / "concerned_202654072501_cid0_nar.html").read_text(encoding="utf-8")
    df = parse_horse_category_table(html, race_id="202654072501", category_type="concerned", source="concerned cid=0")
    assert len(df) == 9


def test_parse_bias_works_unmodified_for_nar():
    html = (FIXTURES / "bias_202654072501_nar.html").read_text(encoding="utf-8")
    df = parse_bias(html, race_id="202654072501")
    assert len(df) == 9
    assert df["bias_horse_id"].notna().all()


def test_parse_shutuba_past_works_unmodified_for_nar():
    html = (FIXTURES / "shutuba_past_202654072501_nar.html").read_text(encoding="utf-8")
    df = parse_shutuba_past(html, race_id="202654072501")
    assert len(df) == 9


def test_parse_speed_index_works_unmodified_for_nar():
    html = (FIXTURES / "speed_202654072501_nar.html").read_text(encoding="utf-8")
    df = parse_speed_index(html, race_id="202654072501")
    assert len(df) == 9


def test_parse_data_breakdown_distance_needs_nar_specific_slot_count():
    """NAR's "distance" breakdown shows one extra nearby-distance row versus
    JRA (5 category rows per horse, not 4) - confirmed by inspecting the actual
    labels (e.g. ダート1000m/850m/1200m/1300m + 全成績)."""
    html = (FIXTURES / "databreak_202635072701_distance_nar.html").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="expected 4 category rows"):
        parse_data_breakdown(html, race_id="202635072701", prefix="data_distance", num_slots=4)

    df = parse_data_breakdown(html, race_id="202635072701", prefix="data_distance", num_slots=5)
    assert len(df) > 0


def test_parse_newspaper_tolerates_missing_writeup_for_nar():
    """Confirmed against multiple real NAR races (including a named/featured
    one) that 厩舎コメント/調教タイム content simply isn't published for NAR -
    require_writeup=False must not raise, unlike the JRA default."""
    html = (FIXTURES / "newspaper_202654072501_nar_no_writeup.html").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="neither Stable_Comment nor OikiriTable"):
        parse_newspaper(html, race_id="202654072501")  # require_writeup=True default

    df = parse_newspaper(html, race_id="202654072501", require_writeup=False)
    assert df.empty
