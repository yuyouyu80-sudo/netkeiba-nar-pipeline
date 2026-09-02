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


def test_parse_bias_extracts_jockey_trainer_ids_and_allowance_mark_for_nar():
    """NAR jockey/trainer IDs are alphanumeric (e.g. "a03e2"), not just digits -
    this fixture has a mix of both forms plus exactly one weight-allowance mark
    (umaban=5, 阿部基嗣, "△")."""
    html = (FIXTURES / "bias_202654072501_nar.html").read_text(encoding="utf-8")
    df = parse_bias(html, race_id="202654072501")
    assert set(df["bias_jockey_id"]) == {
        "05564", "a03e2", "a0270", "a01ad", "05688", "a01bb", "05496", "05536", "05074",
    }
    assert set(df["bias_trainer_id"]) == {
        "a020e", "05785", "a0549", "a003f", "a030b", "05663", "05580",
    }
    marked = df[df["bias_jockey_allowance_mark"] != ""]
    assert list(marked["umaban"]) == ["5"]
    assert marked.iloc[0]["bias_jockey_allowance_mark"] == "△"
    assert marked.iloc[0]["bias_jockey"] == "阿部基嗣"  # mark must not leak into bias_jockey


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


def test_parse_data_breakdown_distance_terminal_label_accepts_variable_row_count():
    """2026-08-13判明: NAR mode=distanceの行数は固定5ではなく、そのレースについて
    netkeiba側に近傍距離の比較データがどれだけあるかで1-4本の距離別行+固定の
    「全成績」行という可変長構造(笠松/門別の一部レースで2行のみのケースを実データで
    確認済み)。terminal_label="全成績"を渡すと、行数の完全一致ではなく最終行の
    ラベル一致で受理される。"""
    html = (FIXTURES / "databreak_202635072701_distance_nar.html").read_text(encoding="utf-8")
    df = parse_data_breakdown(
        html, race_id="202635072701", prefix="data_distance", num_slots=5, terminal_label="全成績"
    )
    assert len(df) > 0


def test_parse_data_breakdown_distance_terminal_label_rejects_wrong_last_row():
    """最終行が「全成績」以外なら、たとえ行数が上限内でも構造異常として引き続き
    raiseする(可変長を許容しても野放図にはしない)。"""
    html = (FIXTURES / "databreak_202635072701_distance_nar.html").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="expected the last of"):
        parse_data_breakdown(
            html, race_id="202635072701", prefix="data_distance", num_slots=5, terminal_label="存在しないラベル"
        )


def test_parse_data_breakdown_distance_terminal_label_handles_real_2slot_race():
    """2026-08-12/13に実際に取得失敗していた笠松081206(202647081206)の実データ。
    近傍距離データがほとんど無い馬ばかりのレースでは、距離別行が「ダートm」
    (具体的な距離すら入らない1行)+「全成績」の計2行しか無い。"""
    html = (FIXTURES / "databreak_202647081206_distance_nar_2slots.html").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="expected 5 category rows"):
        parse_data_breakdown(html, race_id="202647081206", prefix="data_distance", num_slots=5)

    df = parse_data_breakdown(
        html, race_id="202647081206", prefix="data_distance", num_slots=5, terminal_label="全成績"
    )
    assert len(df) > 0
    assert "data_distance_slot3_label" not in df.columns


def test_parse_data_breakdown_distance_terminal_label_rejects_over_upper_bound():
    """num_slotsは可変長モードでも上限のガードとして機能する(想定より大幅に
    多い行数が来た場合は構造変化としてraiseする)。"""
    html = (FIXTURES / "databreak_202635072701_distance_nar.html").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match=r"expected 1-2 category rows"):
        parse_data_breakdown(
            html, race_id="202635072701", prefix="data_distance", num_slots=2, terminal_label="全成績"
        )


def test_parse_newspaper_tolerates_missing_writeup_for_nar():
    """Confirmed against multiple real NAR races (including a named/featured
    one) that 厩舎コメント/調教タイム content simply isn't published for NAR -
    require_writeup=False must not raise, unlike the JRA default."""
    html = (FIXTURES / "newspaper_202654072501_nar_no_writeup.html").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="neither Stable_Comment nor OikiriTable"):
        parse_newspaper(html, race_id="202654072501")  # require_writeup=True default

    df = parse_newspaper(html, race_id="202654072501", require_writeup=False)
    assert df.empty
