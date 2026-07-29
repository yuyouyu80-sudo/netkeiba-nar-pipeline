from pathlib import Path

import pytest

from src.netkeiba_pipeline.parsers.course_analysis_parser import parse_course_analysis

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_course_analysis_full_field():
    html = (FIXTURES / "course_analysis_202603020802_cid0_authed.html").read_text(encoding="utf-8")
    df = parse_course_analysis(html, race_id="202603020802", cid=0)

    assert len(df) == 16  # full 16-horse field, confirmed via authenticated premium session
    assert set(df["category_type"]) == {"waku"}
    assert df["category_label"].iloc[0] == "1枠"
    assert df["horse_name"].iloc[0] == "ジェイエルグリーン"
    assert df["horse_id"].iloc[0] == "2023106314"


def test_parse_course_analysis_raises_on_paywall():
    html = (FIXTURES / "course_analysis_202603020802_cid0_paywalled.html").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="Premium_Regist_Box"):
        parse_course_analysis(html, race_id="202603020802", cid=0)
