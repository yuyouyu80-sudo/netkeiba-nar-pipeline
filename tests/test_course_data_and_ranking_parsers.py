from pathlib import Path

from src.netkeiba_pipeline.parsers.course_analysis_parser import parse_horse_stat_table
from src.netkeiba_pipeline.parsers.ranking_parser import parse_ranking

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_coursedata_sire():
    html = (FIXTURES / "coursedata_202603020802_cid1_sire.html").read_text(encoding="utf-8")
    df = parse_horse_stat_table(html, race_id="202603020802", category_type="sire", source="coursedata cid=1")

    assert len(df) == 16
    assert set(df["category_type"]) == {"sire"}
    assert df["category_label"].iloc[0] == "アニマルキングダム"
    assert df["horse_name"].iloc[0] == "ジェイエルグリーン"


def test_parse_coursedata_broodmare_sire():
    html = (FIXTURES / "coursedata_202603020802_cid4_broodmaresire.html").read_text(encoding="utf-8")
    df = parse_horse_stat_table(html, race_id="202603020802", category_type="broodmare_sire", source="coursedata cid=4")

    assert len(df) == 16
    assert set(df["category_type"]) == {"broodmare_sire"}


def test_parse_surf_summary_speed_index():
    html = (FIXTURES / "surf_summary_202603020802.html").read_text(encoding="utf-8")
    df = parse_horse_stat_table(html, race_id="202603020802", category_type="speed_index", source="surf_summary")

    assert len(df) == 16
    assert set(df["category_type"]) == {"speed_index"}
    # First horse in this fixture has no prior speed-index data ("-" placeholder), not an error.
    assert df["category_label"].iloc[0] == "-"
    assert df["category_label"].iloc[1] == "指数偏差値60"


def test_parse_ranking():
    html = (FIXTURES / "ranking_202603020802.html").read_text(encoding="utf-8")
    df = parse_ranking(html, race_id="202603020802")

    assert set(df["ranking_type"]) == {"jockey", "sire", "trainer"}
    jockeys = df[df["ranking_type"] == "jockey"]
    top = jockeys[jockeys["rank"] == "1"].iloc[0]
    assert top["name"] == "戸崎圭太"
    assert top["win_1st"] == "5"
