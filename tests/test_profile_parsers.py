"""予想ファクター充足度マップ Tier1(生産牧場・馬主/騎手・調教師プロフィール)のパーサーテスト。
フィクスチャは2026-09-02、実際のdb.netkeiba.comページから取得(horse_id=2018101615、
jockey_id=01087(JRA)/05688(NAR)、trainer_id=01114(JRA))。"""

from pathlib import Path

from src.netkeiba_pipeline.parsers.horse_profile_parser import parse_horse_profile
from src.netkeiba_pipeline.parsers.person_profile_parser import parse_person_profile

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_horse_profile_extracts_breeder_and_owner():
    html = (FIXTURES / "horse_profile_2018101615.html").read_text(encoding="utf-8")
    df = parse_horse_profile(html, "2018101615")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["owner"] == "ウイン"
    assert row["owner_id"] == "494800"
    assert row["breeder"] == "コスモヴューファーム"
    assert row["breeder_id"] == "214514"


def test_parse_person_profile_jockey_jra():
    html = (FIXTURES / "jockey_profile_01087_jra.html").read_text(encoding="utf-8")
    df = parse_person_profile(html, "01087", "jockey")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["name"] == "上野翔"
    assert row["name_kana"] == "ウエノショウ"
    assert row["birth_date"] == "1985/12/23"
    # bracketed "[美浦]フリー" form -> area/type split apart
    assert row["affiliation_area"] == "美浦"
    assert row["affiliation_type"] == "フリー"
    assert row["jra_season"] == "2026"
    assert row["jra_season_wins"] == "9"
    assert row["jra_career_wins"] == "107"
    # jockey_id=01087 is JRA-affiliated: the NAR box exists but has near-zero activity
    assert row["nar_season"] == ""


def test_parse_person_profile_jockey_nar():
    """NAR-licensed jockeys show an unbracketed area ("地方", no affiliation_type) and
    their real stats sit in the nar_* columns instead of jra_*."""
    html = (FIXTURES / "jockey_profile_05688_nar.html").read_text(encoding="utf-8")
    df = parse_person_profile(html, "05688", "jockey")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["name"] == "阿部基嗣"
    assert row["affiliation_area"] == "地方"
    assert row["affiliation_type"] == ""
    assert row["jra_season"] == ""  # no JRA activity this year
    assert row["nar_season"] == "2026"
    assert row["nar_season_wins"] == "18"


def test_parse_person_profile_trainer():
    """Trainers show an unbracketed area (like NAR jockeys) with no affiliation_type,
    and their column header says 出走回数 instead of 騎乗回数 - same cell position."""
    html = (FIXTURES / "trainer_profile_01114_jra.html").read_text(encoding="utf-8")
    df = parse_person_profile(html, "01114", "trainer")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["name"] == "和田正一郎"
    assert row["affiliation_area"] == "美浦"
    assert row["affiliation_type"] == ""
    assert row["jra_season"] == "2026"
    assert row["jra_season_starts"] == "143"
    assert row["jra_career_starts"] == "3,870"
