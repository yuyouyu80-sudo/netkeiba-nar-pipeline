from pathlib import Path

from src.netkeiba_pipeline.parsers.race_result_parser import parse_lap_times, parse_race_result

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_race_result_basic_shape():
    html = (FIXTURES / "race_result_202406010101.html").read_text(encoding="utf-8")
    df = parse_race_result(html, race_id="202406010101")

    assert len(df) == 16  # 16 horses ran in this race
    assert set(df["race_id"]) == {"202406010101"}
    assert df["race_date"].iloc[0] == "2024-01-06"
    assert df["distance_m"].iloc[0] == 1200
    assert df["surface"].iloc[0] == "ダ"


def test_parse_race_result_winner_row():
    html = (FIXTURES / "race_result_202406010101.html").read_text(encoding="utf-8")
    df = parse_race_result(html, race_id="202406010101")

    winner = df[df["finish_pos"] == "1"].iloc[0]
    assert winner["horse_name"] == "サンディブロンド"
    assert winner["horse_id"] == "2021105821"
    assert winner["jockey_name"] == "戸崎圭太"
    assert winner["jockey_id"] == "05386"
    assert winner["time"] == "1:12.6"
    assert winner["odds_final"] == "4.2"
    assert winner["prize"] == "550.0"


def test_parse_race_result_raises_on_missing_table():
    import pytest

    with pytest.raises(ValueError):
        parse_race_result("<html><body>no table here</body></html>", race_id="000000000000")


def test_parse_race_result_nar_alphanumeric_jockey_id():
    """NAR (regional racing)-licensed jockeys carry alphanumeric IDs (e.g. "a05df"),
    unlike JRA's numeric-only IDs. This is the real-world case the ID regex fix
    (\\d+ -> \\w+) targets - captured from a real 門別 (Monbetsu) race."""
    html = (FIXTURES / "race_result_202630070101_nar.html").read_text(encoding="utf-8")
    df = parse_race_result(html, race_id="202630070101")

    assert len(df) == 12
    winner = df[df["finish_pos"] == "1"].iloc[0]
    assert winner["horse_name"] == "リュウノアキレス"
    assert winner["jockey_name"] == "近藤翔月"
    assert winner["jockey_id"] == "a05df"


def test_parse_lap_times_basic():
    html = (FIXTURES / "race_result_202406010101.html").read_text(encoding="utf-8")
    df = parse_lap_times(html, race_id="202406010101")

    assert len(df) == 6  # 1200m race, 200m segments
    assert list(df["segment_index"]) == [1, 2, 3, 4, 5, 6]
    assert df["lap_time_sec"].iloc[0] == 11.8
    assert df["lap_time_sec"].sum() == 72.6


def test_parse_lap_times_missing_table_raises():
    import pytest

    with pytest.raises(ValueError):
        parse_lap_times("<html><body>no table here</body></html>", race_id="000000000000")


def test_parse_lap_times_nar_empty_table_returns_empty_df():
    """Some NAR races publish table.result_table_02[summary="ラップタイム"] with no
    <tr> rows at all (lap data simply wasn't recorded for that race) - this is a
    legitimate absence, not a structural break, so it must not raise."""
    html = (FIXTURES / "race_result_202630070101_nar.html").read_text(encoding="utf-8")
    df = parse_lap_times(html, race_id="202630070101")

    assert len(df) == 0
    assert list(df.columns) == ["race_id", "segment_index", "lap_time_sec"]


def test_parse_lap_times_nar_populated():
    html = (FIXTURES / "race_result_202644072401_nar.html").read_text(encoding="utf-8")
    df = parse_lap_times(html, race_id="202644072401")

    assert len(df) == 9
    assert df["lap_time_sec"].iloc[0] == 12.9
    assert df["lap_time_sec"].iloc[-1] == 13.0


def test_parse_race_result_nar_alphanumeric_owner_id():
    """Same NAR alphanumeric-ID phenomenon as jockey_id, but for owner_id (and
    trainer_id) - captured from a real 大井 (Oi) race where owner_id is a mix of
    numeric and alphanumeric IDs within the same race."""
    html = (FIXTURES / "race_result_202644072401_nar.html").read_text(encoding="utf-8")
    df = parse_race_result(html, race_id="202644072401")

    assert len(df) == 15
    assert any(not oid.isdigit() for oid in df["owner_id"])  # at least one alphanumeric owner_id
    assert any(oid.isdigit() for oid in df["owner_id"])  # ...alongside plain-numeric ones
    winner = df[df["finish_pos"] == "1"].iloc[0]
    assert winner["horse_name"] == "ニシノコヌカアメ"
    assert winner["jockey_id"] == "a06ce"
