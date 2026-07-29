import pytest

from src.netkeiba_pipeline.discovery.tracks import NAR_TRACK_CODES, is_nar_race, race_site_domain, race_url

JRA_CODES = {f"{i:02d}" for i in range(1, 11)}


def test_nar_track_codes_never_collide_with_jra():
    assert not (set(NAR_TRACK_CODES) & JRA_CODES)


def test_is_nar_race_true_for_nar_venue_code():
    assert is_nar_race("202630070101") is True  # 30 = 門別


def test_is_nar_race_false_for_jra_venue_code():
    assert is_nar_race("202406010101") is False  # 06 = 中山


def test_is_nar_race_raises_on_malformed_race_id():
    with pytest.raises(ValueError):
        is_nar_race("not-a-race-id")
    with pytest.raises(ValueError):
        is_nar_race("12345")


def test_race_site_domain_switches_on_circuit():
    assert race_site_domain("202630070101") == "nar.netkeiba.com"
    assert race_site_domain("202406010101") == "race.netkeiba.com"


def test_race_url_builds_domain_and_query():
    url = race_url("202630070101", "newspaper.html", rf="shutuba_submenu")
    assert url == "https://nar.netkeiba.com/race/newspaper.html?race_id=202630070101&rf=shutuba_submenu"

    url = race_url("202406010101", "bias.html")
    assert url == "https://race.netkeiba.com/race/bias.html?race_id=202406010101"
