"""予想ファクター充足度マップ Tier2(複勝・馬連オッズ)のパーサーテスト。フィクスチャは
2026-09-02、race.netkeiba.com/api/api_get_jra_odds.htmlの実応答(race_id=202604030301、
12頭、type=1(単勝+複勝)/type=4(馬連))。"""

import json
from pathlib import Path

from src.netkeiba_pipeline.parsers.odds_api_parser import parse_fuku_odds, parse_umaren_odds

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_parse_fuku_odds_extracts_all_horses():
    payload = _load("odds_api_tanfuku_202604030301.json")
    df = parse_fuku_odds(payload, "202604030301")
    assert len(df) == 12
    row = df[df["umaban"] == "1"].iloc[0]
    assert row["fuku_odds_low"] == "1.7"
    assert row["fuku_odds_high"] == "2.7"
    assert row["fuku_ninki"] == "3"


def test_parse_umaren_odds_extracts_all_combinations():
    payload = _load("odds_api_umaren_202604030301.json")
    df = parse_umaren_odds(payload, "202604030301")
    # 12頭 -> C(12,2) = 66通り
    assert len(df) == 66
    row = df[(df["umaban_a"] == "1") & (df["umaban_b"] == "2")].iloc[0]
    assert row["umaren_odds"] == "33.7"
    assert row["umaren_ninki"] == "14"


def test_parse_fuku_odds_empty_on_bad_status():
    df = parse_fuku_odds({"status": "NG"}, "202604030301")
    assert df.empty
    assert list(df.columns) == ["race_id", "umaban", "fuku_odds_low", "fuku_odds_high", "fuku_ninki"]


def test_parse_umaren_odds_empty_on_bad_status():
    df = parse_umaren_odds({"status": "NG"}, "202604030301")
    assert df.empty
    assert list(df.columns) == ["race_id", "umaban_a", "umaban_b", "umaren_odds", "umaren_ninki"]
