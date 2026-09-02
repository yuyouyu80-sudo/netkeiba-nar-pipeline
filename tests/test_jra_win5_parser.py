"""予想ファクター充足度マップ Tier4項目11(WIN5キャリーオーバー)のパーサーテスト。
フィクスチャは2026-09-02、www.jra.go.jp/kouza/win5/result.htmlの実際のページ。"""

from pathlib import Path

from src.netkeiba_pipeline.parsers.jra_win5_parser import parse_win5_carryover_history

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_win5_carryover_history_extracts_all_entries():
    html = (FIXTURES / "jra_win5_result_2026.html").read_text(encoding="utf-8")
    df = parse_win5_carryover_history(html)
    assert len(df) == 16
    assert df.iloc[0]["date"] == "2026-02-01"
    assert df.iloc[0]["carryover_amount_text"] == "5億3,990万5,240円"
    assert df.iloc[-1]["date"] == "2011-06-26"
