from pathlib import Path

import pandas as pd
import pytest

from src.netkeiba_pipeline.parsers.mark_list_parser import parse_mark_list, summarize_marks

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_parse_mark_list_jra_basic_shape():
    html = _load("mark_list_202610020801_jra.html")
    df = parse_mark_list(html, race_id="202610020801")

    assert len(df) == 11  # 11頭立て
    assert list(df["umaban"]) == [str(i) for i in range(1, 12)]
    # 実専門家5名(本紙/小林/藤村/大石川/CP予想)。「予想ビルダー」広告枠(id="...builder")と
    # 名前無しテンプレートブロックは除外されている。
    expert_cols = [c for c in df.columns if c != "umaban"]
    assert set(expert_cols) == {
        "mark_raw_本紙",
        "mark_raw_小林",
        "mark_raw_藤村",
        "mark_raw_大石川",
        "mark_raw_CP予想",
    }


def test_parse_mark_list_jra_marks_align_to_umaban_by_position():
    """<br>区切りの専門家名連結と、position(表示順)ベースの馬番対応の両方を検証する。"""
    html = _load("mark_list_202610020801_jra.html")
    df = parse_mark_list(html, race_id="202610020801").set_index("umaban")

    # 本紙: umaban9(◎)とumaban7(無印)を実データのHTMLから目視確認済み
    assert df.loc["9", "mark_raw_本紙"] == "◎"
    assert df.loc["7", "mark_raw_本紙"] == ""
    assert df.loc["1", "mark_raw_本紙"] == "▲"
    assert df.loc["10", "mark_raw_本紙"] == "☆"

    # CP予想: umaban9(◎)、umaban5(▲)
    assert df.loc["9", "mark_raw_CP予想"] == "◎"
    assert df.loc["5", "mark_raw_CP予想"] == "▲"
    assert df.loc["1", "mark_raw_CP予想"] == ""


def test_parse_mark_list_empty_page_returns_empty_frame():
    df = parse_mark_list("<html><body>no yosoka here</body></html>", race_id="000000000000")
    assert df.empty
    assert "umaban" in df.columns


def test_parse_mark_list_raises_when_mark_count_mismatches_umaban_count():
    html = """
    <html><body>
    <dl class="Umaban"><dt>枠</dt><dd><ul><li class="Num Waku1">1</li></ul></dd></dl>
    <dl class="Umaban"><dt>馬番</dt><dd><ul>
      <li class="Num">1</li><li class="Num">2</li>
    </ul></dd></dl>
    <dl class="Yosoka" id="yoso_goods_seq_0">
      <dt><p class="yosoka_name">本<br>紙<br></p></dt>
      <dd><ul>
        <li class="Mark_Pro mark_1"><span class="Icon_Shirushi Icon_Honmei"></span></li>
      </ul></dd>
    </dl>
    </body></html>
    """
    with pytest.raises(ValueError):
        parse_mark_list(html, race_id="000000000000")


def test_parse_mark_list_nar_no_honshi_expert():
    """NAR実データ(2026-08-26確認): このレースの専門家に「本紙」は含まれない
    (立川/CP予想/マリア/馬場予想/中山の5名)。mark_honshiが空になるのは正常。"""
    html = _load("mark_list_202655082301_nar.html")
    df = parse_mark_list(html, race_id="202655082301")

    expert_cols = [c for c in df.columns if c != "umaban"]
    assert "mark_raw_本紙" not in expert_cols
    assert "mark_raw_CP予想" in expert_cols


def test_summarize_marks_scoring_and_ranking():
    """実データ(202610020801)から手計算で検証したケース:
    - umaban9: 小林/藤村/大石川が全員◎ → 18点、単独1位 → ◎
    - umaban8: 4/3/4点=11点、単独2位 → ○
    - umaban10: 0.5/4/2点=6.5点、単独3位 → ▲
    - umaban1,6: ともに6点でタイ、4位タイ → 両方△
    - umaban3: 5点、5位(上位4段階に入らない) → 無印
    - umaban7: 小林/藤村/大石川いずれも無印 → 0点 → ★
    """
    html = _load("mark_list_202610020801_jra.html")
    raw = parse_mark_list(html, race_id="202610020801")
    summary = summarize_marks(raw).set_index("umaban")

    assert summary.loc["9", "mark_other"] == "◎"
    assert summary.loc["8", "mark_other"] == "○"
    assert summary.loc["10", "mark_other"] == "▲"
    assert summary.loc["1", "mark_other"] == "△"
    assert summary.loc["6", "mark_other"] == "△"
    assert summary.loc["3", "mark_other"] == ""
    assert summary.loc["7", "mark_other"] == "★"

    assert summary.loc["9", "mark_honshi"] == "◎"
    assert summary.loc["9", "mark_cp"] == "◎"


def test_summarize_marks_no_other_experts_all_stars():
    """本紙・CP予想以外の専門家が1人もいない場合、全馬が★になる。"""
    raw = pd.DataFrame(
        {
            "umaban": ["1", "2"],
            "mark_raw_本紙": ["◎", "○"],
            "mark_raw_CP予想": ["○", "◎"],
        }
    )
    summary = summarize_marks(raw).set_index("umaban")
    assert summary.loc["1", "mark_other"] == "★"
    assert summary.loc["2", "mark_other"] == "★"


def test_summarize_marks_empty_input():
    summary = summarize_marks(pd.DataFrame(columns=["umaban"]))
    assert summary.empty
    assert list(summary.columns) == ["umaban", "mark_honshi", "mark_cp", "mark_other"]
