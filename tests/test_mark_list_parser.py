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
    assert df.loc["1", "mark_raw_本紙"] == "△"
    assert df.loc["10", "mark_raw_本紙"] == "☆"

    # CP予想: umaban9(◎)、umaban5(△)
    assert df.loc["9", "mark_raw_CP予想"] == "◎"
    assert df.loc["5", "mark_raw_CP予想"] == "△"
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
    """実データ(202610020801)から手計算で検証したケース(2026-08-27、Icon_Osae/Icon_Kurosanの
    記号対応修正後の値):
    - umaban9: 小林/藤村/大石川が全員◎ → 18点、単独1位 → ◎
    - umaban8: 4/2/4点=10点、単独2位 → ○
    - umaban10: 0.5/4/3点=7.5点、単独3位 → ▲
    - umaban3: 藤村▲(3)+大石川△(2)=5点、単独4位 → △
    - umaban1,5,6: いずれも4点でタイ、5位タイ(上位4段階に入らない) → 無印
    - umaban7: 小林/藤村/大石川いずれも無印 → 0点だが、CP予想が▲(無印でない)を付けている
      ため★にはならず無印(2026-08-26のユーザー指示: ★は0点かつ本紙・CP予想も両方無印の
      場合のみ)
    """
    html = _load("mark_list_202610020801_jra.html")
    raw = parse_mark_list(html, race_id="202610020801")
    summary = summarize_marks(raw).set_index("umaban")

    assert summary.loc["9", "mark_other"] == "◎"
    assert summary.loc["8", "mark_other"] == "○"
    assert summary.loc["10", "mark_other"] == "▲"
    assert summary.loc["3", "mark_other"] == "△"
    assert summary.loc["1", "mark_other"] == ""
    assert summary.loc["5", "mark_other"] == ""
    assert summary.loc["6", "mark_other"] == ""
    assert summary.loc["7", "mark_other"] == ""
    assert summary.loc["7", "mark_cp"] == "▲"

    assert summary.loc["9", "mark_honshi"] == "◎"
    assert summary.loc["9", "mark_cp"] == "◎"


def test_summarize_marks_zero_score_and_honshi_cp_both_blank_gets_star():
    """0点(その他の専門家が1人もいない、または誰も印を付けていない)かつ本紙・CP予想も
    両方無印の場合だけ★になる。"""
    raw = pd.DataFrame(
        {
            "umaban": ["1", "2"],
            "mark_raw_本紙": ["", ""],
            "mark_raw_CP予想": ["", ""],
        }
    )
    summary = summarize_marks(raw).set_index("umaban")
    assert summary.loc["1", "mark_other"] == "★"
    assert summary.loc["2", "mark_other"] == "★"


def test_summarize_marks_zero_score_but_honshi_or_cp_marked_no_star():
    """その他の専門家からのスコアが0点でも、本紙かCP予想のどちらかに印が付いていれば
    ★にはせず無印(空文字)にする(2026-08-26のユーザー指示による仕様変更)。"""
    raw = pd.DataFrame(
        {
            "umaban": ["1", "2", "3"],
            "mark_raw_本紙": ["◎", "", ""],
            "mark_raw_CP予想": ["", "○", ""],
        }
    )
    summary = summarize_marks(raw).set_index("umaban")
    # umaban1: 本紙に印あり(CP予想は無印) -> ★にならない
    assert summary.loc["1", "mark_other"] == ""
    # umaban2: CP予想に印あり(本紙は無印) -> ★にならない
    assert summary.loc["2", "mark_other"] == ""
    # umaban3: 両方無印 -> ★
    assert summary.loc["3", "mark_other"] == "★"


def test_summarize_marks_empty_input():
    summary = summarize_marks(pd.DataFrame(columns=["umaban"]))
    assert summary.empty
    assert list(summary.columns) == ["umaban", "mark_honshi", "mark_cp", "mark_other"]
