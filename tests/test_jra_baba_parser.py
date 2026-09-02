"""予想ファクター充足度マップ Tier3(クッション値・含水率)のPDFパーサーテスト。
フィクスチャは2026-09-02、JRA公式サイト(2026年1回東京競馬、開催回終了済みで
アーカイブ公開済みだったもの)から実際にダウンロードしたPDF。"""

from pathlib import Path

from src.netkeiba_pipeline.parsers.jra_baba_parser import parse_baba_pdf

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_baba_pdf_extracts_all_meet_days():
    pdf_bytes = (FIXTURES / "jra_baba_tokyo01_2026.pdf").read_bytes()
    df = parse_baba_pdf(pdf_bytes, year="2026", venue="東京", kai="1")
    # 2026年1回東京競馬: 第1日〜第8日 + 各回前日(金曜)の計測分 = 13行
    assert len(df) == 13
    assert list(df["date"])[:3] == ["2026-01-30", "2026-01-31", "2026-02-01"]


def test_parse_baba_pdf_values_match_known_row():
    pdf_bytes = (FIXTURES / "jra_baba_tokyo01_2026.pdf").read_bytes()
    df = parse_baba_pdf(pdf_bytes, year="2026", venue="東京", kai="1")
    row = df[df["date"] == "2026-01-31"].iloc[0]
    assert row["day_label"] == "第 1日"
    assert row["weekday"] == "土曜日"
    assert row["turf_course_variant"] == "D"
    assert row["cushion_value"] == "9.6"
    assert row["moisture_turf_goal_pct"] == "14.1"
    assert row["moisture_turf_corner4_pct"] == "12.3"
    assert row["moisture_dirt_goal_pct"] == "1.2"
    assert row["moisture_dirt_corner4_pct"] == "1.7"


def test_parse_baba_pdf_kai_zero_padded():
    pdf_bytes = (FIXTURES / "jra_baba_tokyo01_2026.pdf").read_bytes()
    df = parse_baba_pdf(pdf_bytes, year="2026", venue="東京", kai=1)
    assert (df["kai"] == "01").all()
