from pathlib import Path

import pytest

from src.netkeiba_pipeline.parsers.corner_position_parser import parse_corner3_position, parse_corner4_position

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_parse_corner4_position_jra_basic_shape_and_leader():
    html = _load("corner_position_202607030203_jra.html")
    df = parse_corner4_position(html, race_id="202607030203").set_index("umaban")

    assert len(df) == 9
    # 先頭(corner4_rank=1, gap=0)はエイダ(umaban=3)。馬アイコンのスプライトが基準状態で
    # 左向きに走っており、反時計回り(AntiClockwise)コースはCSSのtransformで見た目だけ
    # 左右反転するため、生データのleft:0%は常に先頭を表す(実データで確認済み)。
    assert df.loc["3", "corner4_rank"] == 1
    assert df.loc["3", "corner4_gap_pct"] == 0.0
    assert df.loc["3", "horse_name"] == "エイダ"
    assert df.loc["3", "corner4_speedup"] == 3

    # 最後方はヴィス(umaban=8)
    assert df.loc["8", "corner4_rank"] == 9
    assert df.loc["8", "corner4_gap_pct"] == 100.0


def test_parse_corner4_position_gap_lengths_conversion():
    html = _load("corner_position_202607030203_jra.html")
    df = parse_corner4_position(html, race_id="202607030203").set_index("umaban")

    # コトテ(umaban=5): gap_pct=11.64 -> 馬身換算(CORNER4_LENGTH_PCT_PER_HORSE=10.0586で除算)
    assert df.loc["5", "corner4_gap_lengths"] == pytest.approx(1.16, abs=0.01)
    # 無印(SpeedUpクラス無し)はspeedup=0
    assert df.loc["5", "corner4_speedup"] == 0


def test_parse_corner4_position_nar_skips_stale_commented_out_line():
    """netkeibaが取消馬反映前の古い座標を`//`行コメントとして残したまま、直後に有効な
    (コメントされていない)最新版を置いているケース(NAR実データで確認済み)。コメント行を
    誤って拾うと頭数が一致せずValueErrorになるはずが、正しくスキップして9頭ぶんだけ返す。"""
    html = _load("newspaper_202654072501_nar_no_writeup.html")
    df = parse_corner4_position(html, race_id="202654072501")

    assert len(df) == 9
    assert set(df["umaban"]) == {str(i) for i in range(1, 10)}


def test_parse_corner3_position_jra_basic_shape_and_leader():
    """#CornerSwitchの実際のタブ表記は"3コーナー"だが内部case識別子は'Corner02'
    (モジュールdocstring参照)。3コーナー時点の先頭はコトテ(umaban=5)、4コーナーの先頭
    エイダ(umaban=3、corner4_rank=1)とは異なる馬(実データで確認済み)。"""
    html = _load("corner_position_202607030203_jra.html")
    df = parse_corner3_position(html, race_id="202607030203").set_index("umaban")

    assert len(df) == 9
    assert "corner3_speedup" not in df.columns  # 加速マークは4コーナー時点のみ描画される仕様

    assert df.loc["5", "corner3_rank"] == 1
    assert df.loc["5", "corner3_gap_pct"] == 0.0
    assert df.loc["5", "horse_name"] == "コトテ"

    # 最後方はヴィス(umaban=8)、4コーナーと同じく最後方のまま
    assert df.loc["8", "corner3_rank"] == 9
    assert df.loc["8", "corner3_gap_pct"] == 100.0


def test_parse_corner3_position_gap_lengths_conversion():
    html = _load("corner_position_202607030203_jra.html")
    df = parse_corner3_position(html, race_id="202607030203").set_index("umaban")

    # エイダ(umaban=3): gap_pct=28.57 -> 馬身換算(CORNER4_LENGTH_PCT_PER_HORSE=10.0586で除算)
    assert df.loc["3", "corner3_gap_pct"] == pytest.approx(28.57, abs=0.01)
    assert df.loc["3", "corner3_gap_lengths"] == pytest.approx(2.84, abs=0.01)


def test_parse_corner3_position_nar_skips_stale_commented_out_line():
    html = _load("newspaper_202654072501_nar_no_writeup.html")
    df = parse_corner3_position(html, race_id="202654072501")

    assert len(df) == 9
    assert set(df["umaban"]) == {str(i) for i in range(1, 10)}


def test_parse_corner3_position_empty_page_returns_empty_frame():
    df = parse_corner3_position("<html><body>no develop widget here</body></html>", race_id="000000000000")
    assert df.empty
    assert "umaban" in df.columns


def test_parse_corner4_position_empty_page_returns_empty_frame():
    df = parse_corner4_position("<html><body>no develop widget here</body></html>", race_id="000000000000")
    assert df.empty
    assert "umaban" in df.columns


def test_parse_corner4_position_raises_on_horse_count_mismatch():
    html = """
    <html><body>
    <div class="DevelopImgWrap">
      <a href="https://db.netkeiba.com/horse/2024000001"><span class="HorseIcon" id="Horse1">
        <span class="Waku Waku1">1</span><span class="HorseName">テストA</span>
      </span></a>
      <a href="https://db.netkeiba.com/horse/2024000002"><span class="HorseIcon" id="Horse2">
        <span class="Waku Waku2">2</span><span class="HorseName">テストB</span>
      </span></a>
    </div>
    <script>
    function updateHorsePosition() {
      var checkbox1Checked = false;
      var checkbox2Checked = false;
      if (!checkbox1Checked && !checkbox2Checked) {
        switch (cornerCheck) {
          case 'Corner03':
          $("#Horse1").css({ 'top':'-4%', 'left':'0%', }).append('<span class="SpeedUp_01"></span>');
          break;
        }
      } else if (checkbox1Checked && !checkbox2Checked) {
        // dummy
      }
    }
    </script>
    </body></html>
    """
    with pytest.raises(ValueError):
        parse_corner4_position(html, race_id="000000000000")
