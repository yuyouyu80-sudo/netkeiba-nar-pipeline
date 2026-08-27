"""newspaper.htmlの「AI展開」4コーナー位置取り予想(DevelopImg01ウィジェット)のパース。

race.netkeiba.com/race/newspaper.html の一部として、既存のfetch_newspaper_html()(requestsのみ、
JS実行不要)で取得したHTMLからそのまま抽出できる。ウィジェット自体はJSでDOM操作されて見た目が
切り替わるが、全コーナー×全チェック状態の座標値は最初から<script>内にJSコードの文字列として
埋め込まれているため、DOM操作を再現する必要はなく文字列として直接パースする。

実データ確認(2026-08-26、race_id=202607030203)で判明した構造・注意点:
- 各馬のアイコン(初期DOM、"スタート後"時点)は
  `<span class="HorseIcon ..." id="Horse{internal_id}">` ブロック内に
  `<span class="Waku Waku{N}">{umaban}</span>` (馬番)、`<span class="HorseName">{name}</span>`
  を持つ。internal_idはHorse{N}のNで、umaban(馬番)とは別の内部連番(mark_list.htmlのmark_Nと
  同種の仕組み)。
- `#CornerSwitch`タブは実データで3つだけ確認済み: `id="Corner01"`(表示ラベルは"スタート後"、
  1コーナーではない)、`id="Corner02"`(表示ラベルは"3コーナー")、`id="Corner03"`(表示ラベルは
  "4コーナー")。内部のcase識別子(Corner01/02/03)と実際のコーナー名がズレている
  (2026-08-27確認)。1コーナー・2コーナー単独のデータはnetkeiba側に存在しない。
- 各状態の座標(`left:X%`)と加速マーク(SpeedUp_01/02/03、4コーナー状態のみ付与)は、
  `updateHorsePosition()` 関数内、`switch (cornerCheck) { case 'CornerNN': ... }` の中に
  `$("#Horse{N}").css({ 'top':'Y%', 'left':'X%', }).append('<span class="SpeedUp_0M"></span>');`
  という形でJS文字列として並んでいる。この関数は「出遅れ率/騎手傾向」チェックボックスの
  4通りの組み合わせごとに同じswitch文が4回繰り返されるため、**最初の
  `if (!checkbox1Checked && !checkbox2Checked)`(チェックなし=既定表示)ブロックの
  各`case`だけ**を対象にする。
- 先頭判定: 実データとCSS(develop.css)を確認した結果、`left:0%`が常に先頭(1着相当)、
  `left:100%`が常に最後方。馬アイコンのスプライト画像(animation_horse01.png等)は変形前の
  基準状態で左向きに走っており、反時計回り(AntiClockwise)コースはCSSの
  `transform: scale(-1,1)`でウィジェット全体を左右反転して見た目だけを調整している
  (データの意味自体はコースの回り方に関係なく不変)。
- 「その他」の分析軸である縦方向(top%)は、隣接する馬との横方向(left%、進行方向)の距離が
  馬アイコン幅(10.5%)未満のときだけ重ならないよう別の段にずらす表示上の仕組みであり、
  横方向(柵からの距離)の実データは持っていない(同一馬でもコーナー/チェック条件によって
  一貫性なく変化することを確認済み)。そのためこのパーサーはtop%を抽出しない。
- 馬身換算: `left:X%`は`.DevelopImgWrap`(横幅`calc(90% - 30px)`)基準の座標。柵の目安
  (`.DevelopImg01::before`、`background-repeat:repeat-x`のSVGタイル、1タイル幅は
  `.DevelopImg01`本体の高さ6.3%×アスペクト比32/24=8.4%)を「1馬身」とみなすユーザー指定の
  換算方法に従い、実際にPlaywrightでこのページをレンダリングしてピクセル単位で両者の比率を
  実測したところ、1馬身 ≈ horse-left%スケールで10.0586ポイントだった
  (定数CORNER4_LENGTH_PCT_PER_HORSE、導出根拠は上記)。この比率は`.DevelopImgWrap`の
  `calc(90% - 30px)`という固定px項を含む計算式に起因してviewportにより多少変動しうる近似値
  であり、目安の換算である点に注意。
- ウィジェット自体が存在しない(netkeiba側でAI展開データが無い)日は空DataFrameを返す
  (既存パーサー規約と同じ)。
"""
import re

import pandas as pd
from bs4 import BeautifulSoup

HORSE_ID_RE = re.compile(r"/horse/(\d+)")
SPEEDUP_RE = re.compile(r"SpeedUp_0(\d)")

# 実測(2026-08-26、race_id=202607030203、Playwrightでのピクセル実測)による換算定数。
# 1馬身 ≈ この値(horse-left%スケール上のポイント数)。導出根拠はモジュールdocstring参照。
CORNER4_LENGTH_PCT_PER_HORSE = 10.0586

_UPDATE_HORSE_POSITION_FN_RE = re.compile(r"function\s+updateHorsePosition\s*\(\s*\)\s*\{")
_NO_CHECKBOX_BLOCK_RE = re.compile(
    r"if\s*\(\s*!checkbox1Checked\s*&&\s*!checkbox2Checked\s*\)\s*\{(.*?)\}\s*else\s+if\s*\(\s*checkbox1Checked\s*&&\s*!checkbox2Checked\s*\)",
    re.DOTALL,
)
_HORSE_CSS_RE = re.compile(
    r'\$\("#Horse(\d+)"\)\.css\(\{[^}]*?\'left\'\s*:\s*\'([\d.]+)%\'[^}]*?\}\)'
    r'(?:\.append\(\'<span class="([^"]*)"></span>\'\))?'
)

# #CornerSwitch のタブは実データで3つだけ確認済み: Corner01="スタート後", Corner02="3コーナー",
# Corner03="4コーナー"(2026-08-27確認、race_id=202607030203)。1コーナー・2コーナー単独の
# データは存在しない(「スタート後」と「3コーナー」の間の中間地点は取得不可)。
CORNER_CASE_LABEL = {"corner3": "Corner02", "corner4": "Corner03"}


def _case_re(case_label: str) -> re.Pattern:
    return re.compile(r"case\s+'" + re.escape(case_label) + r"'\s*:(.*?)break\s*;", re.DOTALL)


def _base_horse_info(soup: BeautifulSoup) -> dict[str, dict]:
    """初期DOM(スタート後時点)のHorseIconブロックからinternal_id -> {umaban, horse_name, horse_id}
    を作る。"""
    info = {}
    for el in soup.select(".DevelopImgWrap .HorseIcon[id]"):
        internal_id = el.get("id", "")
        waku_span = el.select_one(".Waku")
        name_span = el.select_one(".HorseName")
        parent_a = el.find_parent("a", href=True)
        horse_id_match = HORSE_ID_RE.search(parent_a["href"]) if parent_a else None
        info[internal_id] = {
            "umaban": waku_span.get_text(strip=True) if waku_span else "",
            "horse_name": name_span.get_text(strip=True) if name_span else "",
            "horse_id": horse_id_match.group(1) if horse_id_match else "",
        }
    return info


def _parse_corner_case(html: str, race_id: str, corner_key: str, include_speedup: bool) -> pd.DataFrame:
    """corner_key("corner3" or "corner4")に対応するcase(CORNER_CASE_LABEL参照)の座標を
    1行1頭で返す共通実装。列名は corner_key をプレフィックスにして動的に生成する。"""
    case_label = CORNER_CASE_LABEL[corner_key]
    rank_col = f"{corner_key}_rank"
    pct_col = f"{corner_key}_gap_pct"
    len_col = f"{corner_key}_gap_lengths"
    speedup_col = f"{corner_key}_speedup"
    out_cols = ["umaban", "horse_id", "horse_name", rank_col, pct_col, len_col]
    if include_speedup:
        out_cols.append(speedup_col)

    soup = BeautifulSoup(html, "lxml")
    wrap = soup.select_one(".DevelopImgWrap")
    if wrap is None:
        return pd.DataFrame(columns=["umaban"])

    horse_info = _base_horse_info(soup)
    if not horse_info:
        return pd.DataFrame(columns=["umaban"])

    # updateHorsePosition()自体はupdateCheckRap()と同じ"if (!checkbox1Checked && !checkbox2Checked)"
    # という条件文を持つ別関数なので、まずupdateHorsePosition()の開始位置を特定してから
    # そこ以降だけを検索対象にする(そうしないとupdateCheckRap()側に誤ってマッチする)。
    fn_match = _UPDATE_HORSE_POSITION_FN_RE.search(html)
    if fn_match is None:
        raise ValueError(
            f"race_id={race_id}: updateHorsePosition()関数が見つからない - page structure may have changed"
        )
    fn_body = html[fn_match.end() :]

    no_checkbox_match = _NO_CHECKBOX_BLOCK_RE.search(fn_body)
    if no_checkbox_match is None:
        raise ValueError(
            f"race_id={race_id}: updateHorsePosition()のチェックなし分岐が見つからない - "
            "page structure may have changed"
        )

    case_match = _case_re(case_label).search(no_checkbox_match.group(1))
    if case_match is None:
        raise ValueError(
            f"race_id={race_id}: case '{case_label}'ブロックが見つからない - page structure may have changed"
        )

    # netkeiba側が取消・除外反映前の古い計算結果を`//`行コメントとして残したまま、直後に
    # 有効な(コメントアウトされていない)最新版を置いていることがある(NAR実データで確認済み、
    # 取消馬を含む古い頭数の行がまるごとコメントアウトされていた)。JSが実際に実行するのは
    # コメントされていない方だけなので、行コメントを除去してから抽出する。
    case_text = re.sub(r"//[^\n]*", "", case_match.group(1))

    entries = _HORSE_CSS_RE.findall(case_text)
    if not entries:
        raise ValueError(
            f"race_id={race_id}: case '{case_label}'内に馬の座標が1件も見つからない - "
            "page structure may have changed"
        )
    if len(entries) != len(horse_info):
        raise ValueError(
            f"race_id={race_id}: {case_label}の座標数({len(entries)})と初期DOMの馬数({len(horse_info)})"
            "が一致しない - page structure may have changed"
        )

    records = []
    for internal_id, left_pct_str, speedup_class in entries:
        key = f"Horse{internal_id}"
        if key not in horse_info:
            raise ValueError(
                f"race_id={race_id}: {case_label}に出現するinternal_id={internal_id}が初期DOMに無い - "
                "page structure may have changed"
            )
        info = horse_info[key]
        record = {
            "umaban": info["umaban"],
            "horse_id": info["horse_id"],
            "horse_name": info["horse_name"],
            pct_col: float(left_pct_str),
        }
        if include_speedup:
            speedup_match = SPEEDUP_RE.search(speedup_class or "")
            record[speedup_col] = int(speedup_match.group(1)) if speedup_match else 0
        records.append(record)

    df = pd.DataFrame.from_records(records)

    # 先頭(left%最小)を0とする差分に正規化(通常はleft%そのものが既に0始まりのはずだが、
    # 将来的な仕様変更に備えて最小値基準に明示的に正規化する)。
    min_pct = df[pct_col].min()
    df[pct_col] = df[pct_col] - min_pct
    df[len_col] = (df[pct_col] / CORNER4_LENGTH_PCT_PER_HORSE).round(2)

    # dense rank(同着は同順位)、先頭が1位。
    unique_gaps_asc = sorted(df[pct_col].unique())
    rank_of_gap = {gap: rank + 1 for rank, gap in enumerate(unique_gaps_asc)}
    df[rank_col] = df[pct_col].map(rank_of_gap)

    df[pct_col] = df[pct_col].round(2)

    return df[out_cols]


def parse_corner4_position(html: str, race_id: str) -> pd.DataFrame:
    """4コーナー(ラスト直線に入る手前)の予想位置取りを1行1頭で返す。

    列: umaban, horse_id, horse_name, corner4_rank(1=先頭からのdense rank),
    corner4_gap_pct(先頭からの差、horse-left%スケール、0=先頭),
    corner4_gap_lengths(上記を馬身換算した近似値)、corner4_speedup(加速マーク▶の数、0-3)。

    ウィジェット自体が無い(netkeiba側にAI展開データが無い)日は空DataFrameを返す
    (スクレイピング失敗ではなく実際にデータが無い状態)。
    """
    return _parse_corner_case(html, race_id, "corner4", include_speedup=True)


def parse_corner3_position(html: str, race_id: str) -> pd.DataFrame:
    """3コーナーの予想位置取りを1行1頭で返す(#CornerSwitchの実際のタブ表記は"3コーナー"、
    内部のcase識別子は'Corner02'。1コーナー・2コーナー単独のデータはnetkeiba側に存在しない
    ため取得不可、モジュールdocstring参照)。

    列: umaban, horse_id, horse_name, corner3_rank(1=先頭からのdense rank),
    corner3_gap_pct(先頭からの差、horse-left%スケール、0=先頭)、
    corner3_gap_lengths(上記を馬身換算した近似値)。
    加速マーク(▶)は4コーナー時点のみ描画される仕様のため、3コーナーにはcorner3_speedup列は
    存在しない。

    ウィジェット自体が無い(netkeiba側にAI展開データが無い)日は空DataFrameを返す
    (スクレイピング失敗ではなく実際にデータが無い状態)。
    """
    return _parse_corner_case(html, race_id, "corner3", include_speedup=False)
