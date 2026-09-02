"""db.netkeiba.com/jockey/{jockey_id}/ と db.netkeiba.com/trainer/{trainer_id}/ の
プロフィールページ(騎手・調教師で構造がほぼ共通)のパース。

実データ確認(2026-09-02、jockey_id=01087/01123/01120/05688、trainer_id=01114で確認)で
判明した構造:

- 氏名・生年月日・所属: `<div class="Name"><h1>氏名&nbsp;(カナ)</h1><p class="txt_01">
  YYYY/MM/DD [所属地]所属形態&nbsp;<!-- 現役 --></p></div>`。所属欄の書式は3パターン確認済み:
  `[栗東]フリー`(JRA所属騎手、地区+フリー/厩舎名)、`地方`(NAR所属騎手、角括弧無し)、
  `美浦`(調教師、角括弧無し・所属形態相当の記載自体が無い)。角括弧が無いケースは
  「所属地のみで所属形態の概念自体が無い/不明」として扱う。生年月日を正規表現で先に
  抜き出してから残りのテキストで判定するため、この3パターンをまとめて1つのロジックで扱える。
  `<!-- 現役 -->`はHTMLコメントでbeautifulsoupのテキスト抽出には出てこない(実質デッドコード
  の可能性が高く、引退騎手・調教師での見え方は未検証のため、在籍状況の判定には使わない)。

- 年度別成績: ページ内に`table.ResultsByYears`が**常に3つ**存在する
  (`#ResultsBox0`=中央/JRA成績、`#ResultsBox1`=地方/NAR成績、`#ResultsBox`(添字無し)=
  常に空の未使用テンプレート、実データ確認で3件とも確認済み)。中央のみで活動する騎手は
  `#ResultsBox1`側が「年度」空欄の空行のみ、地方のみで活動する騎手は`#ResultsBox0`側が
  同様に空、という形で自然に埋め分けられる(実データ: jockey_id=05688(地方所属)で確認)。
  1行目=累計、2行目=直近年度、以降はプレミアム会員限定で現時点では取得不可。列は騎手が
  「騎乗回数」・調教師が「出走回数」とラベルが違うだけで位置(1着/2着/3着/4着~/出走系/
  重賞出走/重賞勝利/勝率/連対率/複勝率)は共通なので、ラベルではなく列位置で読む。
  中央・地方それぞれの成績を`jra_*`/`nar_*`のプレフィックス付き列として両方保持する
  (中央所属騎手が地方交流重賞に出走する等のクロスオーバーも意味のある予想ファクターのため)。

ページ内のJSローカル変数`TozaiCD`('1'=美浦/'2'=栗東/'3'=地方所属)がどちらのタブを既定表示に
するかを決めているが、上記の通り両方のテーブルを常に取得するため、このスクリプトでは
`TozaiCD`自体は使わない(所属地は`affiliation_area`で別途取得済み)。
"""
import re

import pandas as pd
from bs4 import BeautifulSoup

_BIRTH_DATE_RE = re.compile(r"(\d{4}/\d{2}/\d{2})")
_BRACKETED_AREA_RE = re.compile(r"^\[(.+?)\](.*)$")

# ResultsByYears table column order (0-indexed), common to jockey and trainer pages:
# 年度, 順位, 1着, 2着, 3着, 4着~, 騎乗/出走回数, 重賞出走, 重賞勝利, 勝率, 連対率, 複勝率, 代表馬.
_COL_FINISH1 = 2
_COL_STARTS = 6
_COL_WIN_RATE = 9

_STAT_FIELDS = [
    "season",
    "season_rank",
    "season_starts",
    "season_wins",
    "season_win_rate",
    "career_starts",
    "career_wins",
    "career_win_rate",
]

_COLUMNS = (
    ["id", "name", "name_kana", "birth_date", "affiliation_area", "affiliation_type"]
    + [f"jra_{f}" for f in _STAT_FIELDS]
    + [f"nar_{f}" for f in _STAT_FIELDS]
)


def _parse_header(soup: BeautifulSoup) -> dict:
    name_div = soup.select_one("div.db_head_name div.Name")
    if name_div is None:
        raise ValueError("div.db_head_name div.Name not found on person profile page")

    h1 = name_div.find("h1")
    name_text = h1.get_text(" ", strip=True) if h1 else ""
    # "氏名 (カナ)" -> split off the parenthesised kana, tolerating full-width parens too.
    kana_match = re.search(r"[（(](.+?)[）)]", name_text)
    name_kana = kana_match.group(1) if kana_match else ""
    name = re.sub(r"[（(].+?[）)]", "", name_text).strip()

    p = name_div.find("p", class_="txt_01")
    p_text = p.get_text(" ", strip=True) if p else ""
    birth_match = _BIRTH_DATE_RE.search(p_text)
    birth_date = birth_match.group(1) if birth_match else ""
    remainder = p_text[birth_match.end():].strip() if birth_match else p_text

    bracket_match = _BRACKETED_AREA_RE.match(remainder)
    if bracket_match:
        affiliation_area = bracket_match.group(1).strip()
        affiliation_type = bracket_match.group(2).strip()
    else:
        affiliation_area = remainder.strip()
        affiliation_type = ""

    return {
        "name": name,
        "name_kana": name_kana,
        "birth_date": birth_date,
        "affiliation_area": affiliation_area,
        "affiliation_type": affiliation_type,
    }


def _cell_text(tds, index: int) -> str:
    if index >= len(tds):
        return ""
    a = tds[index].find("a")
    return (a or tds[index]).get_text(strip=True)


def _parse_results_by_years(soup: BeautifulSoup, box_id: str) -> dict:
    """Returns career (累計 row) + most recent season (first non-累計 row) stats from
    the table.ResultsByYears inside #{box_id} (either "ResultsBox0"=中央/JRA or
    "ResultsBox1"=地方/NAR - see module docstring). Older years require a premium
    account and simply aren't in the DOM (rendered as a Premium_Regist_Box instead) -
    this only ever sees what's actually there. A jockey/trainer inactive on that
    circuit gets an all-empty year label in this box's only row, which naturally
    falls through to all-empty stats below."""
    stats = {field: "" for field in _STAT_FIELDS}
    box = soup.find(id=box_id)
    table = box.select_one("table.ResultsByYears") if box else None
    if table is None:
        return stats
    # note: an inactive-circuit box's own placeholder row can render literal "-"
    # in its numeric cells (confirmed for career_* on jockey_id=05688's jra box)
    # rather than leaving them blank - that's the source page's own convention
    # for "no data", passed through as-is rather than normalized to "".

    rows = table.select("tbody tr")
    for tr in rows:
        tds = tr.find_all("td")
        if not tds:
            continue
        year_label = tds[0].get_text(strip=True)
        if year_label == "累計":
            stats["career_starts"] = _cell_text(tds, _COL_STARTS)
            stats["career_wins"] = _cell_text(tds, _COL_FINISH1)
            stats["career_win_rate"] = _cell_text(tds, _COL_WIN_RATE)
        elif year_label and stats["season"] == "":
            # First non-累計 row = most recent season with data (rows are in
            # descending-year order; older premium-only years aren't rendered at all).
            stats["season"] = year_label
            rank_text = tds[1].get_text(strip=True) if len(tds) > 1 else ""
            stats["season_rank"] = rank_text
            stats["season_starts"] = _cell_text(tds, _COL_STARTS)
            stats["season_wins"] = _cell_text(tds, _COL_FINISH1)
            stats["season_win_rate"] = _cell_text(tds, _COL_WIN_RATE)
    return stats


def parse_person_profile(html: str, person_id: str, kind: str) -> pd.DataFrame:
    """kind: "jockey" or "trainer" (both pages share this structure; kind is only
    used for the error message / not for any structural branching)."""
    soup = BeautifulSoup(html, "lxml")
    header = _parse_header(soup)
    jra_stats = _parse_results_by_years(soup, "ResultsBox0")
    nar_stats = _parse_results_by_years(soup, "ResultsBox1")

    row = {
        "id": person_id,
        **header,
        **{f"jra_{k}": v for k, v in jra_stats.items()},
        **{f"nar_{k}": v for k, v in nar_stats.items()},
    }
    return pd.DataFrame([row], columns=_COLUMNS)
