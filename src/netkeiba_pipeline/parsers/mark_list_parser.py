"""mark_list.html(本紙・CP・その他 予想印)のパース。

実データ確認(2026-08-26、JRA 2レース・NAR 1レース)で判明した構造:
- 専門家ごとに <dl class="Yosoka" id="yoso_goods_seq_N"> (Nは整数)が並ぶ。id無し、または
  id="yoso_goods_seq_builder"(「予想ビルダー」という有償AIツールの広告枠で実専門家ではない)は
  除外する。
- 専門家名は <p class="yosoka_name">本<br>紙<br></p> のように1文字ずつ<br>区切りで入っている。
- 印は各専門家ブロック内の <li class="Mark_Pro mark_N"><span class="Icon_Shirushi Icon_XXX">。
  mark_Nの数値Nは物理的な馬番ではなく内部の出走エントリID。◎○▲△☆の5種類のみ実データで確認済み
  (Icon_Honmei/Icon_Taikou/Icon_Osae/Icon_Kurosan/Icon_Hoshi)。空<span class="">は印なし。
- 馬番との対応は表示順(position)ベース: ページ内の自分の印入力欄(dl.Shirushi)を全専門家列・
  枠番・馬番と同期させるJS(getTargets()/syncSelectedRows())が、これらのリストを同一position
  indexで揃えて扱っている実装から、mark_N自体ではなく「各専門家ブロック内でのli出現順」が
  「馬番dl(2つ目のdl.Umaban)でのli出現順」と1:1対応することを確認済み。
"""
import logging
import re

import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

YOSOKA_ID_RE = re.compile(r"^yoso_goods_seq_(\d+)$")

RAW_MARK_PREFIX = "mark_raw_"

# 2026-08-27修正: Icon_Osae/Icon_Kurosanの対応記号が逆だった(旧: Osae=▲, Kurosan=△)。
# netkeiba公式スプライト画像(race_pc_new.css → icon_yoso_shirushi.png、背景位置
# Icon_Kurosan=0 -38px→▲, Icon_Osae=0 -57px→△)で実際の記号を確認し、あわせて実際の
# mark_list.html表示(ユーザー提供のスクリーンショット、2026-08-22中京1R)と全馬照合して確定。
ICON_CLASS_TO_SYMBOL = {
    "Icon_Honmei": "◎",
    "Icon_Taikou": "○",
    "Icon_Osae": "△",
    "Icon_Kurosan": "▲",
    "Icon_Hoshi": "☆",
}

# 「その他」集計の配点(ユーザー指定、確定仕様)。ここに無い記号(未知のIcon_XXXクラス、
# 空マーク)は0点として扱う。
MARK_SCORE = {
    "◎": 6,
    "○": 4,
    "▲": 3,
    "△": 2,
    "☆": 0.5,
}

HONSHI_NAME = "本紙"
CP_NAME = "CP予想"

# その他集計で上位から割り当てる印(dense ranking: 1位グループ→◎, 2位グループ→○, ...)。
# 5位以下(タイ扱いの詰め順位が5以上)は無印(空文字)のまま。
OTHER_RANK_TO_MARK = {1: "◎", 2: "○", 3: "▲", 4: "△"}
NO_MARK_SCORE_SYMBOL = "★"


def _expert_name(dt_tag) -> str:
    name_p = dt_tag.find("p", class_="yosoka_name")
    if name_p is None:
        return ""
    # 各文字が<br>区切りで別テキストノードに入っているので、テキストノードだけを
    # 抽出して連結する(<img>や装飾用<span>は無視)。
    return "".join(name_p.find_all(string=True)).replace("\xa0", "").strip()


def _umaban_order(soup: BeautifulSoup, race_id: str) -> list[str]:
    """<dl class="Umaban">は2つある(1つ目=枠番、2つ目=馬番)。2つ目のli出現順が、
    各Yosokaブロック内のmark_N li出現順と対応する(モジュールdocstring参照)。"""
    umaban_dls = soup.select("dl.Umaban")
    if len(umaban_dls) < 2:
        raise ValueError(
            f'race_id={race_id}: expected 2 <dl class="Umaban"> (waku, umaban), found '
            f"{len(umaban_dls)} - page structure may have changed"
        )
    umaban_dl = umaban_dls[1]
    return [li.get_text(strip=True) for li in umaban_dl.select("dd li.Num")]


def parse_mark_list(html: str, race_id: str) -> pd.DataFrame:
    """1行1馬、専門家ごとに mark_raw_{専門家名} 列を持つワイド形式の生データを返す。

    専門家が1人も掲載されていないレース(印テーブル自体が無い)は空DataFrameを返す
    (スクレイピング失敗ではなく実際にデータが無い状態、既存パーサーのエラーハンドリング
    規約と同じ)。
    """
    soup = BeautifulSoup(html, "lxml")
    yosoka_blocks = [dl for dl in soup.select("dl.Yosoka[id]") if YOSOKA_ID_RE.match(dl.get("id", ""))]

    if not yosoka_blocks:
        return pd.DataFrame(columns=["umaban"])

    umaban_order = _umaban_order(soup, race_id)

    data: dict[str, list] = {"umaban": umaban_order}
    seen_names: dict[str, int] = {}
    for dl in yosoka_blocks:
        name = _expert_name(dl)
        if not name:
            # 名前が取れないブロックは実専門家ではない(念のための保険、実データでは
            # yoso_goods_seq_N(N=整数)は必ず名前を持つことを確認済み)。
            continue

        if name in seen_names:
            seen_names[name] += 1
            name = f"{name}_{seen_names[name]}"
        else:
            seen_names[name] = 0

        mark_lis = dl.select("dd > ul > li.Mark_Pro")
        if len(mark_lis) != len(umaban_order):
            raise ValueError(
                f"race_id={race_id}: expert={name!r} has {len(mark_lis)} marks, expected "
                f"{len(umaban_order)} (== number of umaban entries) - page structure may have changed"
            )

        symbols = []
        for li in mark_lis:
            span = li.find("span")
            classes = [c for c in (span.get("class", []) if span else []) if c != "Icon_Shirushi"]
            if not classes:
                symbols.append("")
                continue
            icon_class = classes[0]
            symbol = ICON_CLASS_TO_SYMBOL.get(icon_class)
            if symbol is None:
                logger.warning(
                    "race_id=%s: expert=%s has unknown mark icon class %r - keeping raw class name as value",
                    race_id,
                    name,
                    icon_class,
                )
                symbol = icon_class
            symbols.append(symbol)

        data[f"{RAW_MARK_PREFIX}{name}"] = symbols

    return pd.DataFrame(data)


def summarize_marks(raw_df: pd.DataFrame) -> pd.DataFrame:
    """parse_mark_list()の生データから、Excel「本誌・CP・その他」に対応する
    umaban/mark_honshi/mark_cp/mark_other の4列を算出する。

    「その他」集計ロジック(ユーザー指定、確定仕様):
    「本紙」「CP予想」を除く全専門家の印を、馬ごとに◎=6/○=4/▲=3/△=2/☆=0.5点
    (無印・未知の記号は0点)でスコアリングして合計する。レース内の馬をスコア降順で
    順位付けし(同スコアは同順位=dense ranking)、1位→◎、2位→○、3位→▲、4位→△を割り当てる。
    5位以下(上位4段階に入らない馬)は無印(空文字)のまま。

    ★の判定条件(2026-08-26にユーザー指示で「本紙」「CP予想」も無印の場合のみに限定):
    スコアが0点(=本紙・CP予想以外の誰からも印を1つももらっていない馬)、**かつ**
    「本紙」「CP予想」の両方が無印の馬だけが★になる。スコアが0点でも「本紙」か「CP予想」の
    どちらかに印が付いている馬は、順位付けの対象外(スコア0点なので上位4段階には入らない)
    として無印(空文字)になる(★にはしない)。
    """
    if raw_df.empty or "umaban" not in raw_df.columns:
        return pd.DataFrame(columns=["umaban", "mark_honshi", "mark_cp", "mark_other"])

    honshi_col = f"{RAW_MARK_PREFIX}{HONSHI_NAME}"
    cp_col = f"{RAW_MARK_PREFIX}{CP_NAME}"

    out = pd.DataFrame({"umaban": raw_df["umaban"]})
    out["mark_honshi"] = raw_df[honshi_col] if honshi_col in raw_df.columns else ""
    out["mark_cp"] = raw_df[cp_col] if cp_col in raw_df.columns else ""

    other_cols = [c for c in raw_df.columns if c.startswith(RAW_MARK_PREFIX) and c not in (honshi_col, cp_col)]

    scores = pd.Series(0.0, index=raw_df.index)
    for col in other_cols:
        scores = scores + raw_df[col].map(MARK_SCORE).fillna(0.0)

    unique_scores_desc = sorted(scores[scores > 0].unique(), reverse=True)
    rank_of_score = {score: rank + 1 for rank, score in enumerate(unique_scores_desc)}

    honshi_blank = out["mark_honshi"].fillna("").astype(str).eq("")
    cp_blank = out["mark_cp"].fillna("").astype(str).eq("")
    honshi_and_cp_both_blank = honshi_blank & cp_blank

    out["mark_other"] = [
        (
            OTHER_RANK_TO_MARK.get(rank_of_score[score], "")
            if score > 0
            else (NO_MARK_SCORE_SYMBOL if both_blank else "")
        )
        for score, both_blank in zip(scores, honshi_and_cp_both_blank)
    ]
    return out
