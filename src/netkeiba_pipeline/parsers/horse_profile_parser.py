"""db.netkeiba.com/horse/{horse_id}/ (競走馬プロフィールページ本体、血統表サブページ
`/horse/ped/{horse_id}/`とは別)のうち、生産者(breeder)・馬主(owner)欄のパース。

実データ確認(2026-09-02、horse_id=2018101615で確認)で判明した構造: `table.db_prof_table`
(summary属性は空/"のプロフィール"のみで信頼できない)内の各`<tr>`が`<th>ラベル</th><td>値</td>`
の1行1属性形式("生年月日"/"調教師"/"馬主"/"生産者"等)。ラベルの並び順・行数はレースの種類
(募集情報の有無等)で変動しうるため、固定インデックスではなく`<th>`テキストで対象行を探す。

馬主・生産者ともに値セルは`<a href="https://db.netkeiba.com/owner/{owner_id}/">{name}</a>`
(馬主は加えて`<img class="OwnerColours">`が前置される)の形。IDはURLから正規表現で抽出する。
"""
import re

import pandas as pd
from bs4 import BeautifulSoup

OWNER_ID_RE = re.compile(r"/owner/(\w+)")
BREEDER_ID_RE = re.compile(r"/breeder/(\w+)")

_COLUMNS = ["horse_id", "owner", "owner_id", "breeder", "breeder_id"]


def parse_horse_profile(html: str, horse_id: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.db_prof_table")
    if table is None:
        raise ValueError(f"horse_id={horse_id}: table.db_prof_table not found on horse profile page")

    row = {"horse_id": horse_id, "owner": "", "owner_id": "", "breeder": "", "breeder_id": ""}
    for tr in table.find_all("tr"):
        th = tr.find("th")
        td = tr.find("td")
        if th is None or td is None:
            continue
        label = th.get_text(strip=True)
        a = td.find("a", href=True)
        if label == "馬主":
            row["owner"] = a.get_text(strip=True) if a else ""
            match = OWNER_ID_RE.search(a["href"]) if a else None
            row["owner_id"] = match.group(1) if match else ""
        elif label == "生産者":
            row["breeder"] = a.get_text(strip=True) if a else ""
            match = BREEDER_ID_RE.search(a["href"]) if a else None
            row["breeder_id"] = match.group(1) if match else ""

    return pd.DataFrame([row], columns=_COLUMNS)
