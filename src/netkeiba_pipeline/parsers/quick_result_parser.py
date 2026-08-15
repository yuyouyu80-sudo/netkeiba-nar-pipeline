import re

import pandas as pd
from bs4 import BeautifulSoup

HORSE_ID_RE = re.compile(r"/horse/(\d+)")

_QUICK_RESULT_COLUMNS = [
    "race_id",
    "finish1_umaban", "finish1_horse", "finish1_ninki", "finish1_odds",
    "finish2_umaban", "finish2_horse",
    "finish3_umaban", "finish3_horse",
    "finish4_umaban", "finish4_horse",
    "finish5_umaban", "finish5_horse",
    "tansho_payout", "tansho_ninki",
]


def _finish_row(tr) -> dict:
    rank_td = tr.find("td", class_="Result_Num")
    num_tds = tr.find_all("td", class_="Num")
    horse_td = tr.find("td", class_="Horse_Info")
    ninki_span = tr.find("span", class_="OddsPeople")
    odds_td = tr.select_one("td.Odds.Txt_R")
    if rank_td is None or len(num_tds) < 2 or horse_td is None:
        raise ValueError("quick result row missing 着順/馬番/馬名 cell")
    horse_a = horse_td.find("a")
    return {
        "rank": rank_td.get_text(strip=True),
        "umaban": num_tds[1].get_text(strip=True),
        "horse": horse_a.get_text(strip=True) if horse_a else horse_td.get_text(strip=True),
        "ninki": ninki_span.get_text(strip=True) if ninki_span else "",
        "odds": odds_td.get_text(strip=True) if odds_td else "",
    }


def parse_quick_result(html: str, race_id: str) -> pd.DataFrame:
    """race.netkeiba.com(JRA)/nar.netkeiba.com(NAR)のrace/result.htmlを1着〜5着+単勝払戻
    だけの簡易1行に要約する(2026-08-15、3着までから5着までに拡張)。
    発走前・速報未掲載のレースはAll_Result_Tableが存在しないため、その場合は
    空のDataFrame(パース失敗ではなく「まだ結果が無い」)を返す。"""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="All_Result_Table")
    if table is None:
        return pd.DataFrame(columns=_QUICK_RESULT_COLUMNS)

    rows = table.select("tbody tr")
    if not rows:
        return pd.DataFrame(columns=_QUICK_RESULT_COLUMNS)

    finishes = [_finish_row(tr) for tr in rows[:5]]
    while len(finishes) < 5:
        finishes.append({"umaban": "", "horse": "", "ninki": "", "odds": ""})

    tansho_payout, tansho_ninki = "", ""
    payout_table = soup.find("table", class_="Payout_Detail_Table")
    if payout_table is not None:
        tansho_tr = payout_table.find("tr", class_="Tansho")
        if tansho_tr is not None:
            payout_td = tansho_tr.find("td", class_="Payout")
            ninki_td = tansho_tr.find("td", class_="Ninki")
            tansho_payout = payout_td.get_text(strip=True) if payout_td else ""
            tansho_ninki = ninki_td.get_text(strip=True) if ninki_td else ""

    record = {
        "race_id": race_id,
        "finish1_umaban": finishes[0]["umaban"], "finish1_horse": finishes[0]["horse"],
        "finish1_ninki": finishes[0]["ninki"], "finish1_odds": finishes[0]["odds"],
        "finish2_umaban": finishes[1]["umaban"], "finish2_horse": finishes[1]["horse"],
        "finish3_umaban": finishes[2]["umaban"], "finish3_horse": finishes[2]["horse"],
        "finish4_umaban": finishes[3]["umaban"], "finish4_horse": finishes[3]["horse"],
        "finish5_umaban": finishes[4]["umaban"], "finish5_horse": finishes[4]["horse"],
        "tansho_payout": tansho_payout, "tansho_ninki": tansho_ninki,
    }
    return pd.DataFrame.from_records([record])
