import re

import pandas as pd
from bs4 import BeautifulSoup

HORSE_ID_RE = re.compile(r"/horse/(\d+)")
_COLOR_RE = re.compile(r"background:\s*(#[0-9A-Fa-f]{6})")

# table.Bias ("血統ビーム出馬表") color-codes each horse's 父(sire)/母父(broodmare
# sire) cell by pedigree line (系統), per the page's own Bias_Colorling legend.
_BLOODLINE_BY_COLOR = {
    "#C4F2F9": "サンデーサイレンス系",
    "#C6FFAA": "ターントゥ系",
    "#E0B7FF": "ノーザンダンサー系",
    "#FFA6E2": "ナスルーラ系",
    "#FFD28E": "ネイティヴダンサー系",
    "#E8BF9B": "ハンプトン系",
    "#FFFF99": "セントサイモン系",
    "#DDDDDD": "その他",
}


def _bloodline(td) -> str:
    match = _COLOR_RE.search(td.get("style", ""))
    if match is None:
        return ""
    return _BLOODLINE_BY_COLOR.get(match.group(1).upper(), "")


_BIAS_COLUMNS = [
    "umaban",
    "bias_horse_id",
    "bias_horse_name",
    "bias_sire",
    "bias_sire_bloodline",
    "bias_dam_sire",
    "bias_dam_sire_bloodline",
    "bias_sex_age",
    "bias_weight_carried",
    "bias_horse_weight",
    "bias_jockey",
    "bias_trainer",
    "bias_win_odds",
    "bias_ninki",
]


def parse_bias(html: str, race_id: str) -> pd.DataFrame:
    """table.Bias (血統ビーム出馬表): one row per horse, with 父/母父 name plus
    their color-coded pedigree line (系統).

    Cell classes here are not reliable landmarks: a scratched (取消) horse's
    row drops or renames several of them (sex/weight lose "Txt_C", jockey/
    trainer switch from plain <td> to "Txt_L", 人気 loses its class
    entirely), and independently some race classes insert an extra 馬体重
    (増減) column between 厩舎 and オッズ that others don't have - the two
    variations combine, so a class-based or fixed-<td>-count read breaks on
    some race/scratch combination. Instead this walks forward from the two
    always-identifiable td.Blood_Cell (sire/dam-sire) cells: sex_age and
    weight_carried are always the next two siblings, jockey and trainer the
    two after that, and 単勝オッズ (td.Txt_R, always present) anchors the
    tail end - whatever sits between trainer and the odds cell is the
    optional horse_weight column."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.Bias")
    if table is None:
        return pd.DataFrame(columns=_BIAS_COLUMNS)

    rows = table.select("tbody tr.HorseList")
    if not rows:
        return pd.DataFrame(columns=_BIAS_COLUMNS)

    records = []
    for tr in rows:
        umaban_td = tr.find("td", class_="UmaBan")
        horse_td = tr.find("td", class_="Horse_Name")
        blood_tds = tr.find_all("td", class_="Blood_Cell")
        label = f"race_id={race_id} umaban={umaban_td.get_text(strip=True) if umaban_td else '?'}"
        if umaban_td is None or horse_td is None or len(blood_tds) != 2:
            raise ValueError(f"{label}: could not find umaban/horse-name/blood cells in a Bias table row")
        sire_td, dam_sire_td = blood_tds

        sex_age_td = dam_sire_td.find_next_sibling("td")
        weight_td = sex_age_td.find_next_sibling("td") if sex_age_td is not None else None
        jockey_td = weight_td.find_next_sibling("td") if weight_td is not None else None
        trainer_td = jockey_td.find_next_sibling("td") if jockey_td is not None else None
        if None in (sex_age_td, weight_td, jockey_td, trainer_td):
            raise ValueError(f"{label}: Bias table row ended before 性齢/斤量/騎手/厩舎 cells - too few <td>")

        after_trainer = trainer_td.find_next_sibling("td")
        if after_trainer is not None and "Txt_R" in (after_trainer.get("class") or []):
            horse_weight_td = None
            odds_td = after_trainer
        else:
            horse_weight_td = after_trainer
            odds_td = after_trainer.find_next_sibling("td") if after_trainer is not None else None
        if odds_td is None or "Txt_R" not in (odds_td.get("class") or []):
            raise ValueError(f"{label}: could not locate 単勝オッズ (td.Txt_R) cell in a Bias table row")

        ninki_td = odds_td.find_next_sibling("td")
        if ninki_td is None:
            raise ValueError(f"{label}: Bias table row has no 人気 cell after オッズ")

        horse_a = horse_td.find("a", href=True)
        horse_id_match = HORSE_ID_RE.search(horse_a["href"]) if horse_a else None
        sire_a = sire_td.find("a")
        dam_sire_a = dam_sire_td.find("a")
        jockey_a = jockey_td.find("a")
        trainer_a = trainer_td.find("a")

        records.append(
            {
                "umaban": umaban_td.get_text(strip=True),
                "bias_horse_id": horse_id_match.group(1) if horse_id_match else "",
                "bias_horse_name": horse_a.get_text(strip=True) if horse_a else "",
                "bias_sire": sire_a.get_text(strip=True) if sire_a else "",
                "bias_sire_bloodline": _bloodline(sire_td),
                "bias_dam_sire": dam_sire_a.get_text(strip=True) if dam_sire_a else "",
                "bias_dam_sire_bloodline": _bloodline(dam_sire_td),
                "bias_sex_age": sex_age_td.get_text(strip=True),
                "bias_weight_carried": weight_td.get_text(strip=True),
                "bias_horse_weight": horse_weight_td.get_text(strip=True) if horse_weight_td else "",
                "bias_jockey": jockey_a.get_text(strip=True) if jockey_a else "",
                "bias_trainer": trainer_a.get_text(strip=True) if trainer_a else "",
                "bias_win_odds": odds_td.get_text(strip=True),
                "bias_ninki": ninki_td.get_text(strip=True),
            }
        )

    return pd.DataFrame.from_records(records)
