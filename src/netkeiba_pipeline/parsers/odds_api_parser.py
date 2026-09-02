"""race.netkeiba.com/api/api_get_jra_odds.htmlのJSON応答パース(複勝・馬連)。
生データの形については src.netkeiba_pipeline.scrapers.odds_api のdocstring参照。

type=1応答には単勝(キー"1")も同時に含まれるが、単勝オッズは既存のbias_parser.py
(watch_odds.pyの既存経路)で別途取得済みのため、ここでは意図的に複勝(キー"2")のみを
対象にする(二重管理を避ける)。
"""
import pandas as pd


def _status_ok(payload: dict) -> bool:
    return payload.get("status") == "result"


def parse_fuku_odds(payload: dict, race_id: str) -> pd.DataFrame:
    """type=1で取得したペイロードから複勝(キー"2")を抽出する。"""
    columns = ["race_id", "umaban", "fuku_odds_low", "fuku_odds_high", "fuku_ninki"]
    if not _status_ok(payload):
        return pd.DataFrame(columns=columns)

    fuku = payload.get("data", {}).get("odds", {}).get("2", {})
    rows = [
        {
            "race_id": race_id,
            "umaban": str(int(umaban)),  # "01" -> "1", horse_id等の他列と表記を揃える
            "fuku_odds_low": values[0],
            "fuku_odds_high": values[1],
            "fuku_ninki": values[2],
        }
        for umaban, values in fuku.items()
    ]
    return pd.DataFrame(rows, columns=columns)


def parse_umaren_odds(payload: dict, race_id: str) -> pd.DataFrame:
    """type=4で取得したペイロードから馬連(キー"4"、4桁の馬番組合せキー)を抽出する。
    キーの並び順はnetkeiba側で既に若い馬番が先("0102"であって"0201"ではない)なので、
    そのまま umaban_a/umaban_b に分解するだけでよい(実データで確認済み)。"""
    columns = ["race_id", "umaban_a", "umaban_b", "umaren_odds", "umaren_ninki"]
    if not _status_ok(payload):
        return pd.DataFrame(columns=columns)

    umaren = payload.get("data", {}).get("odds", {}).get("4", {})
    rows = []
    for combo_key, values in umaren.items():
        if len(combo_key) != 4:
            raise ValueError(f"race_id={race_id}: unexpected umaren combo key {combo_key!r} (want 4 digits)")
        rows.append(
            {
                "race_id": race_id,
                "umaban_a": str(int(combo_key[:2])),
                "umaban_b": str(int(combo_key[2:])),
                "umaren_odds": values[0],
                "umaren_ninki": values[2],
            }
        )
    return pd.DataFrame(rows, columns=columns)
