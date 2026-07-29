import pandas as pd

# Tabs on holding_time.html: each horse's best recorded time, categorized by
# how today's race distance compares to the distance it was run at.
# just = same track & distance as today's race; short/middle/long = shorter/
# middle/longer distance bucket than today's race (netkeiba's own bucketing,
# not redefined here).
_TABS = ["just", "short", "middle", "long"]
_DETAIL_FIELDS = [
    "race_id",
    "race_name",
    "race_date",
    "jyo",
    "baba",
    "babasa",
    "kinryo",
    "time",
    "l3f",
    "jyuni",
    "kyakusitu",
    "wakuban",
    "pace",
    "sex",
    "age",
]


def parse_holding_time(payload: dict, race_id: str) -> pd.DataFrame:
    """payload is the dict returned by fetch_holding_time_data (the
    nkrace_freq_sum::{race_id} value). One row per horse, keyed by horse_id
    (this API has no umaban field - callers must merge on horse_id).

    fetch_holding_time_data already raises if the API call itself failed or
    came back in an unexpected shape, so an empty freq_horse_all here means
    the API succeeded but genuinely has nothing to report (e.g. every
    runner is a debut horse with no recorded best time) - return an empty
    frame rather than raising."""
    freq_horse_all = payload.get("freq_horse_all")
    if not freq_horse_all:
        return pd.DataFrame(columns=["horse_id"])

    records = []
    for horse_id, horse_data in freq_horse_all.items():
        record = {"horse_id": str(horse_id)}
        best_time_detail = horse_data.get("best_time_detail") or {}
        for tab in _TABS:
            prefix = f"holdtime_{tab}"
            detail = best_time_detail.get(tab)
            for field in _DETAIL_FIELDS:
                value = detail.get(field) if detail else None
                record[f"{prefix}_{field}"] = "" if value is None else str(value)
        records.append(record)

    return pd.DataFrame.from_records(records)
