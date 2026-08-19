# -*- coding: utf-8 -*-
"""JRA再モデリング用の共通データセット層。scripts/nar_model/nar_dataset.pyのJRA移植版。

検証済み(race_results と payouts の両方が存在する)JRA開催日のうち、馬柱CSVが実在する
通常戦レース(新馬・未勝利を除く)だけを集めて、以下をまとめてpickleにキャッシュする:

  races:   [{race_id, kaisai_date, racecourse, race_name, df(馬柱DataFrame、取消除外済み)}]
  actual:  {race_id: {bet_type: {組合せkey: 払戻合計}}}

重み探索・回収率検証のスクリプトはここからロードするだけにして、「どのレースを母集団に
したか」の定義が実装ごとにブレるのを防ぐ(除外条件はscripts/predict_pattern29.py L318・
confidence_sweep_v2.py L54と同一の"新馬|未勝利"正規表現)。
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
CACHE = DATA_DIR / "jra_dataset_cache.pkl"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.netkeiba_pipeline.storage.paths import newspaper_csv_path  # noqa: E402

import jra_signals as JS  # noqa: E402

# JRAは枠連が有効(waku列が充填されている)ため、NAR(nar_dataset.pyでは枠連を除外)と異なり
# 8券種すべてを対象にする。
BET_TYPES = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "3連複", "3連単"]


def parse_combo(bet_type: str, combo_text: str):
    if bet_type in ("単勝", "複勝"):
        return int(combo_text)
    if bet_type in ("馬単", "3連単"):
        return tuple(int(x) for x in combo_text.split("→"))
    if "→" in combo_text:
        return None
    return frozenset(int(x) for x in combo_text.split("-"))


def build(verbose: bool = True) -> dict:
    results_dir = PROJECT_ROOT / "data" / "race_results" / "2026"
    payouts_dir = PROJECT_ROOT / "data" / "payouts" / "2026"
    dates = sorted(p.stem for p in results_dir.glob("2026*.csv") if (payouts_dir / p.name).exists())

    meta_rows = []
    for d in dates:
        df = pd.read_csv(results_dir / f"{d}.csv", dtype=str)
        names = df[["race_id", "race_name", "racecourse"]].drop_duplicates("race_id").copy()
        names["kaisai_date"] = d
        meta_rows.append(names)
    meta = pd.concat(meta_rows, ignore_index=True)
    meta = meta[~meta["race_name"].str.contains("新馬|未勝利", regex=True, na=False)]

    races, skipped = [], []
    for _, row in meta.iterrows():
        path = newspaper_csv_path(row["race_id"])
        if not path.exists():
            skipped.append((row["race_id"], "no_newspaper"))
            continue
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
        if df.empty:
            skipped.append((row["race_id"], "empty"))
            continue
        df = JS._drop_scratched(df)
        if df.empty:
            skipped.append((row["race_id"], "all_scratched"))
            continue
        races.append({
            "race_id": row["race_id"], "kaisai_date": row["kaisai_date"],
            "racecourse": row["racecourse"], "race_name": row["race_name"], "df": df,
        })

    used_dates = sorted({r["kaisai_date"] for r in races})
    if used_dates:
        payouts = pd.concat(
            [pd.read_csv(payouts_dir / f"{d}.csv", dtype=str) for d in used_dates], ignore_index=True
        )
        payouts["payout"] = payouts["payout"].astype(int)
    else:
        payouts = pd.DataFrame(columns=["race_id", "bet_type", "combination", "payout"])

    actual = {}
    for race_id, g in payouts.groupby("race_id"):
        per_bt = {}
        for bt in BET_TYPES:
            rows = g[g["bet_type"] == bt]
            m = {}
            for c, p in zip(rows["combination"], rows["payout"]):
                key = parse_combo(bt, c)
                if key is None:
                    continue
                m[key] = m.get(key, 0) + p
            per_bt[bt] = m
        actual[race_id] = per_bt

    data = {"races": races, "actual": actual, "dates": used_dates, "skipped": skipped}
    if verbose:
        by_date = pd.Series([r["kaisai_date"] for r in races]).value_counts().sort_index()
        print(f"dates(results&payouts): {dates}")
        print(f"usable races: {len(races)}  skipped: {len(skipped)}")
        print("races per date:\n" + by_date.to_string())
        sizes = pd.Series([len(r["df"]) for r in races])
        print(f"field size: min={sizes.min()} median={sizes.median()} max={sizes.max()} mean={sizes.mean():.1f}")
    return data


def load(rebuild: bool = False) -> dict:
    if CACHE.exists() and not rebuild:
        with CACHE.open("rb") as f:
            return pickle.load(f)
    data = build()
    with CACHE.open("wb") as f:
        pickle.dump(data, f)
    return data


if __name__ == "__main__":
    d = build()
    with CACHE.open("wb") as f:
        pickle.dump(d, f)
    print(f"\nwrote {CACHE}")
