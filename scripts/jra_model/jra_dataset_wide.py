# -*- coding: utf-8 -*-
"""JRAファクター検証データベース用の拡張母集団ローダー。`jra_dataset.py`は無改造。

`jra_dataset.py`は「新馬|未勝利」を正規表現で除外した通常戦のみを対象にするが、
このファイルはレース名から`race_type`(normal/shinba/mishoubi)をタグ付けし、
通常戦・新馬戦・未勝利戦の3母集団を**全て**まとめて返す(データベース側で
レース種別ごとにフィルタできるようにするため)。

除外条件(いずれもjra_dataset.pyと矛盾しないよう設計、詳細はplanレビュー参照):
  - 障害戦(新馬・未勝利かどうかを問わず「障害」を含むレース名)は全て対象外
    (平地専用のLEGACY_SIGNALS等を適用するのが意味的に不適切なため)。
  - 上記を除いた上で「新馬」を含むレース名→shinba、「未勝利」を含むレース名→mishoubi、
    どちらも含まない→normal。

新馬・未勝利それぞれ実測54・195レース(2026-08-30時点、14開催日)。
"""
import pickle
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
CACHE = DATA_DIR / "jra_dataset_wide_cache.pkl"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.netkeiba_pipeline.storage.paths import newspaper_csv_path  # noqa: E402

import jra_dataset as JD  # noqa: E402 (BET_TYPES/parse_comboを再利用、無改造)
import jra_signals as JS  # noqa: E402 (_drop_scratchedを再利用、無改造)

BET_TYPES = JD.BET_TYPES
HURDLE_RE = re.compile("障害")
SHINBA_RE = re.compile("新馬")
MISHOUBI_RE = re.compile("未勝利")


def race_type_of(race_name: str) -> str | None:
    """レース名からrace_typeを判定する。C-3レビュー反映: 障害戦は新馬・未勝利かどうかに
    関わらず全て対象外(Noneを返す)。"""
    if HURDLE_RE.search(race_name):
        return None
    if SHINBA_RE.search(race_name):
        return "shinba"
    if MISHOUBI_RE.search(race_name):
        return "mishoubi"
    return "normal"


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
    meta["race_type"] = meta["race_name"].apply(race_type_of)
    meta = meta[meta["race_type"].notna()]

    ped_features = None
    ped_cache = DATA_DIR / "pedigree_sire_features_cache.csv"
    if ped_cache.exists():
        ped_features = pd.read_csv(ped_cache, dtype=str, encoding="utf-8")

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
        if ped_features is not None:
            race_ped = ped_features[ped_features["race_id"] == row["race_id"]].drop(columns=["race_id"])
            df = df.merge(race_ped, on="horse_id", how="left")
        races.append({
            "race_id": row["race_id"], "kaisai_date": row["kaisai_date"],
            "racecourse": row["racecourse"], "race_name": row["race_name"],
            "race_type": row["race_type"], "df": df,
        })

    used_dates = sorted({r["kaisai_date"] for r in races})
    actual, offered_bet_types = {}, {}
    if used_dates:
        payouts = pd.concat(
            [pd.read_csv(payouts_dir / f"{d}.csv", dtype=str) for d in used_dates], ignore_index=True
        )
        payouts["payout"] = payouts["payout"].astype(int)
        for race_id, g in payouts.groupby("race_id"):
            # D-1レビュー反映: そのレースで実際に発売された券種一覧(払戻CSVに1行でも
            # 存在する券種)。「未発売」と「発売されたが外れ」を区別するために必須。
            offered_bet_types[race_id] = sorted(set(g["bet_type"]) & set(BET_TYPES))
            per_bt = {}
            for bt in BET_TYPES:
                rows = g[g["bet_type"] == bt]
                m = {}
                for c, p in zip(rows["combination"], rows["payout"]):
                    key = JD.parse_combo(bt, c)
                    if key is None:
                        continue
                    m[key] = m.get(key, 0) + p
                per_bt[bt] = m
            actual[race_id] = per_bt
    for r in races:
        if r["race_id"] not in offered_bet_types:
            offered_bet_types[r["race_id"]] = []

    data = {
        "races": races, "actual": actual, "offered_bet_types": offered_bet_types,
        "dates": used_dates, "skipped": skipped,
    }
    if verbose:
        by_type = pd.Series([r["race_type"] for r in races]).value_counts()
        print(f"usable races: {len(races)}  skipped: {len(skipped)}")
        print("races per race_type:\n" + by_type.to_string())
        by_date = pd.Series([r["kaisai_date"] for r in races]).value_counts().sort_index()
        print("races per date:\n" + by_date.to_string())
    return data


def load(rebuild: bool = False) -> dict:
    if CACHE.exists() and not rebuild:
        with CACHE.open("rb") as f:
            return pickle.load(f)
    data = build()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("wb") as f:
        pickle.dump(data, f)
    return data


if __name__ == "__main__":
    load(rebuild=True)
