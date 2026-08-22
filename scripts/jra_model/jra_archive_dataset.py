# -*- coding: utf-8 -*-
"""馬柱に依存しないJRA母集団層(2026-08-23新設、Step1再挑戦・選択肢B Phase 3)。

`jra_dataset.py`と同じ戻り値契約({"races","actual","dates","skipped"})を返すが、
races[i]["df"]は馬柱DataFrameではなく、race_resultsアーカイブ由来の1頭1行DataFrameに
`jra_history.past_frame()`が生成する past{i}_* 列を結合したもの。

**`jra_dataset.py`は変更しない**(211レース時代の結果の再現性を保つため、本モジュールは
完全に別ファイルとして新設する)。

母集団は既定で2025-04-01〜2026-02-28に絞る(2026-08-22のOpus 5サブエージェント調査により、
2024年後半は履歴被覆が薄くバイアスの原因になること、2026年7-8月は既存探索で使い切った
「汚染済み」区間であることが判明したため)。履歴インデックス(HorseHistoryIndex)自体は
アーカイブ全期間(2024-08-24〜)を使う(母集団を絞ることと、参照できる過去走の範囲を絞る
ことは別軸)。

列名エイリアスは張らない(`odds_final`を`bias_win_odds`と偽称しない)。呼び出し側
(jra_market_model.build_composite_features等)に列名パラメータを渡す形で対応する。
"""
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
CACHE = DATA_DIR / "jra_archive_dataset_cache.pkl"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_dataset as JD  # noqa: E402 (parse_comboを再利用)
import jra_history as JH  # noqa: E402

HISTORY_START = "2024-08-24"  # 履歴インデックスの下限(母集団にはしない)
EXCLUDE_RACE_NAME = r"新馬|未勝利|障害"
MODEL_START_DEFAULT = "2025-04-01"
MODEL_END_DEFAULT = "2026-02-28"
N_PAST = 5

# past_frame()が生成する列のうち、race_resultsの1頭1行DataFrameへそのままコピーする列。
PAST_COLS = [f"past{i}_{c}" for i in range(1, N_PAST + 1)
             for c in ("finish", "beaten_by", "agari_3f", "corner_positions",
                        "field_size", "race_class", "date", "race_id")]


def _load_payouts(dates_iso: list) -> pd.DataFrame:
    """race_date(ISO)のリストから該当するpayouts/{year}/{YYYYMMDD}.csvを全件連結する。"""
    frames = []
    seen = set()
    for d in dates_iso:
        ymd = d.replace("-", "")
        year = ymd[:4]
        if ymd in seen:
            continue
        seen.add(ymd)
        p = PROJECT_ROOT / "data" / "payouts" / year / f"{ymd}.csv"
        if p.exists():
            frames.append(pd.read_csv(p, dtype=str))
    if not frames:
        return pd.DataFrame(columns=["race_id", "bet_type", "combination", "payout"])
    df = pd.concat(frames, ignore_index=True)
    df["payout"] = df["payout"].astype(int)
    return df


def build(model_start: str = MODEL_START_DEFAULT, model_end: str = MODEL_END_DEFAULT,
          exclude_race_name: str = EXCLUDE_RACE_NAME, min_starters: int = 2,
          verbose: bool = True) -> dict:
    if verbose:
        print(f"母集団期間: {model_start} 〜 {model_end}  除外パターン: {exclude_race_name!r}")

    all_results = JH.load_results()
    idx = JH.HorseHistoryIndex(all_results)
    if verbose:
        print(f"履歴インデックス: {len(all_results)}行 {len(idx._by_horse)}頭 "
              f"min_date={idx.min_date}")

    pop = all_results[
        (all_results["race_date"] >= model_start) & (all_results["race_date"] <= model_end)
    ].copy()
    pop = pop[~pop["race_name"].str.contains(exclude_race_name, regex=True, na=False)]

    meta = pop.drop_duplicates("race_id")[
        ["race_id", "race_date", "racecourse", "race_name"]
    ].sort_values(["race_date", "race_id"])

    races, skipped = [], []
    for _, row in meta.iterrows():
        race_id = row["race_id"]
        g = pop[pop["race_id"] == race_id].copy()
        starters = g[JH.is_starter(g)].reset_index(drop=True)
        if len(starters) < min_starters:
            skipped.append((race_id, "too_few_starters"))
            continue

        horse_ids = starters["horse_id"].tolist()
        past_pf = JH.past_frame(idx, horse_ids, row["race_date"], n=N_PAST)
        df = starters[["umaban", "horse_id", "horse_name", "odds_final", "popularity",
                        "finish_pos", "waku", "jockey_id", "trainer_id"]].copy()
        for col in PAST_COLS:
            df[col] = past_pf[col].to_numpy()

        kaisai_date = row["race_date"].replace("-", "")
        races.append({
            "race_id": race_id, "kaisai_date": kaisai_date, "race_date": row["race_date"],
            "racecourse": row["racecourse"], "race_name": row["race_name"], "df": df,
        })

    used_dates_iso = sorted({r["race_date"] for r in races})
    payouts = _load_payouts(used_dates_iso)
    actual = {}
    for race_id, g in payouts.groupby("race_id"):
        per_bt = {}
        for bt in JD.BET_TYPES:
            rows = g[g["bet_type"] == bt]
            m = {}
            for c, p in zip(rows["combination"], rows["payout"]):
                key = JD.parse_combo(bt, c)
                if key is None:
                    continue
                m[key] = m.get(key, 0) + p
            per_bt[bt] = m
        actual[race_id] = per_bt

    used_dates = sorted({r["kaisai_date"] for r in races})
    data = {"races": races, "actual": actual, "dates": used_dates, "skipped": skipped}
    if verbose:
        print(f"usable races: {len(races)}  skipped: {len(skipped)}")
        if races:
            by_date = pd.Series([r["kaisai_date"] for r in races]).value_counts().sort_index()
            print(f"開催日数: {by_date.shape[0]}")
            sizes = pd.Series([len(r["df"]) for r in races])
            print(f"field size: min={sizes.min()} median={sizes.median()} max={sizes.max()} "
                  f"mean={sizes.mean():.1f}")
    return data


def load(rebuild: bool = False, **build_kwargs) -> dict:
    if CACHE.exists() and not rebuild:
        with CACHE.open("rb") as f:
            return pickle.load(f)
    data = build(**build_kwargs)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("wb") as f:
        pickle.dump(data, f)
    return data


if __name__ == "__main__":
    d = build()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("wb") as f:
        pickle.dump(d, f)
    print(f"\nwrote {CACHE}")
