# -*- coding: utf-8 -*-
"""NAR再モデリング用の共通データセット層。

検証済み(race_results と payouts の両方が存在する)NAR開催日のうち、馬柱CSVが実在する
通常戦レースだけを集めて、以下をまとめてpickleにキャッシュする:

  races:   [{race_id, kaisai_date, racecourse, race_name, df(馬柱DataFrame、取消除外済み)}]
  actual:  {race_id: {bet_type: {組合せkey: 払戻合計}}}

重み探索・回収率検証のスクリプトはここからロードするだけにして、
「どのレースを母集団にしたか」の定義が実装ごとにブレるのを防ぐ。
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nar_pipeline"
CACHE = DATA_DIR / "nar_dataset_cache.pkl"

sys.path.insert(0, str(PROJECT_ROOT))
from src.netkeiba_pipeline.storage.paths import newspaper_csv_path  # noqa: E402

BET_TYPES = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "3連複", "3連単"]


def _num(series: pd.Series) -> pd.Series:
    cleaned = series.where(~series.astype(str).isin(["-", "--", "nan", ""]), None)
    return pd.to_numeric(cleaned, errors="coerce")


def drop_scratched(df: pd.DataFrame) -> pd.DataFrame:
    """出走取消・除外馬(単勝オッズと人気の両方が空)を落とす。本番predict_pattern29.pyと同じ判定。"""
    odds = _num(df["bias_win_odds"])
    ninki = _num(df["bias_ninki"])
    return df[odds.notna() & ninki.notna()].reset_index(drop=True)


def parse_combo(bet_type: str, combo_text: str):
    """払戻の組合せ表記をキーに変換する。金沢などで枠単が枠連としてラベルされている行が
    あるため、方向つき表記('→')が方向なし券種に来た場合はNoneを返して無視する。"""
    if bet_type in ("単勝", "複勝"):
        return int(combo_text)
    if bet_type in ("馬単", "3連単"):
        return tuple(int(x) for x in combo_text.split("→"))
    if "→" in combo_text:
        return None
    return frozenset(int(x) for x in combo_text.split("-"))


def build(verbose: bool = True, require_pre_race: bool = False) -> dict:
    """require_pre_race=True: Phase B(2026-08-04、data_provenance_caveat_2026_08_01対応)。
    verify_provenance.pyの判定(newspaper取得時刻 vs 実際の発走時刻)で"pre_race"と
    確認できたレースだけに母集団を絞る。取得時刻が記録されていない過去分
    ("unknown_no_result_yet"や記録自体が無いレース)は遡って判定できないため除外される
    ことに注意 — 2026-08-04より前に取得したレースはこの絞り込みで全滅する。
    既定はFalse(従来どおり絞り込まない)。"""
    results_dir = PROJECT_ROOT / "data" / "race_results" / "nar" / "2026"
    payouts_dir = PROJECT_ROOT / "data" / "payouts" / "nar" / "2026"
    dates = sorted(p.stem for p in results_dir.glob("2026*.csv") if (payouts_dir / p.name).exists())

    meta_rows = []
    for d in dates:
        df = pd.read_csv(results_dir / f"{d}.csv", dtype=str)
        names = df[["race_id", "race_name", "racecourse"]].drop_duplicates("race_id").copy()
        names["kaisai_date"] = d
        meta_rows.append(names)
    meta = pd.concat(meta_rows, ignore_index=True)
    meta = meta[~meta["race_name"].str.contains("新馬|未勝利", regex=True, na=False)]

    pre_race_ids = None
    if require_pre_race:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from verify_provenance import build_provenance_table  # noqa: E402

        prov = build_provenance_table()
        pre_race_ids = set(prov.loc[prov["verdict"] == "pre_race", "race_id"])

    races, skipped = [], []
    for _, row in meta.iterrows():
        if pre_race_ids is not None and row["race_id"] not in pre_race_ids:
            skipped.append((row["race_id"], "not_pre_race_verified"))
            continue
        path = newspaper_csv_path(row["race_id"])
        if not path.exists():
            skipped.append((row["race_id"], "no_newspaper"))
            continue
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
        if df.empty:
            skipped.append((row["race_id"], "empty"))
            continue
        df = drop_scratched(df)
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
        # require_pre_race=True の運用初期など、対象レースが0件になりうる
        # (Phase Bのmanifest記録が始まったばかりで pre_race 判定済みレースがまだ無い)。
        # 空のracesに対してクラッシュせず、空のデータセットとして返す。
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
        print(sizes.value_counts().sort_index().to_string())
    return data


def load(rebuild: bool = False, require_pre_race: bool = False) -> dict:
    cache_path = DATA_DIR / "nar_dataset_cache_pre_race.pkl" if require_pre_race else CACHE
    if cache_path.exists() and not rebuild:
        with cache_path.open("rb") as f:
            return pickle.load(f)
    data = build(require_pre_race=require_pre_race)
    with cache_path.open("wb") as f:
        pickle.dump(data, f)
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NAR再モデリング用データセットをビルドしてpickleにキャッシュする")
    parser.add_argument(
        "--require-pre-race", action="store_true",
        help="verify_provenance.pyでpre_raceと判定されたレースのみに母集団を絞る(Phase B)",
    )
    args = parser.parse_args()

    d = build(require_pre_race=args.require_pre_race)
    cache_path = DATA_DIR / "nar_dataset_cache_pre_race.pkl" if args.require_pre_race else CACHE
    with cache_path.open("wb") as f:
        pickle.dump(d, f)
    print(f"\nwrote {cache_path}")
