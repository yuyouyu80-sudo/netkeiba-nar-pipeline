# -*- coding: utf-8 -*-
"""NAR「5位以下(4着以内に入らない)」予測モデル用データセット層(サイドカー)。

既存 nar_dataset.py は一切変更しない。このモジュールは nar_dataset.build() を
呼ぶだけで、以下を追加する:
  - data/race_results/nar/2026/{date}.csv の finish_pos を (race_id, umaban) キーで結合
  - label_bottom(4着以内に入らなかったか、1=5位以下・DNF/0=4着以内)の生成
  - newspaperのdata_*系スキーマが2026-07-25で完全分断していることに対応した
    Track A(全期間)/Track B(2026-07-25以降のみ)の目印付与

共有キャッシュ(nar_dataset_cache.pkl)は汚染しない: nar_dataset.load()/load(rebuild=True)は
使わず、nar_dataset.build(verbose=False)を直接呼び、結果は本モジュール専用の
nar_bottom_dataset_cache.pkl にのみ保存する。
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nar_pipeline"
CACHE = DATA_DIR / "nar_bottom_dataset_cache.pkl"
RESULTS_DIR = PROJECT_ROOT / "data" / "race_results" / "nar" / "2026"

sys.path.insert(0, str(LIB_DIR))
import nar_dataset  # noqa: E402  既存モジュール、無改造・呼ぶだけ

# newspaperのdata_*系スキーマ(data_distance/data_course/data_others/data_cushion/
# data_baba_water)が導入された日付。実測(2026-08-24)でこれ以前は0%充足・以降は100%充足の
# 完全分断であることを確認済み。
SCHEMA_SPLIT_DATE = "20260725"

# race_results の finish_pos に混在するDNF系1文字コード(実測で確認、"取消"等の全角表記ではない)。
DNF_EXCLUDE = {"取", "除"}   # 取消・除外 → 発走前スクラッチ → 母集団から除外(念のための残存対策)
DNF_POSITIVE = {"中", "失"}  # 中止・失格 → 発走はしたが4着以内という結果を残せなかった → 正例


def _load_finish_pos() -> pd.DataFrame:
    """data/race_results/nar/2026/*.csv から (race_id, umaban, finish_pos) を集める。"""
    frames = []
    for path in sorted(RESULTS_DIR.glob("2026*.csv")):
        df = pd.read_csv(path, dtype=str, usecols=["race_id", "umaban", "finish_pos"])
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["race_id", "umaban", "finish_pos"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["race_id", "umaban"])


def _label_row(raw) -> float:
    """1頭分の finish_pos 生値から label_bottom を返す。
    1.0 = 5位以下(またはDNF系の中止/失格)、0.0 = 4着以内、NaN = 母集団から除外すべき行。"""
    if pd.isna(raw) or str(raw).strip() == "":
        return np.nan  # 結合漏れ・未記録(実測77行、うち母集団内42行) → 除外
    s = str(raw).strip()
    if s in DNF_EXCLUDE:
        return np.nan  # 取消・除外(drop_scratched後も稀に残存、実測 取3件・除6件) → 除外
    if s in DNF_POSITIVE:
        return 1.0
    try:
        return 1.0 if int(s) >= 5 else 0.0
    except ValueError:
        return np.nan  # 未知コード、安全側で除外


def build(verbose: bool = True) -> dict:
    base = nar_dataset.build(verbose=False)  # 母集団定義(新馬+未勝利除外等)は既存のまま流用
    finish = _load_finish_pos()

    races = []
    skipped = list(base["skipped"])
    for r in base["races"]:
        df = r["df"].copy()
        df["umaban"] = df["umaban"].astype(int)
        sub = finish[finish["race_id"] == r["race_id"]][["umaban", "finish_pos"]].copy()
        sub["umaban"] = sub["umaban"].astype(int)
        merged = df.merge(sub, on="umaban", how="left")
        if len(merged) != len(df):
            skipped.append((r["race_id"], "finish_pos_merge_row_mismatch"))
            continue
        merged["label_bottom"] = merged["finish_pos"].map(_label_row)
        merged = merged[merged["label_bottom"].notna()].reset_index(drop=True)
        if len(merged) < 5:
            # 4頭立て以下(5位以下が原理的に存在しない)、またはラベル除外後に頭数が
            # 5未満まで減ったレース。field_size>=5フィルタをここで機械的に適用する。
            skipped.append((r["race_id"], "field_size_lt_5_after_label_cleanup"))
            continue
        races.append({
            "race_id": r["race_id"], "kaisai_date": r["kaisai_date"],
            "racecourse": r["racecourse"], "race_name": r["race_name"],
            "df": merged, "field_size": len(merged),
            "track_b": r["kaisai_date"] >= SCHEMA_SPLIT_DATE,
        })

    data = {
        "races": races,
        "dates": sorted({r["kaisai_date"] for r in races}),
        "skipped": skipped,
    }
    if verbose:
        n_a = len(races)
        n_b = sum(1 for r in races if r["track_b"])
        print(f"Track A(全期間) races: {n_a}")
        print(f"Track B(2026-07-25以降) races: {n_b}")
        print(f"skipped: {len(skipped)}")
        sizes = pd.Series([r["field_size"] for r in races])
        print(f"field size: min={sizes.min()} median={sizes.median()} max={sizes.max()} mean={sizes.mean():.1f}")
        pos_rate = pd.Series([r["df"]["label_bottom"].mean() for r in races]).mean()
        print(f"label_bottom=1 平均割合(レース内平均): {pos_rate * 100:.1f}%")
    return data


def track_a(data: dict) -> list:
    """全期間(N全体)。"""
    return data["races"]


def track_b(data: dict) -> list:
    """2026-07-25以降のみ(data_*系シグナルが100%充足するサブ集団)。"""
    return [r for r in data["races"] if r["track_b"]]


def load(rebuild: bool = False) -> dict:
    if CACHE.exists() and not rebuild:
        with CACHE.open("rb") as f:
            return pickle.load(f)
    data = build(verbose=False)
    with CACHE.open("wb") as f:
        pickle.dump(data, f)
    return data


if __name__ == "__main__":
    d = build(verbose=True)
    with CACHE.open("wb") as f:
        pickle.dump(d, f)
    print(f"\nwrote {CACHE}")
