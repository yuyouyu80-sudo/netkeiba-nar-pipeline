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

後方互換拡張(2026-08-25、K=3〜8スイープ対応):
  - build(min_field_size=5): 引数化。デフォルト値は旧来の固定値5と同じなので、
    既存呼び出し(nar_search_bottom_2026_08_24.py等)は無改造で従来どおり動く。
  - df["finish_pos_numeric"] を追加出力(label_bottomはそのまま残す)。DNF正例
    (中止/失格)は999.0センチネル(実測でfinish_pos実値1〜16と衝突しないことを確認済み)、
    DNF除外/欠損はNaN、それ以外は数値。K=3〜8いずれのしきい値でも同じdf1回のbuildで
    ラベルを再構成できるようにするためのもの。
  - キャッシュにschema_versionを埋め込み、load()はバージョン不一致(旧キャッシュ含む)なら
    自動的にbuild()で再生成する(ファイル名は変えない)。
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

# キャッシュペイロードのスキーマバージョン。finish_pos_numeric追加(2026-08-25)でv2に。
SCHEMA_VERSION = 2

# DNF正例(中止/失格)のfinish_pos_numericセンチネル値。実測(全16805行走査)でfinish_pos
# 実値(1〜16)と衝突しないことを確認済み。
DNF_SENTINEL = 999.0

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


def _finish_pos_numeric_row(raw) -> float:
    """1頭分の finish_pos 生値から数値着順を返す(K=3〜8スイープで任意のしきい値の
    ラベルを再構成するための土台)。DNF正例(中止/失格)はDNF_SENTINEL、DNF除外
    (取消/除外)・欠損はNaN、それ以外は数値そのもの。"""
    if pd.isna(raw) or str(raw).strip() == "":
        return np.nan
    s = str(raw).strip()
    if s in DNF_EXCLUDE:
        return np.nan
    if s in DNF_POSITIVE:
        return DNF_SENTINEL
    try:
        return float(int(s))
    except ValueError:
        return np.nan


def build(verbose: bool = True, min_field_size: int = 5) -> dict:
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
        merged["finish_pos_numeric"] = merged["finish_pos"].map(_finish_pos_numeric_row)
        merged = merged[merged["label_bottom"].notna()].reset_index(drop=True)
        if len(merged) < min_field_size:
            # min_field_size未満(既定5)まで頭数が減ったレース。field_size>=min_field_size
            # フィルタをここで機械的に適用する。
            skipped.append((r["race_id"], f"field_size_lt_{min_field_size}_after_label_cleanup"))
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
        "schema_version": SCHEMA_VERSION,
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
            data = pickle.load(f)
        if data.get("schema_version") == SCHEMA_VERSION:
            return data
        # 旧形式(schema_versionキー無し、または旧バージョン)のキャッシュは自動再生成する。
        # ファイル名は変えない(_v2.pklのような枝分かれを避ける)。
    data = build(verbose=False)
    with CACHE.open("wb") as f:
        pickle.dump(data, f)
    return data


if __name__ == "__main__":
    d = build(verbose=True)
    with CACHE.open("wb") as f:
        pickle.dump(d, f)
    print(f"\nwrote {CACHE}")
