# -*- coding: utf-8 -*-
"""「33ラップ理論」をシグナル化するサイドカー(Part 2)。jra_signals.py は無改造。

型スコア(horse_lap33_type): `jra_history.HorseHistoryIndex.past_starts(horse_id,
before_date, n=N_LOOKBACK)`(race_date<before_dateの厳密カットオフ、リーク無し)で
過去走を取得し、各走のrace_idを`jra_lap33_by_race_2026_08_28.csv`
(jra_lap33_theory_2026_08_28.py の出力、race_id->33ラップ値)と結合。finish_pos<=3の
走(好走)の平均33ラップ − finish_pos>3の走(凡走)の平均33ラップを型スコアとする。

参照値(today_course_ref): jra_lap33_theory_table.py の理論PDF原本の値をそのまま使う
(Part1で再計算した実測値ではなく理論の原本値。「理論を公表通りに使うと当たるか」を
素直に検証するため、自己参照的循環を避ける設計)。

lap33_fit = 型スコア × 参照値、レース内minmax正規化(jra_signals._minmaxと同じ規約:
高いほど「今回のコース傾向に合っている」)。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
SCRATCHPAD = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
LAP33_BY_RACE_CSV = SCRATCHPAD / "jra_lap33_by_race_2026_08_28.csv"

sys.path.insert(0, str(LIB_DIR))
import jra_history as JH  # noqa: E402
import jra_signals as JS  # noqa: E402
from jra_lap33_theory_table import lookup as theory_lookup  # noqa: E402

N_LOOKBACK = 20  # 型スコア算出に使う過去走の本数上限(既存past{i}系の5走より広く取る)
MIN_GROUP_N = 2  # 好走・凡走のいずれかがこの本数未満なら型不明(0.0=中立)


def load_lap33_lookup() -> dict:
    """race_id -> 33ラップ値。jra_lap33_theory_2026_08_28.py を先に実行しておくこと。"""
    if not LAP33_BY_RACE_CSV.exists():
        raise FileNotFoundError(
            f"{LAP33_BY_RACE_CSV} が見つかりません。先に "
            "jra_lap33_theory_2026_08_28.py を実行してください。"
        )
    df = pd.read_csv(LAP33_BY_RACE_CSV, dtype={"race_id": str})
    return dict(zip(df["race_id"], df["lap33"]))


def load_race_surface_distance() -> dict:
    """race_id -> (racecourse, surface, distance_m)。jra_dataset母集団(246レース、既に
    決着済み)は全件race_resultsに存在する前提。"""
    frames = []
    for year_dir in sorted((PROJECT_ROOT / "data" / "race_results").glob("[0-9][0-9][0-9][0-9]")):
        for p in sorted(year_dir.glob("*.csv")):
            frames.append(pd.read_csv(
                p, dtype=str, usecols=["race_id", "surface", "distance_m", "racecourse"]))
    df = pd.concat(frames, ignore_index=True).drop_duplicates("race_id")
    out = {}
    for _, r in df.iterrows():
        try:
            out[r["race_id"]] = (r["racecourse"], r["surface"], int(float(r["distance_m"])))
        except (TypeError, ValueError):
            continue
    return out


def build_history_index() -> "JH.HorseHistoryIndex":
    results = JH.load_results()
    return JH.HorseHistoryIndex(results)


def horse_type_score(horse_id: str, iso_date: str, history_index: "JH.HorseHistoryIndex",
                     lap33_lookup: dict, n_lookback: int = N_LOOKBACK) -> float:
    """好走時(finish<=3)平均33ラップ - 凡走時(finish>3)平均33ラップ。標本不足(いずれかの
    群が MIN_GROUP_N 未満)なら0.0(型不明=中立、NaNではなく明示0とする理由はdocstring参照)。"""
    starts = history_index.past_starts(horse_id, iso_date, n=n_lookback)
    good, bad = [], []
    for s in starts:
        lap33 = lap33_lookup.get(s.race_id)
        if lap33 is None or (isinstance(lap33, float) and np.isnan(lap33)):
            continue
        finish = pd.to_numeric(s.finish_pos, errors="coerce")
        if pd.isna(finish):
            continue
        (good if finish <= 3 else bad).append(lap33)
    if len(good) < MIN_GROUP_N or len(bad) < MIN_GROUP_N:
        return 0.0
    return float(np.mean(good) - np.mean(bad))


def kaisai_date_to_iso(kaisai_date: str) -> str:
    return f"{kaisai_date[:4]}-{kaisai_date[4:6]}-{kaisai_date[6:8]}"


def lap33_fit_matrix(races: list, history_index: "JH.HorseHistoryIndex", lap33_lookup: dict,
                     race_meta: dict, n_lookback: int = N_LOOKBACK) -> dict:
    """races(jra_dataset形式)全体について、race_id -> {"type_score": Series, "lap33_fit": Series}
    を返す(1レースごとにレース内minmax正規化するため、他jra_signals信号と同じ「単一レース分の
    シグナル辞書」形式に合わせて呼び出し側でrace単位に扱う設計)。"""
    out = {}
    for r in races:
        iso_date = kaisai_date_to_iso(r["kaisai_date"])
        meta = race_meta.get(r["race_id"])
        ref = None if meta is None else theory_lookup(*meta)
        type_scores = []
        for hid in r["df"]["horse_id"].astype(str):
            type_scores.append(horse_type_score(hid, iso_date, history_index, lap33_lookup, n_lookback))
        ts = pd.Series(type_scores, index=r["df"].index)
        if ref is None:
            fit = pd.Series(np.nan, index=r["df"].index)
        else:
            fit = JS._minmax(ts * ref)
        out[r["race_id"]] = {"type_score": ts, "lap33_fit": fit, "course_ref": ref}
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "jra_model"))
    import jra_dataset as JD

    print("読み込み中...")
    lap33_lookup = load_lap33_lookup()
    race_meta = load_race_surface_distance()
    history_index = build_history_index()
    data = JD.load(rebuild=False)
    races = data["races"]
    print(f"lap33_lookup: {len(lap33_lookup)}レース  race_meta: {len(race_meta)}レース  "
          f"対象母集団: {len(races)}レース")

    fit = lap33_fit_matrix(races, history_index, lap33_lookup, race_meta)
    n_ref_available = sum(1 for v in fit.values() if v["course_ref"] is not None)
    n_type_nonzero = sum(int((v["type_score"] != 0.0).sum()) for v in fit.values())
    n_horses = sum(len(v["type_score"]) for v in fit.values())
    print(f"参照値(理論表)が引けたレース: {n_ref_available}/{len(races)}")
    print(f"型スコアが非ゼロ(=型判定できた)馬: {n_horses and n_type_nonzero}/{n_horses}"
          f"({n_type_nonzero / n_horses * 100:.1f}%)" if n_horses else "母集団0件")
