# -*- coding: utf-8 -*-
"""血統レーダー v2 の R3: レースの基準(このレースで求められる脚質・キレ)を作る(2026-10-06)。

事前登録 RADAR_V2_PREREG_2026_10_06.md §5。
- 各過去レースのリフト = 上位3着の実際の値の平均 − 出走馬の実際の値の平均(値は Y年のレースなら Y−1年末までで推定した
  係数による脚質・キレの残差、radar_v2_aptitude.run_values)。
- 同等レース(テスト日より前のレースのみ): L1 コース(内外回り込み)×クラス×開催日目 → 15件未満なら L2 コース×クラス →
  10件未満なら L3 芝ダ×距離帯×クラス群 → L4 芝ダ。
- 基準 = 経験ベイズ縮約((n·平均 + k·親の平均)/(n + k)、親=1段上の階層の平均)から、その日より前の全レース平均を引いたもの。
  k = レース間分散 / 群間分散(L2、2011〜2020年のレースで推定して固定)。
- ペース別(速い/平均/遅い)も同じ手順で、過去レースをそのペース区分に限って作る(表示用、評価の主には使わない)。
- 出力: radar_v2/race_req.parquet(2013〜2026年の全レース)、radar_v2/race_req_meta.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import radar_v2_common as C  # noqa: E402
import radar_v2_aptitude as A  # noqa: E402

OUT = C.OUT_DIR / "race_req.parquet"
META = C.OUT_DIR / "race_req_meta.json"
AXES = ["front", "kire"]
MIN_L1, MIN_L2, MIN_L3 = 15, 10, 10
K_FIT_END = 2020


def class_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    co = df["class_ord"]
    df["class_key"] = np.where(df["is_debut_race"], "新馬", co.astype("Int64").astype(str))
    df["class_group"] = np.where(df["is_debut_race"], "新馬", np.where(co == 0, "未勝利", np.where(co == 1, "1勝",
                                 np.where(co <= 3, "2-3勝", "OP以上"))))
    df["day_seg"] = df["race_id"].str[8:10]
    df["L1"] = df["course_key_io"] + "|" + df["class_key"] + "|" + df["day_seg"]
    df["L2"] = df["course_key_io"] + "|" + df["class_key"]
    df["L3"] = df["surface"] + "|" + df["dbucket"] + "|" + df["class_group"]
    df["L4"] = df["surface"]
    return df


def race_lifts(v: pd.DataFrame) -> pd.DataFrame:
    top = v["pos"] <= 3
    out = {}
    for a in AXES:
        out[f"lift_{a}"] = v[top].groupby("race_id")[a].mean() - v.groupby("race_id")[a].mean()
    first = v.drop_duplicates("race_id").set_index("race_id")
    cols = ["date", "year", "surface", "dbucket", "course_key_io", "class_key", "class_group", "day_seg",
            "L1", "L2", "L3", "L4", "pace_tier", "turn_type"]
    return pd.DataFrame(out).join(first[cols])


def prior_stats(R: pd.DataFrame, key: str, val: str) -> pd.DataFrame:
    """各レースについて、同じ key で日付が厳密に前のレースの val の件数・合計(当日のレースは含めない)。"""
    d = R.dropna(subset=[val]).groupby([key, "date"])[val].agg(["sum", "count"]).sort_index()
    cs = d.groupby(level=0).cumsum() - d  # その日付より前の累積
    cs.columns = [f"{key}_sum", f"{key}_n"]
    return R.join(cs, on=[key, "date"])


def estimate_k(R: pd.DataFrame, val: str) -> float:
    d = R[(R["year"] <= K_FIT_END)].dropna(subset=[val])
    g = d.groupby("L2")[val]
    stats = pd.DataFrame({"m": g.mean(), "n": g.count(), "v": g.var()})
    stats = stats[stats["n"] >= 10]
    within = float(np.average(stats["v"], weights=stats["n"] - 1))
    tau2 = float(np.var(stats["m"]) - np.mean(within / stats["n"]))
    return within / max(tau2, 1e-12)


def build_req(R: pd.DataFrame, val: str, k: float) -> pd.Series:
    X = R.copy()
    for L in ["L1", "L2", "L3", "L4"]:
        X = prior_stats(X, L, val)
    # 全体(その日より前の全レース)の平均
    g = X.dropna(subset=[val]).groupby("date")[val].agg(["sum", "count"]).sort_index()
    cs = g.cumsum() - g
    X = X.join(cs.rename(columns={"sum": "all_sum", "count": "all_n"}), on="date")
    mean = lambda L: X[f"{L}_sum"] / X[f"{L}_n"]  # noqa: E731
    m4 = mean("L4").fillna(X["all_sum"] / X["all_n"])
    sh3 = (X["L3_sum"].fillna(0) + k * m4) / (X["L3_n"].fillna(0) + k)
    sh2 = (X["L2_sum"].fillna(0) + k * sh3) / (X["L2_n"].fillna(0) + k)
    sh1 = (X["L1_sum"].fillna(0) + k * sh2) / (X["L1_n"].fillna(0) + k)
    n1, n2, n3 = X["L1_n"].fillna(0), X["L2_n"].fillna(0), X["L3_n"].fillna(0)
    req = np.where(n1 >= MIN_L1, sh1, np.where(n2 >= MIN_L2, sh2, np.where(n3 >= MIN_L3, sh3, m4)))
    rule = np.where(n1 >= MIN_L1, "L1", np.where(n2 >= MIN_L2, "L2", np.where(n3 >= MIN_L3, "L3", "L4")))
    nm = np.where(n1 >= MIN_L1, n1, np.where(n2 >= MIN_L2, n2, np.where(n3 >= MIN_L3, n3, X["L4_n"].fillna(0))))
    allm = X["all_sum"] / X["all_n"]
    return pd.DataFrame({"req": req - allm, "rule": rule, "n_matched": nm}, index=X.index)


def main():
    t0 = time.time()
    runs = pd.read_parquet(A.RUNS)
    th = A.fixed_thresholds(runs)
    runs = class_keys(A.add_tiers(runs, th))
    lifts = []
    # Y年のレースの実際の値は、Y−1年末までで推定した係数で計算する(2011〜2012年は2011〜2012年の係数で記述的に)
    for Y in range(2011, 2027):
        fit = runs[runs["year"] <= max(Y - 1, 2012)]
        app = runs[runs["year"] == Y]
        lifts.append(race_lifts(A.run_values(fit, app)))
        print(time.strftime("%H:%M:%S"), "lifts", Y, flush=True)
    R = pd.concat(lifts).sort_values("date")
    meta = {"k": {}, "rule_counts": {}}
    out = R[["date", "year", "surface", "course_key_io", "class_key", "pace_tier"]].copy()
    for a in AXES:
        val = f"lift_{a}"
        k = estimate_k(R, val)
        meta["k"][a] = round(k, 2)
        rq = build_req(R, val, k)
        out[f"req_{a}"] = rq["req"]
        out[f"rule_{a}"] = rq["rule"]
        out[f"n_matched_{a}"] = rq["n_matched"]
        out[val] = R[val]
        # ペース別(過去レースをそのペース区分に限る)
        for tier in ["速い", "平均", "遅い"]:
            Rt = R.copy()
            Rt.loc[Rt["pace_tier"] != tier, val] = np.nan
            out[f"req_{a}_{tier}"] = build_req(Rt, val, k)["req"]
        meta["rule_counts"][a] = out.loc[out["year"] >= 2013, f"rule_{a}"].value_counts().to_dict()
        # 標準化の基準: 2018〜2020年のレースの基準のSD(R4で req/sd として使う)
        meta[f"sd_req_{a}_2018_2020"] = float(out.loc[out["year"].between(2018, 2020), f"req_{a}"].std())
    out = out[out["year"] >= 2013]
    out.to_parquet(OUT)
    meta["thresholds"] = th
    meta["n_races"] = int(len(out))
    meta["elapsed_sec"] = round(time.time() - t0, 1)
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
