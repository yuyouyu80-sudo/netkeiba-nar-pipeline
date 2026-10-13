# -*- coding: utf-8 -*-
"""finish_v4 G* の発走前の特徴量(2026-10-13、事前登録 §14)。オッズを読むモジュールは import しない(テストで確認)。

- 近5走の特徴(G3 の7系統): 別の会話の analysis/past5_unused_2026_10_12/p1_build.py の load・build をそのまま使う。
  走の表(race_results)の、その日より前のファイルに、その日の出走表の行(結果の列は空)を1つの CSV として足して読ませる。
  出走表の行は「着順なし」になるので、その行の特徴量は前の走だけから作られる(features.parquet と同じ定義)。
- 過去の人気(案B の人気の推定): 走の表の各走の人気・頭数を、馬ごとに日付順でずらしたもの(当日の人気は使わない)。
- 人気の推定 q̂ と組の推定人気 p̂: live_params_{年}_GSTAR.json の係数(gstar.py --fit-live で作る)を当てるだけ。
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import f0_dev_utils as F0  # noqa: E402
import g3_past5 as G3  # noqa: E402
import pl_ext as P  # noqa: E402

ROOT = F0.C.PROJECT_ROOT
sys.path.insert(0, str(ROOT / "analysis" / "past5_unused_2026_10_12"))
POP = ["lpop1", "lpop2", "lpop3", "lpop_mean3", "lpop_best3", "lpop_trend", "has_pop1", "A_pop_vs_fin3"]
POOL = 60


def pop_features(F: pd.DataFrame | None = None) -> pd.DataFrame:
    """走ごとの今回の人気(pop)・頭数を馬ごとに日付順でずらし、過去の人気だけを作る。F を省くと features.parquet。"""
    if F is None:
        F = pd.read_parquet(G3.P5, columns=["race_id", "horse_id", "date", "pop", "field", "A_pop_vs_fin3"])
    F = F[["race_id", "horse_id", "date", "pop", "field", "A_pop_vs_fin3"]]
    F = F.drop_duplicates(["race_id", "horse_id"]).sort_values(["horse_id", "date"]).reset_index(drop=True)
    F["lp"] = -np.log(F["pop"].astype(float) / F["field"].astype(float))    # 大きいほど上位人気(相対)
    g = F.groupby("horse_id")["lp"]
    L = pd.concat([g.shift(k).rename(k) for k in (1, 2, 3)], axis=1)
    out = pd.DataFrame({"race_id": F["race_id"], "horse_id": F["horse_id"]})
    for k in (1, 2, 3):
        out[f"lpop{k}"] = L[k]
    out["lpop_mean3"] = L.mean(axis=1)
    out["lpop_best3"] = L.max(axis=1)
    out["lpop_trend"] = L[1] - L[2]
    out["has_pop1"] = L[1].notna().astype(float)
    out["A_pop_vs_fin3"] = F["A_pop_vs_fin3"]
    return out.set_index(["race_id", "horse_id"]).fillna(0.0)


class _Res:
    """p1_build.RES の代わり: D より前の race_results のファイルと、出走表の行の CSV を返す。"""

    def __init__(self, files):
        self.files = files

    def glob(self, pat):
        return list(self.files)


def card_csv(pz: pd.DataFrame, path: Path):
    """その日の出走表の行(radar_v2_live.pseudo_rows)を race_results の列の形で書く(結果の列は空)。"""
    d = pd.DataFrame({"race_id": pz["race_id"], "race_name": pz["race_name"], "surface": pz["surface"],
                      "distance_m": pz["distance_m"], "going": "", "race_date": pd.to_datetime(pz["date"]).dt.strftime("%Y-%m-%d"),
                      "racecourse": pz["racecourse"], "finish_pos": "", "umaban": pz["umaban"].astype(str), "horse_id": pz["horse_id"],
                      "kinryo": "", "jockey_id": "", "time": "", "passing_order": "", "last_3f": "", "odds_final": "",
                      "popularity": "", "weight": ""})
    d.to_csv(path, index=False, encoding="utf-8")


def past5_live(D: pd.Timestamp, pz: pd.DataFrame) -> pd.DataFrame:
    """出走表の行の近5走の特徴(G3.P5_COLS)と過去の人気(POP)。index は (race_id, horse_id)。"""
    import p1_build as PB
    files = [p for p in sorted((ROOT / "data" / "race_results").glob("20*/*.csv")) if p.stem[:8].isdigit() and p.stem[:8] < D.strftime("%Y%m%d")]
    with tempfile.TemporaryDirectory() as td:
        cp = Path(td) / f"{D.strftime('%Y%m%d')}_card.csv"
        card_csv(pz, cp)
        old = PB.RES
        PB.RES = _Res(files + [cp])
        try:
            d = PB.load()
        finally:
            PB.RES = old
    F = PB.build(d)
    new = set(pz["race_id"])
    pf = pop_features(F)
    F = F[F["race_id"].isin(new)].drop_duplicates(["race_id", "horse_id"]).set_index(["race_id", "horse_id"])
    return F[G3.P5_COLS].join(pf, how="left")


def add_gstar(X: pd.DataFrame, P5: pd.DataFrame) -> pd.DataFrame:
    """F7 の特徴量の表 X(finish_v4_live.build の出力)に、G3 の特徴量と過去の人気を足す。"""
    X = G3.add_g3(X, P5[G3.P5_COLS], G3.course_table())
    X = X.join(P5[POP], on=["race_id", "horse_id"])
    for c in POP:
        X[c] = X[c].fillna(0.0)
    return X


def qhat(X: pd.DataFrame, pop: dict) -> np.ndarray:
    """推定人気(レース内の softmax)。pop: {"features", "mu", "sd", "b"}。"""
    Z = (X[pop["features"]].fillna(0.0).to_numpy(float) - np.asarray(pop["mu"])) / np.asarray(pop["sd"])
    z = Z @ np.asarray(pop["b"])
    s = pd.Series(z, index=X.index)
    e = np.exp(z - s.groupby(X["race_id"]).transform("max").to_numpy())
    return e / pd.Series(e, index=X.index).groupby(X["race_id"]).transform("sum").to_numpy()


def tickets_b(umaban, u, lam, q, gamma: float, mlam, pool: int = POOL, n: int = 20) -> dict:
    """案B(w1.0_topPOOL): モデルの上位 pool 組の中から log p − log p̂ の大きい順に n 点。"""
    um = np.asarray(umaban, int)
    cc, pf = P.triple_probs(np.asarray(u, float), "pl", tuple(lam))
    ce, pe = P.triple_probs(gamma * np.log(np.asarray(q, float)), "pl", tuple(mlam))
    assert (cc == ce).all()
    top = np.argsort(-pf, kind="stable")[:pool]
    s = np.full(len(pf), -np.inf)
    s[top] = np.log(pf[top]) - np.log(pe[top])
    ii = np.argsort(-s, kind="stable")[:n]
    return {"tickets": [sorted(int(x) for x in um[cc[i]]) for i in ii], "p_ticket": [round(float(pf[i]), 6) for i in ii],
            "p_hat_ticket": [round(float(pe[i]), 6) for i in ii], "p_covered": round(float(pf[ii].sum()), 4)}
