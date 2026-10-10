# -*- coding: utf-8 -*-
"""finish_v4 F2(ability_v2)・F0-1(nonfinish)の小さな検査。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "jra_model" / "finish_v4"))
sys.path.insert(0, str(ROOT / "scripts" / "jra_model" / "pedigree_reports"))
import ability_v2 as V  # noqa: E402
import radar_v2_ability as AB  # noqa: E402


def _toy(n_h=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_h):
        d = pd.Timestamp("2015-06-01") + pd.Timedelta(days=int(rng.integers(0, 60)))
        for k in range(int(rng.integers(1, 8))):
            rows.append({"horse_id": f"h{h:03d}", "date": d, "age": 2 + k // 3, "rest_days": np.nan if k == 0 else 30.0,
                         "surface": "芝" if rng.random() < 0.6 else "ダ", "fig": rng.normal(0, 0.5) if rng.random() < 0.9 else np.nan,
                         "nonfinish": False, "ped_S_horse_id": f"s{h % 4}", "dam_id": f"d{h % 9}", "logd": np.log(1200 + 200 * k)})
            d = d + pd.Timedelta(days=30)
    r = pd.DataFrame(rows).sort_values(["horse_id", "date"]).reset_index(drop=True)
    r["is_turf"] = (r["surface"] == "芝").astype(int)
    r["month0"] = r["date"].dt.to_period("M").dt.start_time
    r["year"] = r["date"].dt.year
    return r


def test_kalman2_matches_1d_when_fully_correlated():
    r = _toy()
    par = {"P0": 0.4, "q0": 0.01, "q1": 0.01, "R": 0.24, "rho0": 1.0, "rhoq": 1.0}
    gr = {"2前": 0.3, "2後": 0.2, "3前": 0.1}
    pm = np.linspace(-0.2, 0.2, len(r))
    s2 = V.kalman2(r, r["fig"], pm, pm, par, gr)
    s1 = AB.kalman(r, r["fig"], pd.Series(pm, index=r.index), par, gr)
    assert np.allclose(s2["k_mu"], s1["ab_mu"])
    assert np.allclose(s2["k_sd"], s1["ab_sd"])
    m = s1["ab_ll"].notna()
    assert np.allclose(s2.loc[m, "k_ll"], s1.loc[m, "ab_ll"])


def test_kalman2_nonfinish_row_does_not_move_state():
    r = _toy()
    par = {"P0": 0.4, "q0": 0.01, "q1": 0.01, "R": 0.24, "rho0": 0.7, "rhoq": 0.8}
    gr = {"2前": 0.3}
    pm = np.zeros(len(r))
    base = V.kalman2(r, r["fig"], pm, pm, par, gr)
    # 馬 h000 の2走目と3走目の間に中止の行を足す
    i = r.index[r["horse_id"] == r["horse_id"].value_counts().index[0]][1]
    g = r.loc[[i]].assign(nonfinish=True, fig=np.nan, date=r.loc[i, "date"] + pd.Timedelta(days=10))
    r2 = pd.concat([r, g]).sort_values(["horse_id", "date"], kind="stable").reset_index(drop=True)
    pm2 = np.zeros(len(r2))
    st = V.kalman2(r2, r2["fig"], pm2, pm2, par, gr)
    fin = ~r2["nonfinish"].to_numpy()
    assert np.allclose(st.loc[fin, "k_mu"].to_numpy(), base["k_mu"].to_numpy())
    assert np.allclose(st.loc[fin, "k_sd"].to_numpy(), base["k_sd"].to_numpy())


def test_group_month_value_excludes_own_and_future():
    r = _toy(n_h=40, seed=1)
    resid = r["fig"]
    ev = V.horse_events(r, resid)
    v = V.group_month_value(ev, "ped_S_horse_id", "x", r, k=0.0001)
    # 手計算: ある行について、同じ父の他の馬の「その月より前の走」の平均を重み min(n,5) で
    i = r.index[30]
    row = r.loc[i]
    num = den = 0.0
    for h, g in r[(r["ped_S_horse_id"] == row["ped_S_horse_id"]) & (r["horse_id"] != row["horse_id"])].groupby("horse_id"):
        past = g[(g["date"] < row["month0"]) & g["fig"].notna()]
        if len(past):
            w = min(len(past), 5)
            num += w * past["fig"].mean()
            den += w
    exp = num / (den + 0.0001) if den else 0.0
    assert abs(v[i] - exp) < 1e-6


def test_dist_features_use_only_past_finishes():
    r = _toy(n_h=5, seed=2)
    f = V.dist_features(r)
    for h, g in r.groupby("horse_id"):
        idx = g.index
        assert f.loc[idx[0], "dist_dev"] == 0.0
        if len(idx) > 2:
            exp = g.loc[idx[2], "logd"] - g.loc[idx[:2], "logd"].mean()
            assert abs(f.loc[idx[2], "dist_dev"] - exp) < 1e-12
