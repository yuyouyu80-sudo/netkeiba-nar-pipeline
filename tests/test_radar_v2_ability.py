"""着順予想 Part 2 / G1 の能力値(radar_v2_ability.py)のテスト(2026-10-08)。合成データのみ。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "jra_model" / "pedigree_reports"))
import radar_v2_ability as AB  # noqa: E402


def test_to_sec():
    s = AB.to_sec(pd.Series(["1:34.5", "59.8", "2:01.0", None, "abc"]))
    assert np.allclose(s[:3], [94.5, 59.8, 121.0]) and s[3:].isna().all()


def _runs():
    return pd.DataFrame({"horse_id": ["a"] * 3 + ["b"] * 2, "rest_days": [np.nan, 30, 60, np.nan, 14],
                         "age": [2, 2, 3, 3, 3], "date": pd.to_datetime(["2015-07-01", "2015-08-01", "2015-10-01", "2015-05-01", "2015-05-15"])})


def test_kalman_pre_race_state_excludes_own_figure():
    r = _runs()
    fig = pd.Series([1.0, 5.0, 0.0, -1.0, 2.0])
    pm = pd.Series([0.3, 0.3, 0.3, -0.2, -0.2])
    par = {"P0": 0.4, "q0": 0.01, "q1": 0.01, "R": 0.24}
    st = AB.kalman(r, fig, pm, par, {})
    assert st.loc[0, "ab_mu"] == 0.3 and st.loc[3, "ab_mu"] == -0.2       # 初出走は事前の平均
    assert st["ab_n"].tolist() == [0, 1, 2, 0, 1]
    fig2 = fig.copy()
    fig2[1] = 999.0                                                        # 2走目の fig を変えても 2走目の発走前の値は変わらない
    st2 = AB.kalman(r, fig2, pm, par, {})
    assert np.isclose(st.loc[1, "ab_mu"], st2.loc[1, "ab_mu"]) and not np.isclose(st.loc[2, "ab_mu"], st2.loc[2, "ab_mu"])
    assert st.loc[1, "ab_sd"] < np.sqrt(0.4)                               # 1走観測すると不確かさが減る


def test_ewma_pre_race_and_prior():
    r = _runs()
    fig = pd.Series([1.0, 5.0, 0.0, -1.0, 2.0])
    pm = pd.Series([0.3, 0.3, 0.3, -0.2, -0.2])
    e = AB.ewma(r, fig, pm, 0.5, 1.0)
    assert e[0] == 0.3 and e[3] == -0.2
    assert np.isclose(e[1], (1 * 1.0 + 1 * 0.3) / 2)                       # 1走後: 重み1の本人 + k=1 の事前
