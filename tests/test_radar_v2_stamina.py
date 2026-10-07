"""血統レーダー v2 のスタミナの3軸(radar_v2_stamina.py、事前登録 追補7)のテスト(2026-10-07)。合成データのみ。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "jra_model" / "pedigree_reports"))
import radar_v2_eval as E  # noqa: E402
import radar_v2_stamina as S  # noqa: E402

B = list(S.BUCKET_ORDER)


def test_horse_slopes_recovers_slope_and_drops_thin_horses():
    rng = np.random.default_rng(0)
    rows = []
    for h, slope in [("h1", 0.2), ("h2", -0.1)]:
        for x in np.linspace(-2, 2, 9):
            rows.append({"horse_id": h, "x": x, "y": 0.3 + slope * x + rng.normal(0, 1e-6)})
    rows += [{"horse_id": "h3", "x": x, "y": x} for x in (0.0, 1.0)]            # 2走だけ → 除外
    rows += [{"horse_id": "h4", "x": x, "y": x} for x in (0.0, 0.1, 0.2, 0.3)]  # Σ(x−x̄)² < 1 → 除外
    out = S.horse_slopes(pd.DataFrame(rows), "y", "x")
    assert set(out.index) == {"h1", "h2"}
    assert np.isclose(out.loc["h1", "x"], 0.2, atol=1e-4) and np.isclose(out.loc["h2", "x"], -0.1, atol=1e-4)
    assert np.isclose(out.loc["h1", "n"], np.sum((np.linspace(-2, 2, 9)) ** 2))
    clipped = S.horse_slopes(pd.DataFrame(rows), "y", "x", clip=[-0.05, 0.05])
    assert clipped["x"].max() <= 0.05 and clipped["x"].min() >= -0.05


def _runs(seq):
    """seq = [(距離, 距離帯の番号, perf_adj)]、1頭の時系列。"""
    rows = []
    prev = None
    for i, (dist, b, pa) in enumerate(seq):
        rows.append({"race_id": f"r{i}", "horse_id": "h", "logd": np.log(dist), "prev_logd": prev, "dbucket": B[b],
                     "perf_adj": pa, "career": i, "age_band": "3", "surface": "芝", "front_raw": 0.5})
        prev = np.log(dist)
    return pd.DataFrame(rows)


def test_extension_flags():
    v = _runs([(1400, 0, 0.0), (1600, 1, 0.0), (1600, 1, 0.0), (1500, 1, 0.0), (1650, 1, 0.0), (1800, 1, 0.0), (1801, 2, 0.0)])
    R = pd.DataFrame({"pace_hi": 0.0, "c_z": 0.0}, index=v["race_id"])
    v = S.add_run_cols(v, R)
    # 1400→1600: log比 0.134 ≥ 0.10 → 延長 / 1600→1600 同距離 / 1600→1500 短縮(0.065 < 0.10 で距離帯も同じ → 短縮ではない)
    # 1500→1650: log比 0.095 < 0.10、同じ距離帯 → 延長ではない / 1650→1800: 0.087 → 延長ではない / 1800→1801: 距離帯が上がる → 延長
    assert v["ext"].tolist() == [False, True, False, False, False, False, True]
    assert v["short"].tolist() == [False, False, False, False, False, False, False]
    assert np.isclose(v.loc[1, "ext_amt"], np.log(1600 / 1400)) and v.loc[0, "ext_amt"] == 0
    big = S.add_run_cols(_runs([(1000, 0, 0.0), (2400, 3, 0.0)]), pd.DataFrame({"pace_hi": 0.0, "c_z": 0.0}, index=["r0", "r1"]))
    assert np.isclose(big.loc[1, "ext_amt"], S.EXT_CAP)   # 延長の幅は log 1.5 で切る


def test_horse_ext_uses_only_within_bucket_contrast():
    v = pd.DataFrame({"horse_id": "h", "dbucket": [B[1], B[1], B[1], B[2], B[2]], "ext": [True, False, False, True, True],
                      "r_B": [0.3, 0.1, 0.0, 0.9, 0.8]})
    out = S.horse_ext(v)
    # マイル帯だけが両方を持つ: 0.3 − 0.05 = 0.25、重み 1·2/3。中距離帯は延長だけなので使わない
    assert np.isclose(out.loc["h", "x"], 0.25) and np.isclose(out.loc["h", "n"], 2 / 3)


def _race_frame(seed=0, years=range(2011, 2023)):
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        for i in range(60):
            s = "芝" if i % 2 else "ダ"
            b = B[i % 4]
            dist = [1200, 1600, 2000, 2400][i % 4]
            rows.append({"race_id": f"{y}{i:04d}", "year": y, "date": pd.Timestamp(f"{y}-01-01") + pd.Timedelta(days=i * 5),
                         "surface": s, "dbucket": b, "distance": dist, "class_group": ["未勝利", "1勝"][i % 2 == 0 and 1 or 0],
                         "pace": rng.normal(0, 0.5), "l3_med": 36 + dist / 1000 + rng.normal(0, 0.5)})
    return pd.DataFrame(rows).set_index("race_id")


def test_fixed_quantities_ignore_post_2020_rows():
    R = _race_frame()
    fx1 = S.fit_race_quantities(R)
    R2 = R.copy()
    late = R2["year"] > S.FIX_END
    R2.loc[late, "pace"] = 1e6
    R2.loc[late, "l3_med"] = -1e6
    fx2 = S.fit_race_quantities(R2)
    assert fx1 == fx2


def test_adjustment_with_zero_coefficients_is_identity():
    v = pd.DataFrame({"perf_adj": [0.1, -0.2], "front_raw": [0.9, np.nan], "pace_hi": [1.5, -1.0], "surface": ["芝", "ダ"]})
    out = S.adj_apply(v, "pace_hi", {"芝": [0.0, 0.0], "ダ": [0.5, 0.5]})
    assert np.allclose(out, v["perf_adj"])   # 芝は係数0、ダは前半位置が無い(f=0)ので調整なし


def test_q_prior_uses_only_earlier_races():
    import radar_v2_reference as RF
    R = _race_frame(years=range(2015, 2018))
    R["course_key_io"] = R["surface"] + "|" + R["distance"].astype(str)
    R["class_key"] = R["class_group"]
    R["day_seg"] = "01"
    R["L1"] = R["course_key_io"] + "|" + R["class_key"] + "|01"
    R["L2"] = R["course_key_io"] + "|" + R["class_key"]
    R["L3"] = R["surface"] + "|" + R["dbucket"] + "|" + R["class_group"]
    R["L4"] = R["surface"]
    R["v"] = np.random.default_rng(1).normal(size=len(R))
    a = RF.build_req(R, "v", 2.0)["req"]
    last = R["date"].max()
    R2 = R.copy()
    R2.loc[R2["date"] == last, "v"] = 1e6   # その日のレースの値を変えても、その日の基準は変わらない
    b = RF.build_req(R2, "v", 2.0)["req"]
    assert np.allclose(a, b, equal_nan=True)


def test_zero_interaction_column_leaves_likelihood_unchanged():
    rng = np.random.default_rng(3)
    rows = []
    for r in range(30):
        for k in range(8):
            rows.append({"race_id": f"r{r:02d}", "pos": k + 1, "umaban_num": k + 1, "S": rng.normal(), "z": rng.normal(), "I": 0.0})
    d = pd.DataFrame(rows)
    a = E.PL(d, ["S", "z"]).race_ll(np.array([0.3, 0.1]))
    b = E.PL(d, ["S", "z", "I"]).race_ll(np.array([0.3, 0.1, 0.7]))
    assert np.allclose(a, b, equal_nan=True)


def test_within_race_permutation_stays_within_race():
    import radar_v2_stamina_eval as SE  # noqa: F401
    rng = np.random.default_rng(0)
    key = np.repeat(np.array(["b", "a", "c"]), [3, 4, 2])
    order = np.lexsort((rng.random(len(key)), key))
    srt = np.argsort(key, kind="stable")
    src = np.empty(len(key), int)
    src[srt] = order
    assert (key[src] == key).all() and sorted(src) == list(range(len(key)))


def test_p3_record_is_separate_from_p1_p2_records():
    """P3(スタミナ)の記録は追補6 の前向きの記録(P1・P2)と別のフォルダに書き、P1・P2 の評価はそのフォルダを読まない。"""
    import inspect
    import radar_v2_live as L
    import radar_v2_prospective_eval as PE
    import radar_v2_stamina_live as SL
    assert SL.PROS_ST != L.PROS and SL.PROS_ST.name == "prospective_stamina"
    assert "prospective_stamina" not in inspect.getsource(PE)
    src = inspect.getsource(SL.record)
    assert "PROS_ST" in src and "L.PROS" not in src
