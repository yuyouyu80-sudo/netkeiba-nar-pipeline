"""血統レーダー v2 の評価(radar_v2_eval.py)・基準(radar_v2_reference.py)のテスト(2026-10-06、R4レビュー後に追加)。
合成データのみ。radar_v2_eval は import 時に radar_v2/ の r2_inner_validation.json・race_req_meta.json を読む(Git管理下)。"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "jra_model" / "pedigree_reports"))
import radar_v2_eval as E  # noqa: E402
import radar_v2_reference as R  # noqa: E402


def test_tables_value_leaves_own_runs_out():
    # 父 s の産駒 a(xc=1, w=2)と b(xc=−1, w=3)、k=1。a の値は「b だけで作った縮約値」になるはず。
    T = E.Tables.__new__(E.Tables)
    sw, k = 5.0, 1.0
    mean = (2 * 1 + 3 * -1) / sw
    T.anc = {("front", "-", "sire"): pd.DataFrame({"mean": [mean], "sw": [sw], "k": [k], "tau": [1.0]}, index=pd.Index(["s"], name="ancestor_id"))}
    T.hv = {("front", "-"): pd.DataFrame({"xc": [1.0, -1.0], "w": [2.0, 3.0]}, index=pd.Index(["a", "b"], name="horse_id"))}
    df = pd.DataFrame({"horse_id": ["a", "b", "c"], "ped_S_horse_id": ["s", "s", "s"],
                       "ped_DS_horse_id": [None] * 3, "ped_SS_horse_id": [None] * 3})
    v = T.value(df, "front", "-")
    assert np.isclose(v.iloc[0], (3 * -1) / (3 + k))        # a: b だけ
    assert np.isclose(v.iloc[1], (2 * 1) / (2 + k))          # b: a だけ
    assert np.isclose(v.iloc[2], sw * mean / (sw + k))       # c: 表に本人がいない → 全産駒


def test_prior_stats_excludes_same_day_races():
    Rr = pd.DataFrame({"L1": ["x", "x", "x", "y"], "date": pd.to_datetime(["2021-01-05", "2021-01-05", "2021-01-12", "2021-01-12"]),
                       "v": [1.0, 3.0, 10.0, 7.0]}, index=["r1", "r2", "r3", "r4"])
    out = R.prior_stats(Rr, "L1", "v")
    assert out.loc[["r1", "r2"], "L1_n"].tolist() == [0, 0]   # 同じ日の別レースは入らない
    assert out.loc["r3", "L1_n"] == 2 and out.loc["r3", "L1_sum"] == 4.0
    assert out.loc["r4", "L1_n"] == 0                         # 別のキー


def _brute_top3_ll(u, order):
    """上位3頭を順に選ぶ確率の対数(総当たりの定義どおり)。"""
    rem = list(range(len(u)))
    ll = 0.0
    for k in order[:3]:
        ll += u[k] - np.log(np.sum(np.exp(u[rem])))
        rem.remove(k)
    return ll


def test_pl_matches_brute_force_and_nested_beta_zero():
    rng = np.random.default_rng(0)
    rows = []
    for r in range(3):
        n = 4 + r
        pos = rng.permutation(n) + 1
        for i in range(n):
            rows.append({"race_id": f"R{r}", "pos": pos[i], "umaban_num": i + 1, "S": rng.normal(), "F": rng.normal()})
    df = pd.DataFrame(rows)
    beta = np.array([0.7, -0.3])
    ll = E.PL(df, ["S", "F"]).race_ll(beta)
    for j, (rid, g) in enumerate(df.groupby("race_id")):
        u = g[["S", "F"]].to_numpy() @ beta
        order = list(np.argsort(g["pos"].to_numpy()))
        assert np.isclose(ll[j], _brute_top3_ll(u, order))
    assert np.allclose(E.PL(df, ["S"]).race_ll(np.array([0.7])), E.PL(df, ["S", "F"]).race_ll(np.array([0.7, 0.0])))
    # 勾配は数値微分と一致
    m = E.PL(df, ["S", "F"])
    _, g = m.nll_grad(beta)
    eps = 1e-6
    num = [(m.nll_grad(beta + eps * e)[0] - m.nll_grad(beta - eps * e)[0]) / (2 * eps) for e in np.eye(2)]
    assert np.allclose(g, num, atol=1e-5)


def test_pl_drops_races_with_fewer_than_three_runners():
    df = pd.DataFrame({"race_id": ["A", "A", "B", "B", "B"], "pos": [1, 2, 1, 2, 3], "umaban_num": [1, 2, 1, 2, 3], "S": [0.1] * 5})
    m = E.PL(df, ["S"])
    assert m.race_ids.tolist() == ["B"] and np.isfinite(m.race_ll(np.array([1.0]))).all()


def test_holm_step_down():
    h = E.holm({"a": 0.001, "b": 0.004, "c": 0.03})
    assert [h[k]["pass"] for k in "abc"] == [True, True, False]
    assert np.isclose(h["a"]["threshold"], 0.025 / 3, atol=1e-5)   # 記録は小数5桁に丸め(判定は丸める前の値)
    # 最小の p が通らなければ、閾値を下回る p があっても以降はすべて不合格
    h = E.holm({"a": 0.009, "b": 0.010, "c": 0.011})
    assert not any(v["pass"] for v in h.values())
    # 仮置き p=1 は最後に回り、他の閾値を最も厳しくする
    h = E.holm({"1": 0.0005, "2": 0.001, "5": 1.0})
    assert h["1"]["threshold"] < h["2"]["threshold"] < h["5"]["threshold"] and not h["5"]["pass"]


def test_placebo_noise_feature_has_near_zero_or_negative_dll():
    # 結果と無関係な特徴量を足しても、テストの ΔLL は 0 付近(わずかに負)になる
    rng = np.random.default_rng(1)
    rows = []
    for r in range(600):
        n = 10
        s = rng.normal(size=n)
        u = 0.8 * s + rng.gumbel(size=n)
        pos = np.argsort(np.argsort(-u)) + 1
        for i in range(n):
            rows.append({"race_id": f"R{r:04d}", "pos": pos[i], "umaban_num": i + 1, "S": s[i], "N": rng.normal(),
                         "block": f"B{r // 5}"})
    df = pd.DataFrame(rows)
    fit, test = df[df["race_id"] < "R0300"], df[df["race_id"] >= "R0300"]
    c = E.compare(fit, test, ["S"], ["N"], "noise")
    assert c["dll_per_race"] < 0.01 and c["ci95_block"][0] < 0
