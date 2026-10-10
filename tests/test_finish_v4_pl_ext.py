"""finish_v4/pl_ext.py(PL の λ・集合ロジット・組の確率・ブートストラップ)のテスト(2026-10-10)。合成データのみ。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "jra_model" / "finish_v4"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "jra_model" / "pedigree_reports"))
import pl_ext as P  # noqa: E402


def _sim(R=300, n=10, beta=(0.8, -0.4), seed=0, dnf=False):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(R):
        X = rng.normal(size=(n, len(beta)))
        u = X @ np.array(beta)
        g = u + rng.gumbel(size=n)          # PL の並び(Gumbel の最大の順)
        order = np.argsort(-g)
        pos = np.empty(n)
        pos[order] = np.arange(1, n + 1)
        for i in range(n):
            rows.append({"race_id": f"r{r:04d}", "umaban_num": i + 1, "pos": pos[i], "x0": X[i, 0], "x1": X[i, 1], "horse_id": f"h{r}_{i}"})
    d = pd.DataFrame(rows)
    if dnf:   # 各レースで最下位の馬を「中止」(着順なし)にする
        d.loc[d["pos"] == n, "pos"] = np.nan
    return d


def test_lambda_one_matches_radar_v2_pl():
    import radar_v2_eval as E
    d = _sim(R=50)
    b = P.RaceBatch(d, ["x0", "x1"])
    a = P.race_ll(b, [0.5, -0.2])
    e = E.PL(d, ["x0", "x1"]).race_ll(np.array([0.5, -0.2]))
    assert np.allclose(a, e)


def test_fit_recovers_beta_and_lambda_near_one():
    d = _sim(R=1500, beta=(0.8, -0.4))
    b = P.RaceBatch(d, ["x0", "x1"])
    beta, lam = P.fit(b, fit_lam=True)
    assert np.allclose(beta, [0.8, -0.4], atol=0.08) and all(abs(x - 1) < 0.15 for x in lam)


def test_triple_probs_sum_and_top3():
    rng = np.random.default_rng(1)
    u = rng.normal(size=12)
    for kind, lam in [("pl", (1, 0.8, 0.6)), ("set", (1, 1, 1))]:
        c, p = P.triple_probs(u, kind, lam)
        assert len(c) == 220 and np.isclose(p.sum(), 1.0)
        assert np.isclose(P.top3_probs(c, p, 12).sum(), 3.0)


def test_set_logit_prob_matches_race_ll():
    d = _sim(R=5, n=8)
    b = P.RaceBatch(d, ["x0", "x1"])
    beta = np.array([0.3, 0.2])
    ll = P.race_ll(b, beta, kind="set")
    for r in range(5):
        u = b.X[r][b.M[r]] @ beta
        c, p = P.triple_probs(u, "set")
        top = tuple(sorted(b.orders[b.vrace == r][0]))
        assert np.isclose(np.log(p[[tuple(x) for x in c].index(top)]), ll[r])


def test_pl_set_prob_is_sum_of_orders():
    d = _sim(R=5, n=8)
    b = P.RaceBatch(d, ["x0", "x1"])
    beta, lam = np.array([0.3, 0.2]), (1.0, 0.7, 0.5)
    ll = P.race_ll(b, beta, lam)
    for r in range(5):
        u = b.X[r][b.M[r]] @ beta
        c, p = P.triple_probs(u, "pl", lam)
        top = tuple(sorted(b.orders[b.vrace == r][0]))
        assert np.log(p[[tuple(x) for x in c].index(top)]) >= ll[r] - 1e-12   # 集合の確率 ≥ 実際の並びの確率


def test_non_finisher_is_candidate_but_never_chosen():
    d = _sim(R=20, n=8, dnf=True)
    b = P.RaceBatch(d, ["x0", "x1"])
    assert b.M.sum(axis=1).min() == 8            # 中止の馬も候補(分母)に入る
    for r in range(20):
        for o in b.orders[b.vrace == r]:
            assert not np.isnan(b.P[r, o]).any()


def test_ties_average_over_orders():
    d = pd.DataFrame({"race_id": "r", "umaban_num": range(1, 7), "pos": [1, 2, 2, 4, 5, 6], "x0": [0.5, 0.1, -0.2, 0, 0, 0]})
    b = P.RaceBatch(d, ["x0"])
    assert b.n_var[0] == 2                       # 2着同着 → 2通りの並び
    ll = P.race_ll(b, [1.0])
    d1, d2 = d.copy(), d.copy()
    d1["pos"] = [1, 2, 3, 4, 5, 6]
    d2["pos"] = [1, 3, 2, 4, 5, 6]
    a1 = P.race_ll(P.RaceBatch(d1, ["x0"]), [1.0])[0]
    a2 = P.race_ll(P.RaceBatch(d2, ["x0"]), [1.0])[0]
    assert np.isclose(ll[0], np.log((np.exp(a1) + np.exp(a2)) / 2))


def test_boot_and_ratio():
    rng = np.random.default_rng(0)
    x = rng.normal(0.1, 1, 4000)
    blk = np.repeat(np.arange(400), 10)
    r = P.block_boot(x, blk, n=500)
    assert r["ci_pct"][0] < 0.1 < r["ci_pct"][1] and r["ci_bca"][0] < r["ci_bca"][1]
    rr = P.ratio_boot(np.full(4000, 2.0), np.full(4000, 4.0), blk, n=200)
    assert np.isclose(rr["ratio"], 0.5)


def test_set_logit_fit_moves_away_from_zero_and_ranks_correctly():
    d = _sim(R=800, beta=(0.8, -0.4), seed=3)
    b = P.RaceBatch(d, ["x0", "x1"])
    beta, _ = P.fit(b, kind="set")
    assert np.all(np.isfinite(beta)) and beta[0] > 0.3 and beta[1] < -0.1
    assert P.race_ll(b, beta, kind="set").mean() > P.race_ll(b, [0, 0], kind="set").mean()
