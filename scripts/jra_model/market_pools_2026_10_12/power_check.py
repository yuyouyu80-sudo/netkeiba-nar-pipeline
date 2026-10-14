# -*- coding: utf-8 -*-
"""検出力の目安(2026-10-12 2回目レビュー対応。シニアエンジニアのレビューの検算スクリプトを整理して移したもの)。

実データの特徴・基準の列はそのまま使い、結果(勝ち馬 / 3着内)だけを「基準モデル + β×特徴(1SDあたり)」から作り直して、
事前登録の判定(1日抜き ΔLL、ブロックブートストラップの区間下限>0)を何回通るかを数える。
- 勝ち馬(条件付きロジット): レースごとに1頭を多項で引く
- 3着内(ロジスティック): 各馬を独立に引く近似(レース内で合計3になる制約は無視)
速さのため当て直しは Newton 法(罰則は本番と同じく新特徴だけ L2=1)。プラセボの条件は省く(区間の条件だけでも通らない目安を見る)。
出力: data/jra_pipeline/market_pools_2026_10_12/power.json
使い方: python -X utf8 power_check.py [n_sim=100]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mp_common as M  # noqa: E402


def fit_cl(X, c, y, P, it=40):
    w = np.zeros(X.shape[1])
    w[0] = 1.0
    for _ in range(it):
        u = X @ w
        e = np.exp(u - u.max())
        Z = np.bincount(c, e)
        pr = e / Z[c]
        nw = np.bincount(c, y)
        g = X.T @ (y - pr * nw[c]) - P * w
        Xbar = np.stack([np.bincount(c, pr * X[:, j], minlength=Z.size) for j in range(X.shape[1])], 1)
        Xc = X - Xbar[c]
        H = -(Xc * (pr * nw[c])[:, None]).T @ Xc - np.diag(P)
        s = np.linalg.solve(H, g)
        w -= s
        if np.abs(s).max() < 1e-9:
            break
    return w


def ll_cl(X, c, y, w):
    u = X @ w
    mx = np.full(c.max() + 1, -np.inf)
    np.maximum.at(mx, c, u)
    Z = np.log(np.bincount(c, np.exp(u - mx[c]))) + mx
    return (u - Z[c]) * y


def fit_lr(X, y, P, it=40):
    w = np.zeros(X.shape[1])
    for _ in range(it):
        z = X @ w
        pr = 1 / (1 + np.exp(-z))
        g = X.T @ (y - pr) - P * w
        H = -(X * (pr * (1 - pr))[:, None]).T @ X - np.diag(P)
        s = np.linalg.solve(H, g)
        w -= s
        if np.abs(s).max() < 1e-10:
            break
    return w


def ll_lr(X, y, w):
    z = X @ w
    return y * z - np.logaddexp(0, z)


def run(name, d, base_cols, xcol, kind, betas, nsim, rng):
    d = d.dropna(subset=[xcol]).reset_index(drop=True)
    x = (d[xcol] / d[xcol].std()).to_numpy()
    B = np.column_stack([d[c].to_numpy() for c in base_cols])
    c = pd.factorize(d["race_id"])[0]
    days = d["date"].to_numpy()
    blk = pd.factorize(d["date"] + "_" + d["venue"])[0]
    nb = blk.max() + 1
    nbase = B.shape[1]
    if kind == "cl":
        X0, X1 = B, np.column_stack([B, x])
        P0, P1 = np.r_[0.0, np.ones(nbase - 1)], np.r_[0.0, np.ones(nbase)]
        w_base = fit_cl(X0, c, d["win"].to_numpy(), P0)
    else:
        X0 = np.column_stack([np.ones(len(d)), B])
        X1 = np.column_stack([X0, x])
        P0, P1 = np.r_[0.0, 0.0, np.ones(nbase - 1)], np.r_[0.0, 0.0, np.ones(nbase)]
        w_base = fit_lr(X0, d["top3"].to_numpy(), P0)
    cidx = [np.flatnonzero(c == r) for r in range(c.max() + 1)]

    def sim_y(beta):
        u = X0 @ w_base + beta * x
        if kind == "cl":
            y = np.zeros(len(d))
            for idx in cidx:
                pr = np.exp(u[idx] - u[idx].max())
                pr /= pr.sum()
                y[idx[rng.choice(len(idx), p=pr)]] = 1
            return y
        return (rng.random(len(d)) < 1 / (1 + np.exp(-u))).astype(float)

    def lodo(y):
        out = np.zeros(len(d))
        for dy in np.unique(days):
            tr, te = days != dy, days == dy
            if kind == "cl":
                ctr, cte = pd.factorize(c[tr])[0], pd.factorize(c[te])[0]
                w0, w1 = fit_cl(X0[tr], ctr, y[tr], P0), fit_cl(X1[tr], ctr, y[tr], P1)
                out[te] = ll_cl(X1[te], cte, y[te], w1) - ll_cl(X0[te], cte, y[te], w0)
            else:
                w0, w1 = fit_lr(X0[tr], y[tr], P0), fit_lr(X1[tr], y[tr], P1)
                out[te] = ll_lr(X1[te], y[te], w1) - ll_lr(X0[te], y[te], w0)
        return out

    res = []
    for beta in betas:
        t0 = time.time()
        tots, los = [], []
        for _ in range(nsim):
            dl = lodo(sim_y(beta))
            v = np.bincount(blk, dl, minlength=nb)
            bs = v[rng.integers(0, nb, (1000, nb))].sum(1)
            tots.append(v.sum())
            los.append(np.percentile(bs, 2.5))
        tots, los = np.array(tots), np.array(los)
        r = {"beta": beta, "odds_ratio_per_sd": float(np.exp(beta)), "mean_dll": float(tots.mean()), "sd_dll": float(tots.std()),
             "p_lo_gt0": float((los > 0).mean()), "p_dll_gt0": float((tots > 0).mean())}
        res.append(r)
        print(name, r, f"{time.time() - t0:.0f}s", flush=True)
    return {"name": name, "n_races": int(d["race_id"].nunique()), "base": base_cols, "feature": xcol, "kind": kind,
            "n_sim": nsim, "rows": res}


def main():
    nsim = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    rng = np.random.default_rng(123)
    betas = [0.0, 0.1, 0.2, 0.3]
    t1 = pd.read_pickle(M.OUT / "t1_rows.pkl")
    t2 = pd.read_pickle(M.OUT / "t2_rows.pkl")
    t3 = pd.read_pickle(M.OUT / "t3_rows.pkl")
    t3 = t3[t3.groupby("race_id")["hasc"].transform("max") == 1]
    out = {"H1": run("H1", t1, ["lP3w"], "D", "lr", betas, nsim, rng),
           "H2": run("H2", t2, ["logp"], "drift60", "cl", betas, nsim, rng),
           "H3": run("H3", t3, ["logp", "r01", "r03"], "E", "cl", betas, nsim, rng)}
    (M.OUT / "power.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
