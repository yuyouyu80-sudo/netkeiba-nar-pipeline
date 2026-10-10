# -*- coding: utf-8 -*-
"""finish_v4 F0-3〜5: 開発期間(2018〜2020年)で、市場の物差しを使った3連複の評価(2026-10-10、プラン第3版)。

市場(確定単勝オッズ)は評価だけに使う。モデルの買い目は dev_utils.parquet のモデルの効用だけから作る。
- 毎年 Y について、市場の γ・λ と期待回収率の表 g は 2014〜Y−1年のレースで作る。
- 買い目: モデルの組確率の上位 N 組(N=10・20・30)。陰性対照: 市場の組確率の上位 N 組。
- 1レースの値: 的中、実際の払戻、g による期待払戻、上積み =(実際 − 期待)/(100·N)。
- 増分情報: 組の条件付きロジット a·log p_model + b·log p_mkt(モデルの組の範囲 = 完走馬の組、F0-1 で中止馬を入れる)。
  a・b はテスト年をまとめて推定し、SE は日付×競馬場のクラスタで頑健にした値。
出力: data/jra_pipeline/finish_v4/dev_f0_market.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import f0_dev_utils as F0  # noqa: E402
import market_yardstick as MY  # noqa: E402
import pl_ext as P  # noqa: E402

NS = [10, 20, 30]
MODELS = ["u_M1_set", "u_M5_set", "u_M5_pl"]
log = F0.log


def main():
    t0 = time.time()
    U = pd.read_parquet(F0.OUT_DIR / "dev_utils.parquet")
    params = json.loads((F0.OUT_DIR / "dev_f0_metrics.json").read_text(encoding="utf-8"))["params"]
    assert U["year"].max() <= 2020
    odds = MY.load_odds(range(2014, 2021))
    assert odds["year"].max() <= 2020
    pay = F0.load_sanrenpuku(range(2014, 2021))
    rows, logit = [], []
    meta = {}
    for Y in F0.YEARS:
        tr = odds[odds["year"].between(2014, Y - 1)]
        gamma, lam = MY.fit_market(tr)
        T = MY.build_return_table(tr, pay, gamma, lam)
        meta[Y] = {"gamma": gamma, "lam": lam}
        log("market", Y, "gamma", round(gamma, 4), "lam", np.round(lam, 3))
        te = odds[odds["year"] == Y]
        Uy = U[U["year"] == Y]
        ug = dict(tuple(Uy.groupby("race_id")))
        for rid, g in te.groupby("race_id"):
            if rid not in pay or rid not in ug or len(g) < 3:
                continue
            w = MY.win_keys(pay[rid])
            c_m, p_m, um_all = MY.market_combos(g, gamma, lam)
            k_m = MY.combo_keys(c_m)
            lp_m = np.log(p_m)
            e_m = T.expect(lp_m, len(um_all))           # 100円あたりの期待払戻(市場から見た)
            pm_of = dict(zip(k_m.tolist(), lp_m.tolist()))
            ex_of = dict(zip(k_m.tolist(), e_m.tolist()))
            u = ug[rid]
            umod = u["umaban_num"].to_numpy(int)
            n = len(um_all)
            rec = {"race_id": rid, "year": Y, "block": g["block"].iloc[0], "field": n,
                   "debut_or_maiden": bool(u["is_debut_race"].iloc[0] or u["class_ord"].iloc[0] == 0)}
            # 陰性対照: 市場の上位 N 組
            om = np.argsort(-p_m)
            for N in NS:
                kk = k_m[om[:N]]
                act = sum(w.get(int(x), 0) for x in kk)
                exp = float(e_m[om[:N]].sum())
                rec[f"mkt_hit_{N}"], rec[f"mkt_act_{N}"], rec[f"mkt_exp_{N}"] = int(act > 0), act, exp
            for mname in MODELS:
                _, name, kind = mname.split("_")
                lm = tuple(params[f"{Y}_{name}_{kind}"]["lam"])
                cc, pp = P.triple_probs(u[mname].to_numpy(float), kind, lm)
                kk_all = MY.combo_keys(umod[cc])
                o = np.argsort(-pp)
                for N in NS:
                    kk = kk_all[o[:N]]
                    act = sum(w.get(int(x), 0) for x in kk)
                    exp = float(sum(ex_of.get(int(x), 0.0) for x in kk))
                    rec[f"{mname}_hit_{N}"], rec[f"{mname}_act_{N}"], rec[f"{mname}_exp_{N}"] = int(act > 0), act, exp
                    # 買った組の市場の人気(組確率の順位)の中央値
                    rk = np.empty(len(p_m), int)
                    rk[om] = np.arange(1, len(p_m) + 1)
                    rank_of = dict(zip(k_m.tolist(), rk.tolist()))
                    rec[f"{mname}_mrank_{N}"] = float(np.median([rank_of.get(int(x), len(p_m)) for x in kk]))
                if mname == "u_M5_set":   # 増分情報のロジット用(モデルの組の範囲)
                    lpk = np.array([pm_of.get(int(x), np.nan) for x in kk_all])
                    widx = [i for i, x in enumerate(kk_all.tolist()) if x in w]
                    if widx and np.isfinite(lpk).all():
                        logit.append((np.log(pp), lpk, widx, g["block"].iloc[0]))
            rows.append(rec)
        log("year", Y, "races", sum(r["year"] == Y for r in rows), round(time.time() - t0), "秒")
    R = pd.DataFrame(rows)
    res = {"years": F0.YEARS, "n_races": int(len(R)), "market_fit": meta, "rules": {}}
    for who in ["mkt"] + MODELS:
        for N in NS:
            a, e, h = R[f"{who}_act_{N}"].to_numpy(float), R[f"{who}_exp_{N}"].to_numpy(float), R[f"{who}_hit_{N}"]
            up = (a - e) / (100 * N)
            res["rules"][f"{who}_N{N}"] = {
                "hit_rate": round(float(h.mean()), 4), "roi": P.ratio_boot(a, np.full(len(a), 100.0 * N), R["block"].to_numpy(), n=1000),
                "expected_roi_market": round(float(e.sum() / (100 * N * len(e))), 4),
                "uplift": P.block_boot(up, R["block"].to_numpy(), n=1000),
                "roi_excl_50k": round(float(np.where(a > 50000, 0, a).sum() / (100 * N * len(a))), 4),
                "median_market_rank": round(float(R[f"{who}_mrank_{N}"].median()), 1) if who != "mkt" else None,
                "by_layer": {lab: round(float(((a - e) / (100 * N))[m].mean()), 4)
                             for lab, m in [("新馬・未勝利", R["debut_or_maiden"].to_numpy()), ("それ以外", ~R["debut_or_maiden"].to_numpy())]}}
            log(who, N, "hit", res["rules"][f"{who}_N{N}"]["hit_rate"], "roi", round(res["rules"][f"{who}_N{N}"]["roi"]["ratio"], 4),
                "exp", res["rules"][f"{who}_N{N}"]["expected_roi_market"], "uplift", round(res["rules"][f"{who}_N{N}"]["uplift"]["mean"], 4),
                res["rules"][f"{who}_N{N}"]["uplift"]["ci_bca"])
    res["increment_logit"] = increment_logit(logit)
    log("increment", res["increment_logit"])
    res["elapsed_sec"] = round(time.time() - t0, 1)
    (F0.OUT_DIR / "dev_f0_market.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log("wrote dev_f0_market.json")


def increment_logit(items) -> dict:
    """log P(組) ∝ a·log p_model + b·log p_mkt(レース内で正規化)。最尤と、ブロックで頑健にした SE。"""
    from scipy.optimize import minimize

    def race_terms(theta, it):
        lpm, lpk, widx, _ = it
        z = theta[0] * lpm + theta[1] * lpk
        mx = z.max()
        e = np.exp(z - mx)
        S = e.sum()
        pr = e / S
        X = np.column_stack([lpm, lpk])
        Ex = pr @ X
        ll = np.log(np.mean(np.exp(z[widx] - mx))) - np.log(S)
        # 当たり組が複数(同着)のときの重み
        ww = np.exp(z[widx] - mx)
        ww = ww / ww.sum()
        xw = ww @ X[widx]
        grad = xw - Ex
        H = -(X.T * pr) @ X + np.outer(Ex, Ex)
        return ll, grad, H

    def f(theta):
        ll, g = 0.0, np.zeros(2)
        for it in items:
            l_, g_, _ = race_terms(theta, it)
            ll += l_
            g += g_
        return -ll, -g
    r = minimize(f, np.array([0.0, 1.0]), jac=True, method="L-BFGS-B")
    th = r.x
    Hs = np.zeros((2, 2))
    sc = {}
    ll1 = 0.0
    for it in items:
        l_, g_, H = race_terms(th, it)
        Hs += H
        sc.setdefault(it[3], np.zeros(2))
        sc[it[3]] += g_
        ll1 += l_
    meat = sum(np.outer(v, v) for v in sc.values())
    Hinv = np.linalg.inv(-Hs)
    V = Hinv @ meat @ Hinv
    se = np.sqrt(np.diag(V))
    # 市場だけ(a=0)の尤度との差
    r0 = minimize(lambda b: (lambda t: (-sum(race_terms(np.array([0.0, t[0]]), it)[0] for it in items)))(b), np.array([1.0]),
                  method="L-BFGS-B")
    return {"n_races": len(items), "a": round(float(th[0]), 4), "a_se_cluster": round(float(se[0]), 4),
            "b": round(float(th[1]), 4), "b_se_cluster": round(float(se[1]), 4),
            "z_a": round(float(th[0] / se[0]), 2), "dll_per_race_vs_market_only": round(float((ll1 + r0.fun) / len(items)), 5)}


if __name__ == "__main__":
    main()
