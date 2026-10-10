# -*- coding: utf-8 -*-
"""finish_v4 F0-3〜5: 開発期間(2018〜2020年)で、市場の物差しを使った3連複の評価(2026-10-10、プラン第3版)。

市場(確定単勝オッズ)は評価だけに使う。モデルの買い目は dev_utils.parquet のモデルの効用だけから作る。
- 毎年 Y について、市場の γ・λ と期待回収率の表 g は 2014〜Y−1年のレースで作る。
- 買い目: モデルの組確率の上位 N 組(N=10・20・30)と、累積確率が 30・40・50% に達するまで(上限30組)。陰性対照: 市場の組確率で同じ規則。
- 1レースの値: 的中、実際の払戻、g による期待払戻。上積み = Σ(実際 − 期待)/Σ賭け金(比の推定量、ブロック再抽出の BCa)。
- 層: 頭数(10頭以下・11〜14頭・15〜18頭)と、新馬・未勝利/それ以外。診断: 払戻CSVの当たり組の人気と市場PLの順位の対応。
- 増分情報: 組の条件付きロジット a·log p_model + b·log p_mkt(F0-1 以降、候補は発走した全馬で市場と同じ。候補が一致したレースだけ)。
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
CUMS = [0.3, 0.4, 0.5]
CAP = 30
RULES = [f"N{n}" for n in NS] + [f"C{int(c * 100)}" for c in CUMS]
MODELS_F0 = ["u_M1_set", "u_M5_set", "u_M5_pl"]
LAYER_NAMES = ["10頭以下", "11〜14頭", "15〜18頭"]
log = F0.log


def pick(p: np.ndarray, rule: str) -> np.ndarray:
    """組の確率 p から買う組の位置。N10 = 上位10組、C40 = 累積確率が40%に達するまで(上限30組)。"""
    o = np.argsort(-p)
    if rule[0] == "N":
        return o[:int(rule[1:])]
    k = int(np.searchsorted(np.cumsum(p[o]), int(rule[1:]) / 100.0) + 1)
    return o[:min(k, CAP)]


def band_label(n: int) -> str:
    return LAYER_NAMES[0] if n <= 10 else (LAYER_NAMES[1] if n <= 14 else LAYER_NAMES[2])


def main(utils="dev_utils.parquet", params_file="dev_f0_metrics.json", models=None, logit_model="u_M5_set",
         out_name="dev_f0_market.json"):
    """utils の効用の列(models)を、市場の物差しで評価する。F3 以降は --utils f3_utils.parquet などで呼ぶ。"""
    t0 = time.time()
    MODELS = models or MODELS_F0
    U = pd.read_parquet(F0.OUT_DIR / utils)
    params = json.loads((F0.OUT_DIR / params_file).read_text(encoding="utf-8"))
    params = params.get("params", params)
    assert U["year"].max() <= 2020
    odds = MY.load_odds(range(2014, 2021))
    assert odds["year"].max() <= 2020
    pay = F0.load_sanrenpuku(range(2014, 2021))
    pop = load_popularity(F0.YEARS)
    rows, logit, popd = [], [], []
    meta = {}
    n_same = n_diff = 0
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
            om = np.argsort(-p_m)
            rk = np.empty(len(p_m), int)
            rk[om] = np.arange(1, len(p_m) + 1)
            rank_of = dict(zip(k_m.tolist(), rk.tolist()))
            u = ug[rid]
            umod = u["umaban_num"].to_numpy(int)
            same = set(umod.tolist()) == set(um_all.tolist())
            n_same += same
            n_diff += not same
            n = len(um_all)
            rec = {"race_id": rid, "year": Y, "block": g["block"].iloc[0], "field": n, "band": band_label(n), "same_cands": same,
                   "has_nonfinish": bool(u["nonfinish"].any()),
                   "debut_or_maiden": bool(u["is_debut_race"].iloc[0] or u["class_ord"].iloc[0] == 0)}
            for x in w:                                  # 診断: 当たり組の実際の人気(払戻CSV)と市場PLの順位
                if (rid, x) in pop:
                    popd.append((pop[(rid, x)], rank_of.get(x, np.nan)))
            for rule in RULES:                           # 陰性対照: 市場の組確率で同じ規則
                ii = pick(p_m, rule)
                kk = k_m[ii]
                rec[f"mkt_hit_{rule}"] = int(any(int(x) in w for x in kk))
                rec[f"mkt_act_{rule}"] = sum(w.get(int(x), 0) for x in kk)
                rec[f"mkt_exp_{rule}"] = float(e_m[ii].sum())
                rec[f"mkt_k_{rule}"] = len(kk)
                rec[f"mkt_pcov_{rule}"] = float(p_m[ii].sum())
            for mname in MODELS:
                _, name, kind = mname.split("_")
                lm = tuple(params[f"{Y}_{name}_{kind}"]["lam"])
                cc, pp = P.triple_probs(u[mname].to_numpy(float), kind, lm)
                kk_all = MY.combo_keys(umod[cc])
                for rule in RULES:
                    kk = kk_all[pick(pp, rule)]
                    rec[f"{mname}_hit_{rule}"] = int(any(int(x) in w for x in kk))
                    rec[f"{mname}_act_{rule}"] = sum(w.get(int(x), 0) for x in kk)
                    rec[f"{mname}_exp_{rule}"] = float(sum(ex_of.get(int(x), 0.0) for x in kk))
                    rec[f"{mname}_k_{rule}"] = len(kk)
                    rec[f"{mname}_pcov_{rule}"] = float(pp[pick(pp, rule)].sum())   # 買った組の確率の合計(見送りの判定用)
                    rec[f"{mname}_mrank_{rule}"] = float(np.median([rank_of.get(int(x), len(p_m)) for x in kk]))
                if mname == logit_model and same:   # 増分情報のロジット用(全組、候補が市場と同じレース)
                    lpk = np.array([pm_of.get(int(x), np.nan) for x in kk_all])
                    widx = [i for i, x in enumerate(kk_all.tolist()) if x in w]
                    if widx and np.isfinite(lpk).all():
                        logit.append((np.log(pp), lpk, widx, g["block"].iloc[0]))
            rows.append(rec)
        log("year", Y, "races", sum(r["year"] == Y for r in rows), round(time.time() - t0), "秒")
    R = pd.DataFrame(rows)
    R.to_parquet(F0.OUT_DIR / out_name.replace(".json", "_races.parquet"), index=False)
    res = {"years": F0.YEARS, "n_races": int(len(R)), "market_fit": meta, "cap": CAP, "utils": utils, "models": MODELS,
           "logit_model": logit_model,
           "candidates_same_as_market": {"same": int(n_same), "diff": int(n_diff)},
           "n_races_with_nonfinish": int(R["has_nonfinish"].sum()), "rules": {}}
    blk = R["block"].to_numpy()
    layers = [(b, (R["band"] == b).to_numpy()) for b in LAYER_NAMES] + \
             [("新馬・未勝利", R["debut_or_maiden"].to_numpy()), ("それ以外", ~R["debut_or_maiden"].to_numpy())]
    for who in ["mkt"] + MODELS:
        for rule in RULES:
            a = R[f"{who}_act_{rule}"].to_numpy(float)
            e = R[f"{who}_exp_{rule}"].to_numpy(float)
            h = R[f"{who}_hit_{rule}"].to_numpy(float)
            st = 100.0 * R[f"{who}_k_{rule}"].to_numpy(float)
            r_ = {"hit_rate": round(float(h.mean()), 4), "mean_tickets": round(float(st.mean() / 100), 2),
                  "roi": P.ratio_boot(a, st, blk, n=1000),
                  "expected_roi_market": round(float(e.sum() / st.sum()), 4),
                  "uplift": P.ratio_boot(a - e, st, blk, n=1000),
                  "roi_excl_50k": round(float(np.where(a > 50000, 0, a).sum() / st.sum()), 4),
                  "median_market_rank": round(float(R[f"{who}_mrank_{rule}"].median()), 1) if who != "mkt" else None,
                  "by_layer": {}}
            for lab, m in layers:
                r_["by_layer"][lab] = {"n": int(m.sum()), "hit_rate": round(float(h[m].mean()), 4),
                                       "roi": round(float(a[m].sum() / st[m].sum()), 4),
                                       "uplift": P.ratio_boot((a - e)[m], st[m], blk[m], n=500)}
            res["rules"][f"{who}_{rule}"] = r_
            log(who, rule, "hit", r_["hit_rate"], "roi", round(r_["roi"]["ratio"], 4), "exp", r_["expected_roi_market"],
                "uplift", round(r_["uplift"]["ratio"], 4), np.round(r_["uplift"]["ci_bca"], 4))
    m = R["has_nonfinish"].to_numpy()   # 中止馬のいるレースだけの的中率
    res["nonfinish_races"] = {w_: {"n": int(m.sum()), "hit_N10": round(float(R.loc[m, f"{w_}_hit_N10"].mean()), 4)}
                              for w_ in ["mkt"] + MODELS}
    pdf = pd.DataFrame(popd, columns=["pop", "rank"]).dropna()
    from scipy.stats import spearmanr
    lr = np.abs(np.log(pdf["rank"] / pdf["pop"]))
    res["diag_popularity"] = {"n": int(len(pdf)), "spearman": round(float(spearmanr(pdf["pop"], pdf["rank"])[0]), 4),
                              "median_abs_log_ratio": round(float(np.median(lr)), 4),
                              "within_x2": round(float((lr <= np.log(2)).mean()), 4)}
    log("popularity", res["diag_popularity"])
    res["increment_logit"] = increment_logit(logit)
    log("increment", res["increment_logit"])
    res["elapsed_sec"] = round(time.time() - t0, 1)
    (F0.OUT_DIR / out_name).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log("wrote", out_name)


def load_popularity(years) -> dict:
    """払戻CSVの3連複の当たり組の人気(診断用)。{(race_id, キー): 人気}。"""
    out = {}
    for y in years:
        for p in sorted((F0.C.PROJECT_ROOT / "data" / "payouts" / str(y)).glob("*.csv")):
            d = pd.read_csv(p, dtype=str)
            d = d[d["bet_type"] == "3連複"]
            for rid, comb, pp in zip(d["race_id"], d["combination"], d["popularity"]):
                try:
                    a, b, c = sorted(int(x) for x in comb.split("-"))
                    out[(rid, a * 10000 + b * 100 + c)] = int(float(pp))
                except (ValueError, AttributeError, TypeError):
                    pass
    return out


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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--utils", default="dev_utils.parquet")
    ap.add_argument("--params", default="dev_f0_metrics.json")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--logit-model", default="u_M5_set")
    ap.add_argument("--out", default="dev_f0_market.json")
    a = ap.parse_args()
    main(a.utils, a.params, a.models, a.logit_model, a.out)
