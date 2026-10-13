# -*- coding: utf-8 -*-
"""finish_v4 G5 案B(2026-10-13 ユーザー決定): 当日のオッズを使わずに人気を推定し、モデルの確率が推定人気より高い組を買う。

- 人気の推定(q̂): 1レースの中の softmax。特徴量は G3* と同じ本人の情報 + 過去の人気(前走〜3走前の人気の相対の対数、
  人気と着順のずれ)。学習の正解は、その年より前のレースの確定単勝オッズを割合にしたもの(1/オッズ をレース内で合計1にそろえる)だけ。
  予想するレースの当日のオッズは、買い目の選び方に一切使わない(評価の物差しにだけ使う)。
- 組の推定人気: q̂ を市場の物差しと同じ写し方(γ・λ つきの PL、2014〜前年の確定オッズと着順で当てはめ)で組の確率 p_E にする。
- 買い目: 組を s = log p_G3*(組) − w·log p_E(組) の大きい順に20点。候補は全組か、モデルの上位60組の中だけ。
- 参考(使わない): 推定人気の代わりに実際のオッズで同じ並べ替えをした版(案B の上限の目安)。
- 選び方(結果を見る前に決めた): オッズを使わないルール(w>0)のうち、2018〜2019年の人気をそろえた上積みの点推定が最大のもの。
  ただし2年とも w=0(G3* の上位20組)の上積みを上回ること。満たすものが無ければ案B は採らない。2020年はここでは見ない。
出力: data/jra_pipeline/finish_v4/g5_select.json、dev_ledger.json に試した数。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import f0_dev_utils as F0  # noqa: E402
import g1_decompose as G1  # noqa: E402
import g3_past5 as G3  # noqa: E402
import market_yardstick as MY  # noqa: E402
import model as MD  # noqa: E402
import pl_ext as P  # noqa: E402

OUT = F0.OUT_DIR
SELECT_OUT = OUT / "g5_select.json"
log = F0.log
POP = ["lpop1", "lpop2", "lpop3", "lpop_mean3", "lpop_best3", "lpop_trend", "has_pop1", "A_pop_vs_fin3"]
W_GRID = (0.25, 0.5, 0.75, 1.0)
POOL = 60
L2 = 1e-3


def pop_features() -> pd.DataFrame:
    """features.parquet の今回の人気(pop)・頭数を馬ごとに日付順でずらし、過去の人気だけを作る。"""
    F = pd.read_parquet(G3.P5, columns=["race_id", "horse_id", "date", "pop", "field", "A_pop_vs_fin3"])
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


def fit_soft(X: np.ndarray, q: np.ndarray, starts: np.ndarray, l2: float = L2) -> np.ndarray:
    """レース内 softmax を、割合の正解 q との交差エントロピーで当てはめる(行はレース順に並んでいること)。"""
    sizes = np.diff(np.append(starts, len(q)))
    nr = len(starts)

    def f(b):
        z = X @ b
        zm = np.maximum.reduceat(z, starts)
        ez = np.exp(z - np.repeat(zm, sizes))
        S = np.add.reduceat(ez, starts)
        lse = np.repeat(zm + np.log(S), sizes)
        s = ez / np.repeat(S, sizes)
        return -(q * (z - lse)).sum() / nr + l2 * b @ b, X.T @ (s - q) / nr + 2 * l2 * b

    r = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return r.x


def race_softmax(z: np.ndarray, rid: np.ndarray) -> np.ndarray:
    s = pd.Series(z)
    zm = s.groupby(rid).transform("max").to_numpy()
    e = np.exp(z - zm)
    return e / pd.Series(e).groupby(rid).transform("sum").to_numpy()


def with_q(D: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """確定オッズの割合 q(学習の正解、評価の物差し)。オッズの無い馬は外し、レース内で合計1にそろえる。"""
    o = odds.set_index(["race_id", "horse_id"])["lo"]
    D = D.join(o, on=["race_id", "horse_id"])
    D = D[D["lo"].notna()].copy()
    e = np.exp(D["lo"].to_numpy())
    D["q"] = e / pd.Series(e, index=D.index).groupby(D["race_id"]).transform("sum").to_numpy()
    return D


RULES = ["w0"] + [f"w{w}" for w in W_GRID] + [f"w{w}_top{POOL}" for w in (0.5, 1.0)] + ["ref_mkt_w0.5", "ref_mkt_w1.0"]


def run_years(years, feats, rules=RULES, races_out=OUT / "g5_dev_races.parquet") -> tuple[pd.DataFrame, dict]:
    """years の各年: G3* の効用と推定人気から、各ルールの20点の的中・払戻・市場から見た期待を集める(係数は 2014〜Y−1)。"""
    efeats = feats + POP
    F = pd.read_parquet(G3.P5, columns=["race_id", "horse_id"] + G3.P5_COLS).drop_duplicates(["race_id", "horse_id"]).set_index(["race_id", "horse_id"])
    PF = pop_features()
    C = G3.course_table()
    pay = F0.load_sanrenpuku(range(2014, max(years) + 1))
    odds = MY.load_odds(range(2014, max(years) + 1))
    assert odds["year"].max() <= max(years)
    rows, qual = [], {}
    for Y, D in G1.load_tables().items():
        if Y not in years:
            continue
        D = G3.add_g3(D, F, C).join(PF, on=["race_id", "horse_id"])
        for c in POP:
            D[c] = D[c].fillna(0.0)
        tr, te = D[D["year"].between(F0.TRAIN_START, Y - 1)], D[D["year"] == Y]
        beta, lam = MD.fit_std(tr, feats, 0.0)
        _, _, u = MD.test_eval(te, feats, beta, lam, pay)
        te = te.assign(u=u)
        # 人気の推定(学習は 2014〜Y−1 の確定オッズの割合)
        trq = with_q(tr, odds).sort_values("race_id", kind="stable")
        Etr = trq[efeats].fillna(0.0)             # 欠けは0(fit_std・test_eval と同じ扱い)
        mu, sd = Etr.mean(), Etr.std().replace(0, 1.0).fillna(1.0)
        Xtr = ((Etr - mu) / sd).to_numpy(float)
        starts = np.flatnonzero(np.r_[True, trq["race_id"].to_numpy()[1:] != trq["race_id"].to_numpy()[:-1]])
        b = fit_soft(Xtr, trq["q"].to_numpy(float), starts)
        te = te.assign(qhat=race_softmax(((te[efeats].fillna(0.0) - mu) / sd).to_numpy(float) @ b, te["race_id"].to_numpy()))
        # 市場の物差し(評価だけ)
        trm = odds[odds["year"].between(2014, Y - 1)]
        gamma, mlam = MY.fit_market(trm)
        Tg = MY.build_return_table(trm, pay, gamma, mlam)
        teq = with_q(te, odds)
        kl = []
        for rid, g in teq.groupby("race_id"):
            q, qh, pw = g["q"].to_numpy(), g["qhat"].to_numpy(), np.exp(g["u"] - g["u"].max()).to_numpy()
            qh, pw = qh / qh.sum(), pw / pw.sum()
            kl.append((float((q * np.log(q / qh)).sum()), float((q * np.log(q / pw)).sum()),
                       float(pd.Series(q).corr(pd.Series(qh), method="spearman")), float(pd.Series(q).corr(pd.Series(pw), method="spearman"))))
        kl = np.array(kl)
        qual[Y] = {"KL_market_to_qhat": round(float(kl[:, 0].mean()), 4), "KL_market_to_model_win": round(float(kl[:, 1].mean()), 4),
                   "spearman_qhat": round(float(np.nanmean(kl[:, 2])), 4), "spearman_model": round(float(np.nanmean(kl[:, 3])), 4)}
        log(Y, "人気の推定", qual[Y])
        ug = dict(tuple(te.groupby("race_id")))
        for rid, g in odds[odds["year"] == Y].groupby("race_id"):
            if rid not in pay or rid not in ug or len(g) < 3:
                continue
            x = ug[rid]
            umod = x["umaban_num"].to_numpy(int)
            if set(umod.tolist()) != set(g["umaban_num"].to_numpy(int).tolist()):
                continue
            w = MY.win_keys(pay[rid])
            c_m, p_m, _ = MY.market_combos(g, gamma, mlam)
            k_m = MY.combo_keys(c_m)
            ex_of = dict(zip(k_m.tolist(), Tg.expect(np.log(p_m), len(g)).tolist()))
            lpm_of = dict(zip(k_m.tolist(), np.log(p_m).tolist()))
            om = np.argsort(-p_m)
            rk = np.empty(len(p_m), int)
            rk[om] = np.arange(1, len(p_m) + 1)
            rank_of = dict(zip(k_m.tolist(), rk.tolist()))
            cc, pf = P.triple_probs(x["u"].to_numpy(float), "pl", tuple(lam))
            ce, pe = P.triple_probs(gamma * np.log(x["qhat"].to_numpy(float)), "pl", tuple(mlam))
            assert (cc == ce).all()
            kk = MY.combo_keys(umod[cc])
            lpf, lpe = np.log(pf), np.log(pe)
            lpm = np.array([lpm_of[int(k)] for k in kk])
            top = np.argsort(-pf, kind="stable")[:POOL]
            rec = {"race_id": rid, "year": Y, "block": g["block"].iloc[0]}
            for rule in rules:
                if rule == "w0":
                    s = lpf
                elif rule.startswith("ref_mkt"):
                    s = lpf - float(rule.split("_w")[1]) * lpm
                else:
                    s = lpf - float(rule[1:].split("_")[0]) * lpe
                if rule.endswith(f"top{POOL}"):
                    s = np.where(np.isin(np.arange(len(s)), top), s, -np.inf)
                ii = np.argsort(-s, kind="stable")[:20]
                kx = kk[ii]
                rec[f"{rule}_hit"] = int(any(int(k) in w for k in kx))
                rec[f"{rule}_act"] = sum(w.get(int(k), 0) for k in kx)
                rec[f"{rule}_exp"] = float(sum(ex_of.get(int(k), 0.0) for k in kx))
                rec[f"{rule}_mrank"] = float(np.median([rank_of.get(int(k), len(p_m)) for k in kx]))
                rec[f"{rule}_pcov"] = float(pf[ii].sum())
            rows.append(rec)
        assert np.isfinite(te["qhat"]).all(), "推定人気に NaN"
        log(Y, "races", sum(r["year"] == Y for r in rows))
    R = pd.DataFrame(rows)
    R.to_parquet(races_out, index=False)
    return R, qual


def summarize(R: pd.DataFrame, rules, years) -> dict:
    b_ = R["block"].to_numpy()
    st = np.full(len(R), 2000.0)
    res = {}
    for rule in rules:
        a, e = R[f"{rule}_act"].to_numpy(float), R[f"{rule}_exp"].to_numpy(float)
        res[rule] = {"hit": round(float(R[f"{rule}_hit"].mean()), 4), "roi": P.ratio_boot(a, st, b_, n=1000),
                     "uplift": P.ratio_boot(a - e, st, b_, n=1000), "expected_roi_market": round(float(e.sum() / st.sum()), 4),
                     "median_market_rank": float(R[f"{rule}_mrank"].median()),
                     "by_year_uplift": {int(y): round(float(((a - e)[R["year"] == y]).sum() / st[R["year"] == y].sum()), 4) for y in years},
                     "by_year_roi": {int(y): round(float(a[R["year"] == y].sum() / st[R["year"] == y].sum()), 4) for y in years}}
        log(rule, res[rule]["hit"], round(res[rule]["roi"]["ratio"], 4), round(res[rule]["uplift"]["ratio"], 4),
            [round(x, 4) for x in res[rule]["uplift"]["ci_bca"]], res[rule]["by_year_uplift"], "mrank", res[rule]["median_market_rank"])
    return res


def main():
    sel3 = json.loads(G3.SELECT_OUT.read_text(encoding="utf-8"))
    years, rules = G1.SEL_YEARS, RULES
    R, qual = run_years(years, sel3["g3_spec"])
    res = summarize(R, rules, years)
    base = res["w0"]["by_year_uplift"]
    ok = [r for r in rules if r != "w0" and not r.startswith("ref") and all(res[r]["by_year_uplift"][y] > base[y] for y in years)]
    chosen = max(ok, key=lambda r: res[r]["uplift"]["ratio"]) if ok else None
    out = {"years": years, "g3_name": sel3["g3_name"], "pop_features": POP, "rules": rules, "pool": POOL, "l2": L2,
           "n_races": int(len(R)), "popularity_quality": qual, "results": res, "eligible": ok, "chosen": chosen}
    SELECT_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    G1.ledger({"stage": "G5 案B select", "n_specs_tried": len([r for r in rules if not r.startswith("ref")]), "chosen": chosen})
    log("G5 chosen", chosen)


if __name__ == "__main__":
    main()
