# -*- coding: utf-8 -*-
"""荒れ指数・荒れない指数と、予想ページの score・期待値(EV)の関係(2026-10-05追加)。

対象: predictions_full_{date}.csv(予想パイプラインのscratchpad、各馬の _score)がある 2026-08-01〜10-04 の平地レース。
期待値はページと同じ定義: score をレース内で比例配分した簡易勝率 × 単勝オッズ(確定オッズ)。
レース単位の要約(score の1位、1位−2位差、ばらつき、予想1位馬の人気、EV100%超の頭数、EV最大値、予想1位馬のEV)を作り、
(1) 2つの指数との順位相関、(2) 実際の結果(3連複1万円以上/1,000円以下)に対し、指数に足して当たりが良くなるか
(開催日単位の交差検証)を調べる。出力: out_relation/
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s6_race_scores as R  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out_relation"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(7)


def main():
    files = sorted(glob.glob(str(R.DEFAULT_RN_DIR / "predictions_full_2026*.csv")))
    pf = pd.concat([pd.read_csv(f, dtype={"race_id": str}, encoding="utf-8-sig") for f in files])
    dates = sorted(pf["kaisai_date"].astype(str).unique())
    recs = []
    for d in dates:
        r = R.records_from_results(d)
        if r:
            recs += r
    sc = R.score(recs)  # 荒れ指数・荒れない指数(確定オッズ)
    pay = {}
    for d in dates:
        p = ROOT / "data" / "payouts" / d[:4] / f"{d}.csv"
        x = pd.read_csv(p, dtype=str)
        x = x[x["bet_type"] == "3連複"]
        for rid, v in zip(x["race_id"], x["payout"]):
            pay[rid] = max(pay.get(rid, 0), float(v.replace(",", "")))
    res = {}
    for d in dates:
        x = pd.read_csv(ROOT / "data" / "race_results" / d[:4] / f"{d}.csv", dtype=str)
        x = x[~x["finish_pos"].isin(["取", "除"])]
        for rid, g in x.groupby("race_id"):
            res[rid] = dict(zip(g["umaban"].astype(int), pd.to_numeric(g["odds_final"], errors="coerce")))
    rows = []
    for rid, g in pf.groupby("race_id"):
        if rid not in res or rid not in pay:
            continue
        g = g[g["umaban"].astype(int).isin(res[rid])].copy()
        g["odds"] = g["umaban"].astype(int).map(res[rid])
        g = g[g["odds"].notna() & g["_score"].notna()]
        if len(g) < 5:
            continue
        s = g["_score"].clip(lower=1e-6).to_numpy()
        prob = s / s.sum()
        ev = prob * g["odds"].to_numpy()
        ss = np.sort(s)[::-1]
        top = g.loc[g["_score"].idxmax()]
        pop = g["odds"].rank(method="min").to_numpy()
        rows.append(dict(race_id=rid, date=str(g["kaisai_date"].iloc[0]),
                         score_top=ss[0], score_gap12=ss[0] - ss[1], score_cv=s.std() / s.mean(),
                         score_top3_share=ss[:3].sum() / s.sum(),
                         top_pop=float(pop[g["_score"].to_numpy().argmax()]),
                         n_ev100=int((ev >= 1.0).sum()), share_ev100=float((ev >= 1.0).mean()),
                         ev_max=float(ev.max()), ev_top=float(ev[g["_score"].to_numpy().argmax()]),
                         sanrenpuku=pay[rid]))
    df = pd.DataFrame(rows).merge(sc[["race_id", "category", "arere", "katai", "n_horses"]], on="race_id")
    df["y_arere"] = (df["sanrenpuku"] >= 10000).astype(int)
    df["y_katai"] = (df["sanrenpuku"] <= 1000).astype(int)
    df.to_csv(OUT / "race_level.csv", index=False, encoding="utf-8-sig")
    F = ["score_top", "score_gap12", "score_cv", "score_top3_share", "top_pop", "n_ev100", "share_ev100", "ev_max", "ev_top"]
    LAB = {"score_top": "score 1位の値", "score_gap12": "score 1位−2位の差", "score_cv": "score のばらつき",
           "score_top3_share": "score 上位3頭の占有率", "top_pop": "予想1位馬の人気", "n_ev100": "期待値100%超の頭数",
           "share_ev100": "期待値100%超の割合", "ev_max": "期待値の最大値", "ev_top": "予想1位馬の期待値"}
    cor = []
    for f in F:
        cor.append(dict(item=LAB[f], rho_arere_index=spearmanr(df[f], df["arere"]).statistic,
                        rho_katai_index=spearmanr(df[f], df["katai"]).statistic,
                        AUC_actual_arere=roc_auc_score(df["y_arere"], df[f]),
                        AUC_actual_katai=roc_auc_score(df["y_katai"], df[f])))
    cor = pd.DataFrame(cor)
    cor.to_csv(OUT / "correlation.csv", index=False, encoding="utf-8-sig")
    # 指数に足して良くなるか(開催日を1日ずつ抜く交差検証)
    inc = []
    grp = df["date"].to_numpy()
    for tname, ycol, icol in [("荒れ", "y_arere", "arere"), ("荒れない", "y_katai", "katai")]:
        y = df[ycol].to_numpy()
        base = np.log(df[icol] / (1 - df[icol])).to_numpy()[:, None]
        for extra_name, cols in [("score系", F[:5]), ("期待値系", F[5:]), ("score+期待値", F)]:
            Xs = np.hstack([base, df[cols].to_numpy(float)])
            p0, p1 = np.zeros(len(y)), np.zeros(len(y))
            for g in np.unique(grp):
                te = grp == g
                m0 = LogisticRegression(C=10).fit(base[~te], y[~te]); p0[te] = m0.predict_proba(base[te])[:, 1]
                sc_ = StandardScaler().fit(Xs[~te])
                m1 = LogisticRegression(C=0.5).fit(sc_.transform(Xs[~te]), y[~te]); p1[te] = m1.predict_proba(sc_.transform(Xs[te]))[:, 1]
            d = np.array([(np.log(np.where(y[grp == g] == 1, p1[grp == g], 1 - p1[grp == g])) -
                           np.log(np.where(y[grp == g] == 1, p0[grp == g], 1 - p0[grp == g]))).sum() for g in np.unique(grp)])
            bs = [d[RNG.integers(0, len(d), len(d))].sum() for _ in range(2000)]
            inc.append(dict(target=tname, added=extra_name, n=len(y), AUC_index=roc_auc_score(y, p0), AUC_index_plus=roc_auc_score(y, p1),
                            dLL=d.sum(), dLL_lo=np.percentile(bs, 2.5), dLL_hi=np.percentile(bs, 97.5)))
    inc = pd.DataFrame(inc)
    inc.to_csv(OUT / "increment.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(len(df), "races", df["date"].min(), df["date"].max(), df["category"].value_counts().to_dict())
    print(cor.round(3).to_string())
    print(inc.round(3).to_string())
    print("AUC 指数単独: 荒れ", round(roc_auc_score(df.y_arere, df.arere), 3), "荒れない", round(roc_auc_score(df.y_katai, df.katai), 3))


if __name__ == "__main__":
    main()
