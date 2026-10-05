# -*- coding: utf-8 -*-
"""荒れ度指数 × 荒れない指数 の組み合わせ検証(2026-10-05追加)。

両指数とも「条件+オッズ」モデル(2011〜2022学習、out/s1_models.pkl と out_katai/s1_models.pkl)。
評価は学習に使っていない 2023〜2025年 と 2026-07-11〜09-27(ファクターDBの790レース)。
- 5×5の格子: 区切りは学習期間(2011〜2022)の予測値の5分位で決める(評価期間の情報を区切りに使わない)
- 3分類: 堅い(3連複1,000円以下) / 中間 / 荒れ(1万円以上)
- 合成指数 D = logit(荒れ度) − logit(荒れない) を、当てはめ無しでそのまま評価
- 片方の指数に、もう片方を足すと当たりが良くなるか(2023年で係数を決め、2024〜2025年で評価)
出力: out_combined/
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_stats_model as S  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
OUT = HERE / "out_combined"
OUT.mkdir(exist_ok=True)
CATS = ["一般", "新馬", "未勝利"]
MODEL = "条件+オッズ"
RNG = np.random.default_rng(10052)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def boot_auc_diff(y, a, b, B=1000):
    n = len(y)
    s = []
    for _ in range(B):
        i = RNG.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        s.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return np.percentile(s, [2.5, 97.5])


def main():
    m_a = pickle.load(open(HERE / "out" / "s1_models.pkl", "rb"))
    m_k = pickle.load(open(HERE / "out_katai" / "s1_models.pkl", "rb"))
    df = pd.read_csv(S.RACES, dtype={"race_id": str})
    df = df[df["sanrenpuku"].notna() & df["fav_odds"].notna()].copy()
    df["y_arere"] = (df["sanrenpuku"] >= 10000).astype(int)
    df["y_katai"] = (df["sanrenpuku"] <= 1000).astype(int)
    df["cls3"] = np.where(df["y_katai"] == 1, "堅い", np.where(df["y_arere"] == 1, "荒れ", "中間"))
    fdb = json.load(open(S.HERE.parents[1] / "data/jra_pipeline/factor_database.json", encoding="utf-8"))
    ids26 = {r["race_id"] for r in fdb["races"]}
    X = S.add_bins(S.prep_numeric(df))
    df["p_arere"] = np.nan
    df["p_katai"] = np.nan
    for cat in CATS:
        msk = (df["category"] == cat).to_numpy()
        df.loc[msk, "p_arere"] = m_a[(cat, MODEL)].predict_proba(X[msk])[:, 1]
        df.loc[msk, "p_katai"] = m_k[(cat, MODEL)].predict_proba(X[msk])[:, 1]
    df["D"] = logit(df["p_arere"]) - logit(df["p_katai"])
    df["period"] = np.where(df["year"] <= 2022, "学習2011-2022",
                            np.where(df["year"] <= 2025, "評価2023-2025",
                                     np.where(df["race_id"].isin(ids26), "2026年7-9月", "other")))
    df = df[df["period"] != "other"]
    df[["race_id", "date", "category", "period", "sanrenpuku", "cls3", "p_arere", "p_katai", "D"]].to_csv(
        OUT / "scores_by_race.csv.gz", index=False, encoding="utf-8")

    grid_rows, summ_rows, inc_rows, q_rows = [], [], [], []
    for cat in CATS:
        d = df[df["category"] == cat]
        tr = d[d["period"] == "学習2011-2022"]
        qa = np.quantile(tr["p_arere"], [0.2, 0.4, 0.6, 0.8])
        qk = np.quantile(tr["p_katai"], [0.2, 0.4, 0.6, 0.8])
        for per in ["評価2023-2025", "2026年7-9月"]:
            e = d[d["period"] == per].copy()
            ya, yk = e["y_arere"].to_numpy(), e["y_katai"].to_numpy()
            pa, pk, D = e["p_arere"].to_numpy(), e["p_katai"].to_numpy(), e["D"].to_numpy()
            rho = pd.Series(pa).corr(pd.Series(pk), method="spearman")
            row = dict(category=cat, period=per, n=len(e), rate_arere=ya.mean(), rate_katai=yk.mean(),
                       spearman_arere_vs_katai=rho,
                       AUC_arere_by_arere_idx=roc_auc_score(ya, pa), AUC_arere_by_neg_katai_idx=roc_auc_score(ya, -pk),
                       AUC_arere_by_D=roc_auc_score(ya, D),
                       AUC_katai_by_katai_idx=roc_auc_score(yk, pk), AUC_katai_by_neg_arere_idx=roc_auc_score(yk, -pa),
                       AUC_katai_by_negD=roc_auc_score(yk, -D))
            lo, hi = boot_auc_diff(ya, D, pa)
            row.update(dAUC_arere_D_minus_arere=row["AUC_arere_by_D"] - row["AUC_arere_by_arere_idx"], dAUC_arere_lo=lo, dAUC_arere_hi=hi)
            lo, hi = boot_auc_diff(yk, -D, pk)
            row.update(dAUC_katai_negD_minus_katai=row["AUC_katai_by_negD"] - row["AUC_katai_by_katai_idx"], dAUC_katai_lo=lo, dAUC_katai_hi=hi)
            summ_rows.append(row)
            e["qa"] = np.searchsorted(qa, pa) + 1
            e["qk"] = np.searchsorted(qk, pk) + 1
            g = e.groupby(["qa", "qk"]).agg(n=("cls3", "size"), arere=("y_arere", "mean"), katai=("y_katai", "mean"),
                                           median_pay=("sanrenpuku", "median")).reset_index()
            g.insert(0, "period", per)
            g.insert(0, "category", cat)
            grid_rows.append(g)
            # 4象限(どちらも学習期間の中央値で分割)
            ma, mk = np.median(tr["p_arere"]), np.median(tr["p_katai"])
            e["quad"] = np.where(pa >= ma, "荒れ度 高", "荒れ度 低") + "×" + np.where(pk >= mk, "荒れない 高", "荒れない 低")
            for qd, x in e.groupby("quad"):
                q_rows.append(dict(category=cat, period=per, quadrant=qd, n=len(x), share=len(x) / len(e),
                                   arere=x["y_arere"].mean(), katai=x["y_katai"].mean(),
                                   middle=(x["cls3"] == "中間").mean(), median_pay=x["sanrenpuku"].median()))
        # 増分: 2023で係数、2024-2025で評価
        e = d[d["period"] == "評価2023-2025"]
        fit, ev = e[e["year"] == 2023], e[e["year"] >= 2024]
        for tgt, own, other in [("y_arere", "p_arere", "p_katai"), ("y_katai", "p_katai", "p_arere")]:
            f0 = LogisticRegression(C=100).fit(logit(fit[[own]].to_numpy()), fit[tgt])
            f1 = LogisticRegression(C=100).fit(logit(fit[[own, other]].to_numpy()), fit[tgt])
            p0 = f0.predict_proba(logit(ev[[own]].to_numpy()))[:, 1]
            p1 = f1.predict_proba(logit(ev[[own, other]].to_numpy()))[:, 1]
            inc_rows.append(dict(category=cat, target="荒れ" if tgt == "y_arere" else "堅い", n_eval=len(ev),
                                 AUC_own=roc_auc_score(ev[tgt], p0), AUC_own_plus_other=roc_auc_score(ev[tgt], p1),
                                 logloss_own=log_loss(ev[tgt], p0), logloss_own_plus_other=log_loss(ev[tgt], p1),
                                 coef_other=float(f1.coef_[0][1])))
    pd.DataFrame(summ_rows).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(grid_rows).to_csv(OUT / "grid5x5.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(q_rows).to_csv(OUT / "quadrants.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(inc_rows).to_csv(OUT / "increment.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summ_rows).round(3).T.to_string())
    print(pd.DataFrame(q_rows).round(3).to_string())
    print(pd.DataFrame(inc_rows).round(4).to_string())
    G = pd.concat(grid_rows)
    for cat in CATS:
        x = G[(G.category == cat) & (G.period == "評価2023-2025")]
        print("==", cat, "arere%")
        print((x.pivot(index="qa", columns="qk", values="arere") * 100).round(0).to_string())
        print("katai%")
        print((x.pivot(index="qa", columns="qk", values="katai") * 100).round(0).to_string())
        print("n")
        print(x.pivot(index="qa", columns="qk", values="n").to_string())


if __name__ == "__main__":
    main()
