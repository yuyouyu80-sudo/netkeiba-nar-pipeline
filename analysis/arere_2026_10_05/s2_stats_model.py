# -*- coding: utf-8 -*-
"""荒れ度分析 手順1-b: 15年版(2011〜2025 JRA平地)の万馬券率集計と荒れ度スコア。

入力: out/races_all.csv.gz(s1_build_races.py)
出力(out/):
  s1_overview.csv            区分ごとのレース数・万馬券率(95%CI)・年別
  s1_factor_rates.csv        要因×水準ごとの万馬券率(95% Wilson CI)
  s1_factor_importance.csv   要因ごとの説明力(単変量ロジットのAIC改善/レース, McFadden R2)
  s1_model_eval.csv          ロジスティック回帰(オッズあり/なし/オッズのみ)の評価(2023〜2025)
  s1_calibration.csv         十分位キャリブレーション表
  s1_models.pkl              2011〜2022学習済みモデル(手順2で2026年へ適用)
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
# 目的変数の切替(2026-10-05追加): 既定は「荒れ」(3連複1万円以上)。
# ARERE_TARGET=katai で「荒れない」(3連複1,000円以下)を目的変数にし、出力を out_katai/ へ分ける。
# どちらも列名は arere のまま使う(=その実行での目的変数)。レース表 races_all は out/ の共通ファイル。
TARGET = os.environ.get("ARERE_TARGET", "arere")
RACES = HERE / "out" / "races_all.csv.gz"
OUT = HERE / ("out_katai" if TARGET == "katai" else "out")
OUT.mkdir(exist_ok=True)


def apply_target(df):
    df = df[df["sanrenpuku"].notna()].copy()
    if TARGET == "katai":
        df["arere"] = (df["sanrenpuku"] <= 1000).astype(int)
    else:
        df["arere"] = (df["sanrenpuku"] >= 10000).astype(int)
    return df
CATS = ["一般", "新馬", "未勝利"]
RNG = np.random.default_rng(20261005)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def add_bins(df):
    df = df.copy()
    df["コース"] = df["venue"] + df["surface"] + df["dist_band"]
    df["競馬場"] = df["venue"]
    df["芝ダ"] = df["surface"]
    df["距離帯"] = df["dist_band"]
    df["クラス"] = df["cls"]
    df["年齢条件"] = df["age_cond"]
    df["頭数"] = pd.cut(df["n_horses"], [0, 8, 10, 12, 14, 16, 18], labels=["~8", "9-10", "11-12", "13-14", "15-16", "17-18"]).astype(str)
    df["ハンデ戦(推定)"] = np.where(df["handicap_est"], "ハンデ", "非ハンデ")
    df["牝馬限定(全頭牝)"] = np.where(df["filly_only"], "牝馬のみ", "混合")
    df["馬場状態"] = df["going"].fillna("不明")
    df["天候"] = df["weather"].fillna("不明")
    df["月"] = df["month"].astype(str).str.zfill(2)
    df["開催日目"] = pd.cut(df["nichi"], [0, 2, 4, 6, 8, 12], labels=["1-2日", "3-4日", "5-6日", "7-8日", "9日~"]).astype(str)
    df["1番人気オッズ"] = pd.cut(df["fav_odds"], [0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 999],
                             labels=["~1.5", "1.6-2.0", "2.1-2.5", "2.6-3.0", "3.1-4.0", "4.1-6.0", "6.1~"]).astype(str)
    for col, lab in [("odds_gap12", "1-2人気オッズ比(対数)"), ("odds_gap23", "2-3人気オッズ比(対数)"),
                     ("entropy", "オッズのエントロピー"), ("gini", "オッズのジニ係数"),
                     ("top3_share", "上位3頭の支持率合計"), ("harv_p_cheap", "Harville安い組の確率")]:
        df[lab] = pd.qcut(df[col], 5, labels=["Q1(低)", "Q2", "Q3", "Q4", "Q5(高)"], duplicates="drop").astype(str)
    df["10倍未満の頭数"] = df["n_under10"].clip(upper=7).astype("Int64").astype(str).replace("7", "7+")
    return df


COND_FACTORS = ["コース", "競馬場", "芝ダ", "距離帯", "クラス", "年齢条件", "頭数", "ハンデ戦(推定)", "牝馬限定(全頭牝)",
                "馬場状態", "天候", "月", "開催日目"]
ODDS_FACTORS = ["1番人気オッズ", "1-2人気オッズ比(対数)", "2-3人気オッズ比(対数)", "オッズのエントロピー", "オッズのジニ係数",
                "上位3頭の支持率合計", "Harville安い組の確率", "10倍未満の頭数"]


def ll_bern(k, n):
    p = k / n
    if p <= 0 or p >= 1:
        return 0.0
    return k * np.log(p) + (n - k) * np.log(1 - p)


def factor_tables(df):
    rate_rows, imp_rows = [], []
    for cat in CATS:
        d = df[df["category"] == cat]
        N, K = len(d), int(d["arere"].sum())
        ll0 = ll_bern(K, N)
        for fac in COND_FACTORS + ODDS_FACTORS:
            if fac == "クラス" and cat != "一般":
                continue
            g = d.groupby(fac)["arere"].agg(["size", "sum"])
            ll1 = sum(ll_bern(r["sum"], r["size"]) for _, r in g.iterrows())
            dfree = len(g) - 1
            imp_rows.append(dict(category=cat, factor=fac, kind="オッズ構造" if fac in ODDS_FACTORS else "レース条件",
                                 n_levels=len(g), LR_chi2=2 * (ll1 - ll0), df=dfree,
                                 AIC_gain_per_1000R=1000 * (2 * (ll1 - ll0) - 2 * dfree) / N,
                                 mcfadden_r2=1 - ll1 / ll0,
                                 rate_min=(g["sum"] / g["size"])[g["size"] >= 100].min(),
                                 rate_max=(g["sum"] / g["size"])[g["size"] >= 100].max()))
            for lv, r in g.iterrows():
                lo, hi = wilson(r["sum"], r["size"])
                rate_rows.append(dict(category=cat, factor=fac, level=lv, n=int(r["size"]), n_arere=int(r["sum"]),
                                      rate=r["sum"] / r["size"], ci_lo=lo, ci_hi=hi, lift=(r["sum"] / r["size"]) / (K / N)))
    imp = pd.DataFrame(imp_rows).sort_values(["category", "AIC_gain_per_1000R"], ascending=[True, False])
    return pd.DataFrame(rate_rows), imp


NUM_ODDS = ["lfav", "lodds2", "lodds3", "odds_gap12", "odds_gap23", "entropy", "gini", "top3_share", "n_under10",
            "lharv_cheap", "lharv_best"]


def prep_numeric(df):
    df = df.copy()
    df["lfav"] = np.log(df["fav_odds"])
    df["lodds2"] = np.log(df["odds2"])
    df["lodds3"] = np.log(df["odds3"])
    pc = df["harv_p_cheap"].clip(1e-4, 1 - 1e-4)
    df["lharv_cheap"] = np.log(pc / (1 - pc))
    df["lharv_best"] = np.log(df["harv_p_best"])
    df["n_horses_num"] = df["n_horses"].astype(float)
    return df


def build_model(cat, with_cond, with_odds):
    cat_cols = []
    if with_cond:
        cat_cols = ["コース", "競馬場", "芝ダ", "距離帯", "年齢条件", "頭数", "ハンデ戦(推定)", "牝馬限定(全頭牝)", "馬場状態", "天候",
                    "月", "開催日目"]
        if cat == "一般":
            cat_cols.append("クラス")
    trs = []
    if cat_cols:
        trs.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), cat_cols))
        trs.append(("nh", StandardScaler(), ["n_horses_num"]))
    if with_odds:
        trs.append(("odds", Pipeline([("sc", StandardScaler()), ("sp", SplineTransformer(n_knots=4, degree=3))]), NUM_ODDS))
    ct = ColumnTransformer(trs)
    return Pipeline([("ct", ct), ("lr", LogisticRegression(C=0.3, max_iter=3000))]), cat_cols


def boot_auc(y, p, B=1000):
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    s = []
    for _ in range(B):
        i = RNG.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        s.append(roc_auc_score(y[i], p[i]))
    return np.percentile(s, [2.5, 97.5])


def main():
    df = apply_target(pd.read_csv(RACES, dtype={"race_id": str}))
    df = add_bins(prep_numeric(df))
    d15 = df[(df["year"] >= 2011) & (df["year"] <= 2025)].copy()

    # 概要
    ov = []
    for cat in CATS + ["全体"]:
        d = d15 if cat == "全体" else d15[d15["category"] == cat]
        lo, hi = wilson(d["arere"].sum(), len(d))
        ov.append(dict(category=cat, period="2011-2025", n_races=len(d), n_arere=int(d["arere"].sum()),
                       rate=d["arere"].mean(), ci_lo=lo, ci_hi=hi, median_sanrenpuku=d["sanrenpuku"].median(),
                       mean_sanrenpuku=d["sanrenpuku"].mean(), n_handicap_est=int(d["handicap_est"].sum()),
                       odds_missing=int(d["fav_odds"].isna().sum())))
        for y, g in d.groupby("year"):
            lo, hi = wilson(g["arere"].sum(), len(g))
            ov.append(dict(category=cat, period=str(y), n_races=len(g), n_arere=int(g["arere"].sum()), rate=g["arere"].mean(),
                           ci_lo=lo, ci_hi=hi, median_sanrenpuku=g["sanrenpuku"].median(), mean_sanrenpuku=g["sanrenpuku"].mean()))
    pd.DataFrame(ov).to_csv(OUT / "s1_overview.csv", index=False, encoding="utf-8-sig")

    d15o = d15[d15["fav_odds"].notna()]
    rates, imp = factor_tables(d15o)
    rates.to_csv(OUT / "s1_factor_rates.csv", index=False, encoding="utf-8-sig")
    imp.to_csv(OUT / "s1_factor_importance.csv", index=False, encoding="utf-8-sig")

    # モデル: 2011-2022学習 → 2023-2025評価
    ev_rows, cal_rows, models = [], [], {}
    for cat in CATS:
        d = d15o[d15o["category"] == cat]
        tr, te = d[d["year"] <= 2022], d[d["year"] >= 2023]
        base = tr["arere"].mean()
        for name, wc, wo in [("条件のみ(オッズなし)", True, False), ("条件+オッズ", True, True), ("オッズのみ", False, True)]:
            m, _ = build_model(cat, wc, wo)
            m.fit(tr, tr["arere"])
            p = m.predict_proba(te)[:, 1]
            lo, hi = boot_auc(te["arere"], p)
            q = pd.qcut(p, 10, labels=False, duplicates="drop")
            cal = pd.DataFrame({"q": q, "p": p, "y": te["arere"].to_numpy()}).groupby("q").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
            ece = float((cal["n"] * (cal["pred"] - cal["obs"]).abs()).sum() / cal["n"].sum())
            ev_rows.append(dict(category=cat, model=name, n_train=len(tr), n_test=len(te), base_rate_train=base,
                                rate_test=te["arere"].mean(), AUC=roc_auc_score(te["arere"], p), AUC_lo=lo, AUC_hi=hi,
                                logloss=log_loss(te["arere"], p), logloss_baserate=log_loss(te["arere"], np.full(len(te), base)),
                                brier=brier_score_loss(te["arere"], p), brier_baserate=brier_score_loss(te["arere"], np.full(len(te), base)),
                                ECE=ece, top_decile_rate=cal["obs"].iloc[-1], bottom_decile_rate=cal["obs"].iloc[0]))
            for qq, r in cal.iterrows():
                cal_rows.append(dict(category=cat, model=name, decile=int(qq) + 1, n=int(r["n"]), pred_mean=r["pred"], obs_rate=r["obs"]))
            models[(cat, name)] = m
            print(cat, name, f"AUC={ev_rows[-1]['AUC']:.3f} [{lo:.3f},{hi:.3f}] ECE={ece:.3f}", flush=True)
    pd.DataFrame(ev_rows).to_csv(OUT / "s1_model_eval.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cal_rows).to_csv(OUT / "s1_calibration.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "s1_models.pkl", "wb") as f:
        pickle.dump(models, f)

    # 要約表示
    print(pd.DataFrame(ov).query("period=='2011-2025'")[["category", "n_races", "n_arere", "rate", "ci_lo", "ci_hi"]].to_string())
    for cat in CATS:
        print("==", cat)
        print(imp[imp["category"] == cat].head(12)[["factor", "kind", "AIC_gain_per_1000R", "mcfadden_r2", "rate_min", "rate_max"]].to_string())


if __name__ == "__main__":
    main()
