# -*- coding: utf-8 -*-
"""レースごとの 荒れ指数・荒れない指数・予想3連複払戻額(2026-10-05追加)。

- 荒れ指数 / 荒れない指数: out/s1_models.pkl・out_katai/s1_models.pkl の「条件+オッズ」モデル(2011〜2022学習)の確率
- 予想払戻額: 同じ特徴量(レース条件+確定単勝オッズの形)から log(3連複払戻) の分位点(25%/50%/75%)を
  勾配ブースティングで推定。2011〜2022年で学習、2023〜2025年で精度を評価。
出力(out_payout/):
  eval.csv            2023〜2025年の精度(区分別)
  calib.csv           予想中央値の帯ごとの実際の払戻(2023〜2025年)
  races_2026.csv      2026年の全レース(1/4〜9/27)の 荒れ指数・荒れない指数・予想払戻(25/50/75%)・実際の払戻
  payout_models.pkl   払戻の分位点モデル(s6_race_scores.py が使う)
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_stats_model as S  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
OUT = HERE / "out_payout"
OUT.mkdir(exist_ok=True)
CATS = ["一般", "新馬", "未勝利"]
MODEL = "条件+オッズ"
CAT_COLS = ["category", "コース", "競馬場", "芝ダ", "距離帯", "年齢条件", "クラス", "ハンデ戦(推定)", "牝馬限定(全頭牝)",
            "馬場状態", "天候", "月", "開催日目"]
NUM_COLS = S.NUM_ODDS + ["n_horses_num"]
QS = [0.25, 0.5, 0.75]


def features(X):
    F = X[NUM_COLS].copy()
    for c in CAT_COLS:
        F[c] = X[c].astype("category")
    return F


def main():
    m_a = pickle.load(open(HERE / "out" / "s1_models.pkl", "rb"))
    m_k = pickle.load(open(HERE / "out_katai" / "s1_models.pkl", "rb"))
    df = pd.read_csv(S.RACES, dtype={"race_id": str})
    df = df[df["sanrenpuku"].notna() & df["fav_odds"].notna()].copy()
    X = S.add_bins(S.prep_numeric(df))
    df["荒れ指数"] = np.nan
    df["荒れない指数"] = np.nan
    for cat in CATS:
        msk = (df["category"] == cat).to_numpy()
        df.loc[msk, "荒れ指数"] = m_a[(cat, MODEL)].predict_proba(X[msk])[:, 1]
        df.loc[msk, "荒れない指数"] = m_k[(cat, MODEL)].predict_proba(X[msk])[:, 1]

    F = features(X)
    # 学習期間に無い水準(2026年の新しい組み合わせ等)は欠損扱いにするため、カテゴリを学習期間で固定
    tr = (df["year"] <= 2022).to_numpy()
    for c in CAT_COLS:
        cats = F.loc[tr, c].cat.categories
        F[c] = pd.Categorical(F[c].astype(str), categories=cats)
    y = np.log(df["sanrenpuku"].to_numpy())
    cat_levels = {c: list(F.loc[tr, c].cat.categories) for c in CAT_COLS}
    qmodels = {}
    for q in QS:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=400, learning_rate=0.05,
                                          max_leaf_nodes=31, min_samples_leaf=100, categorical_features="from_dtype",
                                          random_state=0)
        m.fit(F[tr], y[tr])
        qmodels[q] = m
        df[f"予想払戻{int(q * 100)}%"] = np.round(np.exp(m.predict(F)), -1)
        print("fit q", q, flush=True)

    # 予想ページ表示用(s6_race_scores.py)に保存
    with open(OUT / "payout_models.pkl", "wb") as f:
        pickle.dump({"models": qmodels, "cat_levels": cat_levels, "cat_cols": CAT_COLS, "num_cols": NUM_COLS}, f)

    te = df[df["year"].between(2023, 2025)]
    ev = []
    for cat in CATS + ["全体"]:
        x = te if cat == "全体" else te[te["category"] == cat]
        a, p = x["sanrenpuku"].to_numpy(), x["予想払戻50%"].to_numpy()
        lr = np.log(a / p)
        ev.append(dict(category=cat, n=len(x), spearman=spearmanr(a, p).statistic,
                       median_abs_ratio=float(np.exp(np.median(np.abs(lr)))),
                       within_x2=float(np.mean(np.abs(lr) <= np.log(2))), within_x3=float(np.mean(np.abs(lr) <= np.log(3))),
                       in_25_75=float(np.mean((a >= x["予想払戻25%"]) & (a <= x["予想払戻75%"]))),
                       below_50=float(np.mean(a <= p)),
                       median_actual=float(np.median(a)), median_pred=float(np.median(p))))
    pd.DataFrame(ev).to_csv(OUT / "eval.csv", index=False, encoding="utf-8-sig")
    bands = [0, 1000, 2000, 5000, 10000, 20000, 1e9]
    labs = ["~1,000", "1,000-2,000", "2,000-5,000", "5,000-1万", "1万-2万", "2万~"]
    te = te.assign(帯=pd.cut(te["予想払戻50%"], bands, labels=labs))
    cal = te.groupby(["category", "帯"], observed=True).agg(
        n=("sanrenpuku", "size"), 実際の中央値=("sanrenpuku", "median"),
        実際の25パーセンタイル=("sanrenpuku", lambda s: s.quantile(.25)),
        実際の75パーセンタイル=("sanrenpuku", lambda s: s.quantile(.75)),
        万馬券率=("sanrenpuku", lambda s: (s >= 10000).mean()), 千円以下率=("sanrenpuku", lambda s: (s <= 1000).mean())).reset_index()
    cal.to_csv(OUT / "calib.csv", index=False, encoding="utf-8-sig")

    r26 = df[df["year"] == 2026].sort_values(["date", "venue", "race_no"])
    cols = ["race_id", "date", "venue", "race_no", "race_name", "category", "surface", "distance", "n_horses", "fav_odds",
            "荒れ指数", "荒れない指数", "予想払戻25%", "予想払戻50%", "予想払戻75%", "sanrenpuku"]
    out = r26[cols].rename(columns={"sanrenpuku": "実際の3連複払戻", "fav_odds": "1番人気オッズ"})
    out["荒れ指数"] = (out["荒れ指数"] * 100).round(1)
    out["荒れない指数"] = (out["荒れない指数"] * 100).round(1)
    out.to_csv(OUT / "races_2026.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(ev).round(3).to_string())
    print(cal.round(3).to_string())
    print(out.tail(12).to_string())


if __name__ == "__main__":
    main()
