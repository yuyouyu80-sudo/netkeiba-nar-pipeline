# -*- coding: utf-8 -*-
"""荒れ指数・荒れない指数の精度向上の検証(2026-10-05追加)。

改良案:
  (3) 単勝オッズからの3連複推定(Harville)を補正する: 2・3着の確率を p^λ2・p^λ3 で計算する
      割引Harville(Lo-Bacon-Shone型)。λ=1 が従来のHarville。あわせて「払戻1,000円以下になりそうな組の確率合計」
      (堅い組の確率)を新しい特徴として追加する。
  (5) 直近の年を重視する: 学習時に年ごとの重み 0.5^((基準年−年)/半減期) を付ける。
選び方(事前に固定): λ と半減期は 2011〜2020年で学習・2021〜2022年で評価して、3区分合計の logloss が最小の組を選ぶ。
最終評価: 選んだ設定で 2011〜2022年に学習し、2023〜2025年と 2026年1〜9月で従来版と比べる(AUC差の95%区間つき)。
出力: out_improve/
"""
import glob
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_stats_model as S  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out_improve"
OUT.mkdir(exist_ok=True)
CATS = ["一般", "新馬", "未勝利"]
LAMBDAS = [(1.0, 1.0), (0.9, 0.8), (0.81, 0.65), (0.7, 0.55)]
HALFLIVES = [None, 8, 4, 2]
RNG = np.random.default_rng(20261005)
P10000 = 0.75 / 100   # 3連複の控除率25%: 払戻1万円以上 ⇔ 推定確率 0.0075 以下
P1000 = 0.75 / 10     # 払戻1,000円以下 ⇔ 推定確率 0.075 以上
TRIPLES = {}


def odds_vectors(race_ids):
    cache = OUT / "odds_vectors.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    want = set(race_ids)
    vec = {}
    for y in range(2011, 2027):
        for f in sorted(glob.glob(str(ROOT / "data" / "race_results" / str(y) / "*.csv"))):
            d = pd.read_csv(f, dtype=str, usecols=["race_id", "finish_pos", "odds_final"])
            d = d[d["race_id"].isin(want) & ~d["finish_pos"].isin(["取", "除"])]
            for rid, g in d.groupby("race_id"):
                o = pd.to_numeric(g["odds_final"], errors="coerce").to_numpy()
                if np.isfinite(o).all() and (o > 0).all():
                    vec[rid] = o
        print("odds", y, len(vec), flush=True)
    pickle.dump(vec, open(cache, "wb"))
    return vec


def triple_probs(p, l2, l3):
    n = len(p)
    if n not in TRIPLES:
        TRIPLES[n] = np.array(list(itertools.combinations(range(n), 3)))
    idx = TRIPLES[n]
    q2, q3 = p ** l2, p ** l3
    s2, s3 = q2.sum(), q3.sum()
    tot = np.zeros(len(idx))
    for a, b, c in itertools.permutations(range(3)):
        i, j, k = idx[:, a], idx[:, b], idx[:, c]
        tot += p[i] * q2[j] / (s2 - q2[i]) * q3[k] / (s3 - q3[i] - q3[j])
    return tot


def lam_features(vec, l2, l3):
    rows = {}
    for rid, o in vec.items():
        q = 1.0 / o
        p = q / q.sum()
        if len(p) < 3:
            continue
        t = triple_probs(p, l2, l3)
        t = t / t.sum()
        rows[rid] = (t[t <= P10000].sum(), t[t >= P1000].sum(), t.max(), -(t * np.log(t)).sum())
    return pd.DataFrame.from_dict(rows, orient="index", columns=["m_ge10000", "m_le1000", "t_max", "t_ent"])


def logit(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def make_X(base, feats):
    X = base.join(feats, on="race_id")
    X["lharv_cheap"] = logit(1 - X["m_ge10000"])       # 従来列を置き換え(安い組の確率)
    X["lharv_best"] = np.log(X["t_max"])
    X["l_le1000"] = logit(X["m_le1000"])
    X["t_ent"] = X["t_ent"]
    return X


NUM_NEW = S.NUM_ODDS + ["l_le1000", "t_ent"]


def fit_eval(X, y, tr, te, cat, half, ref_year, num_cols):
    S.NUM_ODDS[:] = num_cols
    m, _ = S.build_model(cat, True, True)
    w = None
    if half:
        w = 0.5 ** ((ref_year - X.loc[tr, "year"].to_numpy()) / half)
    m.fit(X[tr], y[tr], lr__sample_weight=w)
    return m.predict_proba(X[te])[:, 1]


def boot_diff(y, a, b, B=1000):
    n = len(y)
    s = []
    for _ in range(B):
        i = RNG.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        s.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return np.percentile(s, [2.5, 97.5])


def main():
    NUM_BASE = list(S.NUM_ODDS)
    races = pd.read_csv(S.RACES, dtype={"race_id": str})
    races = races[races["sanrenpuku"].notna() & races["fav_odds"].notna()].copy()
    base = S.add_bins(S.prep_numeric(races)).reset_index(drop=True)
    vec = odds_vectors(base["race_id"])
    feats = {}
    for l2, l3 in LAMBDAS:
        fp = OUT / f"feat_{l2}_{l3}.pkl"
        if fp.exists():
            feats[(l2, l3)] = pd.read_pickle(fp)
        else:
            feats[(l2, l3)] = lam_features(vec, l2, l3)
            feats[(l2, l3)].to_pickle(fp)
        print("features", l2, l3, flush=True)

    res = []
    for tname, tcol in [("荒れ", base["sanrenpuku"] >= 10000), ("荒れない", base["sanrenpuku"] <= 1000)]:
        y = tcol.astype(int).to_numpy()
        # --- 選択: 2011〜2020学習 → 2021〜2022評価 ---
        sel = []
        for lam in LAMBDAS:
            X = make_X(base, feats[lam])
            ok = X["m_ge10000"].notna().to_numpy()
            for half in HALFLIVES:
                tot = 0.0
                for cat in CATS:
                    c = (X["category"] == cat).to_numpy() & ok
                    tr = c & (X["year"] <= 2020).to_numpy()
                    va = c & X["year"].between(2021, 2022).to_numpy()
                    p = fit_eval(X, y, tr, va, cat, half, 2020, NUM_NEW)
                    tot += log_loss(y[va], p) * va.sum()
                sel.append(dict(target=tname, l2=lam[0], l3=lam[1], half=half or 0, val_logloss_sum=tot))
                print(tname, lam, half, round(tot, 2), flush=True)
        # 従来版(λ=1・新特徴なし・重みなし)の検証値
        X0 = make_X(base, feats[(1.0, 1.0)])
        tot0 = 0.0
        for cat in CATS:
            c = (X0["category"] == cat).to_numpy() & X0["m_ge10000"].notna().to_numpy()
            tr = c & (X0["year"] <= 2020).to_numpy()
            va = c & X0["year"].between(2021, 2022).to_numpy()
            p = fit_eval(X0, y, tr, va, cat, None, 2020, NUM_BASE)
            tot0 += log_loss(y[va], p) * va.sum()
        sel.append(dict(target=tname, l2="従来", l3="従来", half=0, val_logloss_sum=tot0))
        sdf = pd.DataFrame(sel)
        sdf.to_csv(OUT / f"selection_{tname}.csv", index=False, encoding="utf-8-sig")
        best = sdf[sdf["l2"] != "従来"].sort_values("val_logloss_sum").iloc[0]
        lam, half = (best["l2"], best["l3"]), (int(best["half"]) or None)
        print(tname, "選択:", lam, half, "従来の検証logloss合計", round(tot0, 2), "→", round(best["val_logloss_sum"], 2), flush=True)

        # --- 最終評価: 2011〜2022学習 → 2023〜2025 / 2026 ---
        X = make_X(base, feats[lam])
        ok = X["m_ge10000"].notna().to_numpy()
        for cat in CATS:
            c = (X["category"] == cat).to_numpy() & ok
            tr = c & (X["year"] <= 2022).to_numpy()
            for pname, te in [("2023-2025", c & X["year"].between(2023, 2025).to_numpy()),
                              ("2026年1-9月", c & (X["year"] == 2026).to_numpy())]:
                p_old = fit_eval(X0, y, tr, te, cat, None, 2022, NUM_BASE)
                p_new = fit_eval(X, y, tr, te, cat, half, 2022, NUM_NEW)
                lo, hi = boot_diff(y[te], p_new, p_old)
                res.append(dict(target=tname, category=cat, period=pname, n=int(te.sum()), rate=y[te].mean(),
                                lam=f"{lam[0]}/{lam[1]}", halflife=half or "なし",
                                AUC_old=roc_auc_score(y[te], p_old), AUC_new=roc_auc_score(y[te], p_new),
                                dAUC_lo=lo, dAUC_hi=hi,
                                LL_old=log_loss(y[te], p_old), LL_new=log_loss(y[te], p_new)))
                print(res[-1], flush=True)
    S.NUM_ODDS[:] = NUM_BASE
    out = pd.DataFrame(res)
    out["dAUC"] = out["AUC_new"] - out["AUC_old"]
    out.to_csv(OUT / "final_eval.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(out[["target", "category", "period", "n", "lam", "halflife", "AUC_old", "AUC_new", "dAUC", "dAUC_lo", "dAUC_hi", "LL_old", "LL_new"]].round(4).to_string())


if __name__ == "__main__":
    main()
