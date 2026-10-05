# -*- coding: utf-8 -*-
"""荒れ指数・荒れない指数に、取得済みデータ(加工データ含む)の第2弾の材料を足して改良できるか(2026-10-05追加)。

第1弾(s3、馬柱の印・指数・調教・凡走予測など10個)は上乗せ無しだったため、今回は確定単勝オッズに含まれない
「市場の別の見方」と「展開」を試す。
  A オッズの動き(8/1〜): 予想パイプラインが保存した単勝オッズ推移(発走1時間前 → 確定)
      1番人気オッズの変化、人気の散らばり(エントロピー)の変化、上位3頭支持率の変化、1時間前と確定の人気順の一致度
  B 複勝オッズ(9/5〜、data/odds_history_fuku の確定値): 複勝下限オッズから見た市場の「3着内確率」の集中度
      (上位3頭の取り分・エントロピー)と、単勝から見た上位3頭支持率との食い違い
  C 馬連オッズ(9/5〜、data/odds_history_umaren の確定値): 最低馬連オッズ、組確率の集中度(上位組の取り分・エントロピー)
  D 展開(全期間、馬柱のAI展開): 4コーナー予想差(馬身)のばらつき、上位3人気の予想4角位置
いずれも発走前に分かる値(確定オッズは締切時点の値)。着順・払戻は目的変数にだけ使う。
評価: 本番の指数(荒れ=従来版、荒れない=改良版)を土台に、各グループを足したときの対数尤度の差を
開催日を1日ずつ抜く交差検証で比べる(開催日単位ブートストラップの95%区間)。出力: out_extra/
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s6_race_scores as R  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out_extra"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(1010)


def dist_feats(o):
    q = 1.0 / np.asarray(o, float)
    p = q / q.sum()
    ps = np.sort(p)[::-1]
    return float(ps[:3].sum()), float(-(p * np.log(p)).sum() / np.log(len(p)))


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return np.nan


def main():
    dates = sorted({Path(f).stem.split("_")[-1] for f in glob.glob(str(R.DEFAULT_RN_DIR / "odds_history_2026*.csv"))})
    recs = []
    for d in dates:
        r = R.records_from_results(d)
        if r:
            recs += r
    sc = R.score(recs)
    pay = {}
    for d in dates:
        x = pd.read_csv(ROOT / "data" / "payouts" / d[:4] / f"{d}.csv", dtype=str)
        x = x[x["bet_type"] == "3連複"]
        for rid, v in zip(x["race_id"], x["payout"]):
            pay[rid] = max(pay.get(rid, 0), num(v))
    sc["sanrenpuku"] = sc["race_id"].map(pay)
    sc = sc[sc["sanrenpuku"].notna()].copy()
    sc["date"] = sc["race_id"].map({r["race_id"]: r["date"] for r in recs})
    final = dict(zip(sc["race_id"], sc["_odds"]))

    rows = {rid: {} for rid in sc["race_id"]}
    # A: 単勝オッズの動き
    for d in dates:
        h = pd.read_csv(R.DEFAULT_RN_DIR / f"odds_history_{d}.csv", dtype={"race_id": str}, encoding="utf-8-sig")
        h["win_odds"] = pd.to_numeric(h["win_odds"], errors="coerce")
        h["checkpoint_offset_sec"] = pd.to_numeric(h["checkpoint_offset_sec"], errors="coerce")
        for rid, g in h.groupby("race_id"):
            if rid not in rows:
                continue
            first = g[g["checkpoint_offset_sec"] == g["checkpoint_offset_sec"].max()]
            o1 = first.dropna(subset=["win_odds"])
            o1 = o1[o1["win_odds"] > 0]
            if len(o1) < 5:
                continue
            t1, e1 = dist_feats(o1["win_odds"])
            tf, ef = dist_feats(final[rid])
            rows[rid].update(A_dfav=np.log(np.min(final[rid]) / o1["win_odds"].min()), A_dent=ef - e1, A_dtop3=tf - t1)
            rk1 = o1.set_index("umaban")["win_odds"].rank()
            rows[rid]["A_rank_change"] = float(1 - pd.Series(np.argsort(np.argsort(final[rid]))).corr(
                pd.Series(rk1.to_numpy()), method="spearman")) if len(rk1) == len(final[rid]) else np.nan
    # B: 複勝オッズ(確定値)
    for f in glob.glob(str(ROOT / "data" / "odds_history_fuku" / "2026" / "*.csv")):
        h = pd.read_csv(f, dtype={"race_id": str}, encoding="utf-8-sig")
        h = h[h["checkpoint_label"] == "確定"]
        for rid, g in h.groupby("race_id"):
            if rid not in rows:
                continue
            lo = pd.to_numeric(g["fuku_odds_low"], errors="coerce").dropna()
            lo = lo[lo > 0]
            if len(lo) < 5:
                continue
            q = 1.0 / lo.to_numpy()
            p = q / q.sum()
            ps = np.sort(p)[::-1]
            tf, _ = dist_feats(final[rid])
            rows[rid].update(B_top3=float(ps[:3].sum()), B_ent=float(-(p * np.log(p)).sum() / np.log(len(p))),
                             B_vs_win=float(ps[:3].sum()) - tf, B_minfuku=float(np.log(lo.min())))
    # C: 馬連オッズ(確定値)
    for f in glob.glob(str(ROOT / "data" / "odds_history_umaren" / "2026" / "*.csv")):
        h = pd.read_csv(f, dtype={"race_id": str}, encoding="utf-8-sig")
        h = h[h["checkpoint_label"] == "確定"]
        for rid, g in h.groupby("race_id"):
            if rid not in rows:
                continue
            o = pd.to_numeric(g["umaren_odds"], errors="coerce").dropna()
            o = o[o > 0].to_numpy()
            if len(o) < 10:
                continue
            q = 1.0 / o
            p = q / q.sum()
            ps = np.sort(p)[::-1]
            rows[rid].update(C_minumaren=float(np.log(o.min())), C_top3pairs=float(ps[:3].sum()),
                             C_ent=float(-(p * np.log(p)).sum() / np.log(len(p))))
    # D: 展開(AI展開の4コーナー予想)
    for rid in rows:
        p = ROOT / "data" / "newspaper" / f"{rid}.csv"
        if not p.exists():
            continue
        nw = pd.read_csv(p, dtype=str)
        if "corner4_gap_lengths" not in nw:
            continue
        gap = pd.to_numeric(nw["corner4_gap_lengths"], errors="coerce")
        rk = pd.to_numeric(nw["corner4_rank"], errors="coerce")
        ninki = pd.to_numeric(nw["bias_ninki"], errors="coerce")
        if gap.notna().sum() < 5:
            continue
        rows[rid].update(D_gap_sd=float(gap.std()), D_fav_pos=float(rk[ninki <= 3].mean() / len(nw)))
    ex = pd.DataFrame.from_dict(rows, orient="index")
    df = sc[["race_id", "date", "category", "arere", "katai", "sanrenpuku", "n_horses"]].merge(ex, left_on="race_id", right_index=True)
    df["y_arere"] = (df["sanrenpuku"] >= 10000).astype(int)
    df["y_katai"] = (df["sanrenpuku"] <= 1000).astype(int)
    df.to_csv(OUT / "race_level.csv", index=False, encoding="utf-8-sig")

    groups = {"A オッズの動き": ["A_dfav", "A_dent", "A_dtop3", "A_rank_change"],
              "B 複勝オッズ": ["B_top3", "B_ent", "B_vs_win", "B_minfuku"],
              "C 馬連オッズ": ["C_minumaren", "C_top3pairs", "C_ent"],
              "D 展開": ["D_gap_sd", "D_fav_pos"]}
    out = []
    for tname, ycol, icol in [("荒れ", "y_arere", "arere"), ("荒れない", "y_katai", "katai")]:
        for gname, cols in groups.items():
            x = df.dropna(subset=cols)
            y = x[ycol].to_numpy()
            grp = x["date"].to_numpy()
            base = np.log(x[icol] / (1 - x[icol])).to_numpy()[:, None]
            Xs = np.hstack([base, x[cols].to_numpy(float)])
            p0, p1 = np.zeros(len(y)), np.zeros(len(y))
            for g in np.unique(grp):
                te = grp == g
                p0[te] = LogisticRegression(C=10).fit(base[~te], y[~te]).predict_proba(base[te])[:, 1]
                s_ = StandardScaler().fit(Xs[~te])
                p1[te] = LogisticRegression(C=0.5).fit(s_.transform(Xs[~te]), y[~te]).predict_proba(s_.transform(Xs[te]))[:, 1]
            ll = lambda p: np.log(np.where(y == 1, p, 1 - p))  # noqa: E731
            dd = ll(p1) - ll(p0)
            by = np.array([dd[grp == g].sum() for g in np.unique(grp)])
            bs = [by[RNG.integers(0, len(by), len(by))].sum() for _ in range(2000)]
            uni = {c: roc_auc_score(y, x[c]) for c in cols}
            out.append(dict(target=tname, group=gname, n=len(y), n_dates=len(by), rate=y.mean(),
                            AUC_index=roc_auc_score(y, p0), AUC_plus=roc_auc_score(y, p1),
                            dLL=dd.sum(), dLL_lo=np.percentile(bs, 2.5), dLL_hi=np.percentile(bs, 97.5),
                            univariate_AUC=", ".join(f"{k}={v:.2f}" for k, v in uni.items())))
            print(out[-1], flush=True)
    res = pd.DataFrame(out)
    res.to_csv(OUT / "increment.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(res[["target", "group", "n", "n_dates", "AUC_index", "AUC_plus", "dLL", "dLL_lo", "dLL_hi"]].round(3).to_string())


if __name__ == "__main__":
    main()
