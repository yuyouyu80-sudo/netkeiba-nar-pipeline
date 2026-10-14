# -*- coding: utf-8 -*-
"""T1: 単勝と複勝の食い違い(Hausch・Ziemba・Rubinstein 型)。事前登録: PREREG_market_pools_2026_10_12.md

主 H1: 3着内の二値ロジスティック [logit P3w] + D、D = logit(q) − logit(P3w)
副 H1b: 勝ち馬の条件付きロジット [log p] + D
参考 H1-sync: P3w を複勝と同じ時点(「確定」取得)の単勝オッズで計算した版
買い方(参考): P3w × 複勝下限オッズ ≥ T の馬の複勝を100円ずつ
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mp_common as M  # noqa: E402


def logit(x):
    return np.log(x / (1 - x))


def build():
    lam = M.fit_lambda()
    l2, l3 = lam["lambda2"], lam["lambda3"]
    rows = []
    for f in sorted(glob.glob(str(M.ROOT / "data" / "odds_history_fuku" / "2026" / "*.csv"))):
        date = Path(f).stem
        if not M.in_period(date):
            continue
        fu = pd.read_csv(f, dtype={"race_id": str}, encoding="utf-8-sig")
        fu = fu[fu["checkpoint_label"] == "確定"].drop_duplicates(["race_id", "umaban"], keep="last")
        res = M.results([date])
        if res.empty:
            continue
        th_path = M.tansho_hist_path(date)
        th = None
        if th_path is not None:
            th = pd.read_csv(th_path, dtype={"race_id": str}, encoding="utf-8-sig")
            th = th[th["checkpoint_label"] == "確定"].drop_duplicates(["race_id", "umaban"], keep="last")
            th = th.set_index(["race_id", "umaban"])["win_odds"]
        for rid, g in res.groupby("race_id"):
            fg = fu[fu["race_id"] == rid].set_index("umaban")
            if fg.empty:
                continue
            # 2026-10-12 レビュー対応: 出走7頭以下は複勝が2着までの払いなので外す(3着内を目的にできない)
            if (g["odds"] > 0).sum() < 8:
                continue
            g = g[(g["odds"] > 0) & g["um"].isin(fg.index)].copy()
            if len(g) < 8 or g["fin"].isna().all():
                continue
            g["low"] = pd.to_numeric(fg.loc[g["um"].astype(int), "fuku_odds_low"], errors="coerce").to_numpy()
            g["high"] = pd.to_numeric(fg.loc[g["um"].astype(int), "fuku_odds_high"], errors="coerce").to_numpy()
            if not ((g["low"] > 0) & (g["high"] > 0)).all():
                continue
            p = M.norm_p(g["odds"].to_numpy())
            g["p"] = p
            g["P3w"] = M.top3_prob(p, l2, l3)
            q = 1.0 / np.sqrt(g["low"].to_numpy() * g["high"].to_numpy())
            q = np.clip(q / q.sum() * 3.0, 1e-4, 0.98)
            g["q"] = q
            if th is not None:
                wo = pd.to_numeric(pd.Series([th.get((rid, int(u)), np.nan) for u in g["um"]]), errors="coerce").to_numpy()
                if (wo > 0).all():
                    g["P3w_sync"] = M.top3_prob(M.norm_p(wo), l2, l3)
            rows.append(g)
    df = pd.concat(rows, ignore_index=True)
    df["top3"] = (df["fin"] <= 3).astype(float)
    df["win"] = (df["fin"] == 1).astype(float)
    df["lP3w"] = logit(df["P3w"])
    df["D"] = logit(df["q"]) - df["lP3w"]
    df["logp"] = np.log(df["p"])
    df["pop"] = df.groupby("race_id")["p"].rank(ascending=False, method="first").clip(upper=12)
    if "P3w_sync" in df:
        df["lP3s"] = logit(df["P3w_sync"])
        df["Ds"] = logit(df["q"]) - df["lP3s"]
    # 特徴はレース内平均を引き、全体の標準偏差で割る
    # 参考(事後、レビュー対応): 幾何中点の変換は人気馬の q を押しつぶすため、人気順ごとの平均 D を引いた残差
    df["Dres"] = df["D"] - df.groupby("pop")["D"].transform("mean")
    for c in ("D", "Ds", "Dres"):
        if c in df:
            df[c] = df[c] - df.groupby("race_id")[c].transform("mean")
            df[c] = df[c] / df[c].std()
    return df


def place_bets(df, Ts=(1.0, 1.1, 1.2)):
    pays = []
    for d in sorted(df["date"].unique()):
        p = M.ROOT / "data" / "payouts" / d[:4] / f"{d}.csv"
        if p.exists():
            x = pd.read_csv(p, dtype=str)
            pays.append(x[x["bet_type"] == "複勝"])
    pay = pd.concat(pays)
    pay["um"] = pd.to_numeric(pay["combination"], errors="coerce")
    pay["payout"] = pd.to_numeric(pay["payout"], errors="coerce")
    pmap = pay.set_index(["race_id", "um"])["payout"]
    have = set(pay["race_id"])
    d = df[df["race_id"].isin(have)].copy()
    d["ret"] = [pmap.get((r, u), 0.0) for r, u in zip(d["race_id"], d["um"])]
    out = []
    rng = np.random.default_rng(0)
    for col in ("P3w", "q"):
        for T in Ts:
            b = d[d[col] * d["low"] >= T]
            if b.empty:
                continue
            blk = (b["date"] + "_" + b["venue"]).to_numpy()
            s = b.groupby(blk).agg(ret=("ret", "sum"), n=("ret", "size"))
            v = s.to_numpy()
            bs = v[rng.integers(0, len(v), (4000, len(v)))].sum(axis=1)
            roi = bs[:, 0] / (bs[:, 1] * 100)
            out.append({"prob": col, "T": T, "n_bets": int(len(b)), "n_races": int(b["race_id"].nunique()),
                        "hit": float((b["ret"] > 0).mean()), "roi": float(b["ret"].sum() / (len(b) * 100)),
                        "lo": float(np.percentile(roi, 2.5)), "hi": float(np.percentile(roi, 97.5))})
    out.append({"prob": "全馬", "T": None, "n_bets": int(len(d)), "n_races": int(d["race_id"].nunique()),
                "hit": float((d["ret"] > 0).mean()), "roi": float(d["ret"].sum() / (len(d) * 100))})
    return out, sorted(d["date"].unique())


def main():
    df = build()
    df.to_pickle(M.OUT / "t1_rows.pkl")
    res = {"H1": M.evaluate(df, ["lP3w"], ["D"], "top3", "logit", "H1 3着内 + 単複食い違い", n_placebo=200),
           "H1b": M.evaluate(df, ["logp"], ["D"], "win", "clogit", "H1b 勝ち馬 + 単複食い違い")}
    if "Ds" in df:
        ds = df.dropna(subset=["Ds"])
        res["H1_sync"] = M.evaluate(ds, ["lP3s"], ["Ds"], "top3", "logit", "参考 同時点の単勝で計算")
    res["H1_popres"] = M.evaluate(df, ["lP3w"], ["Dres"], "top3", "logit", "参考(事後) 人気順の偏りを除いた食い違い")
    raw = np.log(df["q"] / (1 - df["q"])) - df["lP3w"]  # 人気順ごとの D の平均(標準化前)
    res["D_by_pop"] = {int(k): float(v) for k, v in raw.groupby(df["pop"]).mean().items()}
    # 較正の確認: 3着内率(実際)と P3w・q の平均
    cal = {"top3_rate": float(df["top3"].mean() * df.groupby("race_id").size().mean() / 3),
           "mean_P3w_by_bin": df.groupby(pd.qcut(df["P3w"], 5))[["P3w", "q", "top3"]].mean().round(3).values.tolist()}
    bets, bet_dates = place_bets(df)
    res.update({"bets": bets, "bet_dates": bet_dates, "calibration": cal, "dates": sorted(df["date"].unique()),
                "src_count": df.drop_duplicates("race_id")["src"].value_counts().to_dict()})
    (M.OUT / "t1_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for k in ("H1", "H1b", "H1_sync", "H1_popres"):
        if k in res:
            r = res[k]
            print(f"{r['name']}: {r['n_races']}R/{r['n_days']}日 ΔLL {r['dll']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] "
                  f"プラセボ最大 {r['placebo_max']:+.2f} 係数 {r['coef']} 合格={r['pass']}")
    for b in bets:
        print(b)
    print(cal)


if __name__ == "__main__":
    main()
