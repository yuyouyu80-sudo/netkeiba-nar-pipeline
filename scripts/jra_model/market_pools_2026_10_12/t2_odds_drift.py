# -*- coding: utf-8 -*-
"""T2: 馬ごとの単勝オッズの動き。事前登録: PREREG_market_pools_2026_10_12.md

主 H2: 勝ち馬の条件付きロジット [log p_final] + drift60(= log p_final − log p_1時間前)
(2026-10-12 2回目レビュー対応: 予定時刻から±60秒以内に取得したスナップショットだけを使う)
副 H2b: drift10(10分前から確定)、H2c: 3着内ロジスティック [logit P3w] + drift60
参考: drift60 の5分位ごとの単勝回収率(確定オッズで買った場合)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mp_common as M  # noqa: E402


def build():
    lam = M.fit_lambda()
    rows = []
    for date in [d for d in M.tansho_hist_dates() if M.in_period(d)]:
        h = pd.read_csv(M.tansho_hist_path(date), dtype={"race_id": str}, encoding="utf-8-sig")
        h["win_odds"] = pd.to_numeric(h["win_odds"], errors="coerce")
        res = M.results([date])
        if res.empty:
            continue
        # 2026-10-12 2回目レビュー対応: 取得が予定時刻より60秒以上遅れたスナップショット(発走後の取得を含む)は使わない。
        # 1レースでも欠けた馬があればそのレースの動きは欠損にする
        h["late"] = (pd.to_datetime(h["fetched_time"]) - pd.to_datetime(h["scheduled_time"])).dt.total_seconds()
        cp = {}
        for lab in ("1時間前", "10分前"):
            x = h[h["checkpoint_label"] == lab].drop_duplicates(["race_id", "umaban"], keep="last")
            x = x[x["late"].between(-60, 60)]
            cp[lab] = x.set_index(["race_id", "umaban"])["win_odds"]
        for rid, g in res.groupby("race_id"):
            if rid not in set(h["race_id"]):
                continue
            g = g[g["odds"] > 0].copy()
            if len(g) < 6 or g["fin"].isna().all():
                continue
            g["p"] = M.norm_p(g["odds"].to_numpy())
            g["P3w"] = M.top3_prob(g["p"].to_numpy(), lam["lambda2"], lam["lambda3"])
            for lab, key in (("1時間前", "p60"), ("10分前", "p10")):
                o = np.array([cp[lab].get((rid, int(u)), np.nan) for u in g["um"]], dtype=float)
                g[key] = M.norm_p(o) if (o > 0).all() else np.nan
            rows.append(g)
    df = pd.concat(rows, ignore_index=True)
    df["win"] = (df["fin"] == 1).astype(float)
    df["top3"] = (df["fin"] <= 3).astype(float)
    df["logp"] = np.log(df["p"])
    df["lP3w"] = np.log(df["P3w"] / (1 - df["P3w"]))
    df["drift60"] = df["logp"] - np.log(df["p60"])
    df["drift10"] = df["logp"] - np.log(df["p10"])
    # 参考(事後、レビュー対応): 動きは人気と強く交絡する(1番人気の大半が「売れた」側)。
    # 同じ人気帯のふだんの動き(帯ごとの中央値)を引いた相対的な動きを作る
    pop = df.groupby("race_id")["p"].rank(ascending=False, method="first")
    df["band"] = pd.cut(pop, [0, 1, 2, 3, 6, 9, 99], labels=["1番人気", "2番人気", "3番人気", "4〜6番人気", "7〜9番人気", "10番人気以下"])
    df["drift_rel"] = df["drift60"] - df.groupby("band", observed=True)["drift60"].transform("median")
    return df


def std_in(df, c):
    d = df.dropna(subset=[c]).copy()
    d[c] = d[c] / d[c].std()
    return d


def roi_by_quintile(d, col="drift60"):
    d = d.copy()
    d["ret"] = np.where(d["win"] == 1, d["odds"] * 100, 0.0)
    d["qbin"] = pd.qcut(d[col], 5, labels=["Q1 売れなかった", "Q2", "Q3", "Q4", "Q5 大きく売れた"])
    out = []
    for q, g in d.groupby("qbin", observed=True):
        out.append({"bin": str(q), "n": len(g), "drift_mean": float(g[col].mean()),
                    "win_rate": float(g["win"].mean()), "mkt_p": float(g["p"].mean()),
                    "roi": float(g["ret"].sum() / (len(g) * 100))})
    return out


def main():
    df = build()
    df.to_pickle(M.OUT / "t2_rows.pkl")
    d60 = std_in(df, "drift60")
    d10 = std_in(df, "drift10")
    drel = std_in(df, "drift_rel")
    res = {"H2": M.evaluate(d60, ["logp"], ["drift60"], "win", "clogit", "H2 勝ち馬 + 1時間前からの動き", n_placebo=200),
           "H2b": M.evaluate(d10, ["logp"], ["drift10"], "win", "clogit", "H2b 勝ち馬 + 10分前からの動き"),
           "H2c": M.evaluate(d60, ["lP3w"], ["drift60"], "top3", "logit", "H2c 3着内 + 1時間前からの動き"),
           "H2_rel": M.evaluate(drel, ["logp"], ["drift_rel"], "win", "clogit", "参考(事後) 人気帯の中で見た動き"),
           "roi_quintile": roi_by_quintile(df.dropna(subset=["drift60"])),
           "roi_quintile_rel": roi_by_quintile(df.dropna(subset=["drift_rel"]), "drift_rel"),
           "fav_share_sold": float((pd.qcut(df["drift60"], 3, labels=False)[df["band"] == "1番人気"] == 2).mean()),
           "dates": sorted(df["date"].unique()),
           "src_count": df.drop_duplicates("race_id")["src"].value_counts().to_dict()}
    (M.OUT / "t2_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for k in ("H2", "H2b", "H2c", "H2_rel"):
        r = res[k]
        print(f"{r['name']}: {r['n_races']}R/{r['n_days']}日 ΔLL {r['dll']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] "
              f"プラセボ最大 {r['placebo_max']:+.2f} 係数 {r['coef']} 合格={r['pass']}")
    for q in res["roi_quintile"]:
        print(q)
    print(res["src_count"])


if __name__ == "__main__":
    main()
