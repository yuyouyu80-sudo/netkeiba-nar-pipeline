# -*- coding: utf-8 -*-
"""予想ファクター充足度マップ(2026-09-02)で「有望そう」に見えた人・組織の指標を、確定単勝オッズへの上乗せで測るふるい分け(2026-10-13)。

マップの数字(騎手リーディング1〜10位の勝率15.0%など)はオッズで調整していない生の勝率。
ここでは各レースの前日までの成績だけで作った指標(結果の先取りなし)を、勝ち馬の条件付きロジット [log p_確定] に足す。
- 騎手・調教師: 直近365日(前日まで)の勝率(縮め済み)と、オッズに対する上振れ (勝ち数 − 市場の勝率の合計)/(出走数+縮め)
- 馬主・生産者: 2011年以降の前日までの通算で同じ2つ
- 外国人騎手(jockey_profile の所属が「海外」)
当てはめ 2017〜2020年、確認 2021〜2025年(1回の分割。事前登録の判定ではなく、次に進めるかのふるい)。
出力: data/jra_pipeline/factor_map_screen_2026_10_13/screen_result.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "jra_model" / "market_pools_2026_10_12"))
import mp_common as M  # noqa: E402

OUT = ROOT / "data" / "jra_pipeline" / "factor_map_screen_2026_10_13"
OUT.mkdir(parents=True, exist_ok=True)
SHRINK = 50.0


def load_runs():
    cols = ["race_id", "surface", "umaban", "finish_pos", "odds_final", "horse_id", "jockey_id", "trainer_id", "owner_id"]
    fr = []
    for y in range(2011, 2026):
        for f in sorted(glob.glob(str(ROOT / "data" / "race_results" / str(y) / f"{y}*.csv"))):
            d = pd.read_csv(f, dtype=str, usecols=lambda c: c in cols)
            d["date"] = Path(f).stem
            fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d = d[~d["surface"].astype(str).str.contains("障")]
    d["fin"] = pd.to_numeric(d["finish_pos"], errors="coerce")
    d["odds"] = pd.to_numeric(d["odds_final"], errors="coerce")
    d = d[d["odds"] > 0].drop_duplicates(["race_id", "umaban"])
    n = d.groupby("race_id")["odds"].transform("size")
    w = d.groupby("race_id")["fin"].transform(lambda s: (s == 1).sum())
    d = d[(n >= 6) & (w >= 1)].copy()
    d["p"] = 1.0 / d["odds"]
    d["p"] = d["p"] / d.groupby("race_id")["p"].transform("sum")
    d["win"] = (d["fin"] == 1).astype(float)
    d["dt"] = pd.to_datetime(d["date"])
    prof = pd.read_parquet(ROOT / "data" / "jra_pipeline" / "finish_v4" / "profile.parquet", columns=["horse_id", "breeder_id"])
    d = d.merge(prof, on="horse_id", how="left")
    return d.reset_index(drop=True)


def pit_stats(d, key, window_days=None):
    """前日までの出走数・勝ち数・市場の勝率の合計(日ごとに集計してから累積/365日窓)。"""
    g = d.dropna(subset=[key]).groupby([key, "dt"]).agg(n=("win", "size"), w=("win", "sum"), ep=("p", "sum")).reset_index()
    g = g.sort_values([key, "dt"])
    if window_days is None:
        for c in ("n", "w", "ep"):
            g[c + "_c"] = g.groupby(key)[c].cumsum() - g[c]  # 当日を含めない
    else:
        parts = []
        for k, x in g.groupby(key, sort=False):
            x = x.set_index("dt")
            r = x[["n", "w", "ep"]].rolling(f"{window_days}D", closed="left").sum().fillna(0.0)
            r.columns = [c + "_c" for c in r.columns]
            parts.append(pd.concat([x[[key]], r], axis=1).reset_index())
        g = pd.concat(parts, ignore_index=True)
    m = d[[key, "dt"]].merge(g[[key, "dt", "n_c", "w_c", "ep_c"]], on=[key, "dt"], how="left")
    base = 0.075
    n = m["n_c"].fillna(0).to_numpy()
    w = m["w_c"].fillna(0).to_numpy()
    ep = m["ep_c"].fillna(0).to_numpy()
    wr = (w + SHRINK * base) / (n + SHRINK)
    ex = (w - ep) / (n + SHRINK)
    return np.log(wr / (1 - wr)), ex, n


def evaluate(df, feats, tr, te, name):
    """当てはめ期間で [logp]+特徴 を推定し、確認期間の ΔLL(行ごと)をブロックブートストラップ。"""
    X0 = df[["logp"]].to_numpy()
    X1 = df[["logp"] + feats].to_numpy()
    y = df["win"].to_numpy()
    rid = df["race_id"].to_numpy()
    w0 = M.fit_clogit(X0[tr], rid[tr], y[tr], lam=0.0)
    w1 = M.fit_clogit(X1[tr], rid[tr], y[tr], lam=0.0)
    dll = M.clogit_ll(X1[te], rid[te], y[te], w1) - M.clogit_ll(X0[te], rid[te], y[te], w0)
    dte = df[te].reset_index(drop=True)
    tot, lo, hi, nb = M.block_ci(dte, dll)
    nr = int(dte["race_id"].nunique())
    # 確認期間だけで当て直した係数(大きさの目安、レース単位ブートストラップ 50回)
    wt = M.fit_clogit(X1[te], rid[te], y[te], lam=0.0)
    rng = np.random.default_rng(0)
    races = dte["race_id"].to_numpy()
    ur, inv = np.unique(races, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(ur)))
    ends = np.r_[starts[1:], len(order)]
    Xte, yte = X1[te], y[te]
    bs = []
    for _ in range(50):
        pick = rng.integers(0, len(ur), len(ur))
        idx = np.concatenate([order[starts[k]:ends[k]] for k in pick])
        tag = np.repeat(np.arange(len(pick)), ends[pick] - starts[pick])
        bs.append(M.fit_clogit(Xte[idx], tag, yte[idx], lam=0.0)[1:])
    bs = np.array(bs)
    sd = df.loc[te, feats].std().to_numpy()
    ci = {f: [float(np.exp(np.percentile(bs[:, j], 2.5) * sd[j])), float(np.exp(np.percentile(bs[:, j], 97.5) * sd[j]))]
          for j, f in enumerate(feats)}
    return {"name": name, "feats": feats, "n_races_test": nr, "dll": tot, "lo": lo, "hi": hi, "dll_per_race": tot / nr,
            "coef_train": dict(zip(feats, map(float, w1[1:]))), "coef_test": dict(zip(feats, map(float, wt[1:]))),
            "or_per_sd_test_ci": ci, "pass_screen": bool(lo > 0)}


def roi_quintiles(dte, f):
    q = pd.qcut(dte[f].rank(method="first"), 5, labels=False)
    out = []
    for k in range(5):
        g = dte[q == k]
        out.append({"q": k + 1, "n": len(g), "win": float(g["win"].mean()), "mkt_p": float(g["p"].mean()),
                    "roi": float((g["win"] * g["odds"]).sum() / len(g))})
    return out


def main():
    d = load_runs()
    print("runs", len(d), "races", d["race_id"].nunique())
    for key, win, pre in (("jockey_id", 365, "j"), ("trainer_id", 365, "t"), ("owner_id", None, "o"), ("breeder_id", None, "b")):
        lw, ex, n = pit_stats(d, key, win)
        d[pre + "_wr"], d[pre + "_ex"], d[pre + "_n"] = lw, ex, n
        print(key, "done", float(np.mean(n > 0)))
    jp = {}
    for f in glob.glob(str(ROOT / "data" / "jockey_profile" / "*.csv")):
        x = pd.read_csv(f, dtype=str, nrows=1)
        if len(x):
            jp[str(x["id"].iloc[0]).zfill(5)] = str(x["affiliation_area"].iloc[0])
    d["j_foreign"] = d["jockey_id"].astype(str).str.zfill(5).map(jp).eq("海外").astype(float)
    d["logp"] = np.log(d["p"])
    d["year"] = d["date"].str[:4].astype(int)
    d["venue"] = d["race_id"].str[4:6]
    d = d[d["year"] >= 2017].reset_index(drop=True)
    tr = (d["year"] <= 2020).to_numpy()
    te = (d["year"] >= 2021).to_numpy()
    feats_all = ["j_wr", "j_ex", "t_wr", "t_ex", "o_wr", "o_ex", "b_wr", "b_ex", "j_foreign"]
    for f in feats_all[:-1]:
        d[f] = (d[f] - d[f].mean()) / d[f].std()
    d.loc[d["breeder_id"].isna(), ["b_wr", "b_ex"]] = 0.0
    res = {"n_races_train": int(d.loc[tr, "race_id"].nunique()), "n_races_test": int(d.loc[te, "race_id"].nunique()),
           "breeder_coverage": float(d["breeder_id"].notna().mean()), "foreign_share": float(d["j_foreign"].mean()), "tests": []}
    for name, fs in (("騎手 勝率", ["j_wr"]), ("騎手 上振れ", ["j_ex"]), ("調教師 勝率", ["t_wr"]), ("調教師 上振れ", ["t_ex"]),
                     ("馬主 勝率", ["o_wr"]), ("馬主 上振れ", ["o_ex"]), ("生産者 勝率", ["b_wr"]), ("生産者 上振れ", ["b_ex"]),
                     ("外国人騎手", ["j_foreign"]), ("全部", feats_all)):
        r = evaluate(d, fs, tr, te, name)
        res["tests"].append(r)
        print(f'{name}: 確認 {r["n_races_test"]}R ΔLL {r["dll"]:+.1f} [{r["lo"]:+.1f}, {r["hi"]:+.1f}] /R {r["dll_per_race"]:+.4f} '
              f'係数(当て)={ {k: round(v, 3) for k, v in r["coef_train"].items()} } OR/SD={ {k: [round(a, 3) for a in v] for k, v in r["or_per_sd_test_ci"].items()} }', flush=True)
    dte = d[te]
    res["roi_quintiles"] = {f: roi_quintiles(dte, f) for f in ("j_wr", "j_ex", "t_ex", "o_ex", "b_ex")}
    (OUT / "screen_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
