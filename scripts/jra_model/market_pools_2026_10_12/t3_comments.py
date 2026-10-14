# -*- coding: utf-8 -*-
"""T3: 厩舎コメント本文を言語モデル(この会話の Claude)が読んだ点数の上乗せ。事前登録: PREREG_market_pools_2026_10_12.md

点数(読む前に固定): E 期待度 −2〜+2、U 不安 0/1。馬名・レースID・日付を伏せ、無作為な通し番号で読んだ。
ラベルの正本: data/jra_pipeline/market_pools_2026_10_12/comment_labels.csv(cid, race_id, umaban, E, U)

主 H3: 勝ち馬の条件付きロジット [log p, 評価コード01, 03, コメント有無] + E
副 H3b: [log p] + E・U / H3r: [log p] + 評価コード(既存の物差しの確認) / H3c: 3着内 [logit P3w, 評価コード, 有無] + E
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

LABEL_CSV = M.OUT / "comment_labels.csv"
EMAP = {"A": -2, "B": -1, "C": 0, "D": 1, "E": 2}


def build_labels(label_dir):
    """scratchpad の labels_XX.txt(10件ごとの行: 先頭cid E文字列 U文字列)と key.csv を結合して正本 CSV を作る。"""
    label_dir = Path(label_dir)
    key = pd.read_csv(label_dir / "key.csv", dtype={"race_id": str})
    rec = {}
    for f in sorted(label_dir.glob("labels_*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            c0, es, us = line.split()
            c0 = int(c0)
            if len(es) != len(us):
                raise ValueError(f"{f.name} {c0}: E と U の長さが違う")
            for k, (e, u) in enumerate(zip(es, us)):
                if c0 + k in rec:
                    raise ValueError(f"cid {c0 + k} が重複")
                rec[c0 + k] = (EMAP[e], int(u))
    miss = sorted(set(key["cid"]) - set(rec))
    extra = sorted(set(rec) - set(key["cid"]))
    if miss or extra:
        raise ValueError(f"欠け {miss[:10]}… / 余り {extra[:10]}…")
    key["E"] = key["cid"].map(lambda c: rec[c][0])
    key["U"] = key["cid"].map(lambda c: rec[c][1])
    key[["cid", "race_id", "umaban", "E", "U", "stable_comment"]].to_csv(LABEL_CSV, index=False, encoding="utf-8-sig")
    return key


def build():
    lab = pd.read_csv(LABEL_CSV, dtype={"race_id": str})
    lab["um"] = pd.to_numeric(lab["umaban"], errors="coerce")
    lam = M.fit_lambda()
    news = []
    for f in sorted(glob.glob(str(M.ROOT / "data" / "newspaper" / "2026*.csv"))):
        d = pd.read_csv(f, dtype=str, usecols=lambda c: c in ("race_id", "umaban", "stable_comment_rating_code"))
        news.append(d)
    nw = pd.concat(news, ignore_index=True)
    nw["um"] = pd.to_numeric(nw["umaban"], errors="coerce")
    nw["rc"] = nw["stable_comment_rating_code"].astype(str).str.strip()
    nw = nw.drop_duplicates(["race_id", "um"]).set_index(["race_id", "um"])["rc"]
    rids = set(nw.index.get_level_values(0))
    dates = sorted({Path(p).stem for p in glob.glob(str(M.ROOT / "data" / "race_results" / "2026" / "2026*.csv"))}
                   | {Path(p).stem for p in glob.glob(str(M.ROOT / "data" / "intraday_results" / "2026*.csv"))
                      if not p.endswith("manifest.json")})
    res = M.results([d for d in dates if d.isdigit() and M.in_period(d)])
    res = res[res["race_id"].isin(rids)]
    lmap = lab.set_index(["race_id", "um"])[["E", "U"]]
    rows = []
    for rid, g in res.groupby("race_id"):
        g = g[g["odds"] > 0].copy()
        if len(g) < 6 or g["fin"].isna().all():
            continue
        g["p"] = M.norm_p(g["odds"].to_numpy())
        g["P3w"] = M.top3_prob(g["p"].to_numpy(), lam["lambda2"], lam["lambda3"])
        keys = list(zip([rid] * len(g), g["um"]))
        g["rc"] = [nw.get(k, "nan") for k in keys]
        lk = lmap.reindex(keys)
        g["hasc"] = lk["E"].notna().astype(float).to_numpy()
        g["E"] = lk["E"].fillna(0).to_numpy()
        g["U"] = lk["U"].fillna(0).to_numpy()
        rows.append(g)
    df = pd.concat(rows, ignore_index=True)
    df["win"] = (df["fin"] == 1).astype(float)
    df["top3"] = (df["fin"] <= 3).astype(float)
    df["logp"] = np.log(df["p"])
    df["lP3w"] = np.log(df["P3w"] / (1 - df["P3w"]))
    df["r01"] = (df["rc"] == "01").astype(float)
    df["r03"] = (df["rc"] == "03").astype(float)
    return df


def describe(df):
    c = df[df["hasc"] == 1]
    out = []
    for e, g in c.groupby("E"):
        out.append({"E": int(e), "n": len(g), "win": float(g["win"].mean()), "mkt_p": float(g["p"].mean()),
                    "top3": float(g["top3"].mean()), "mkt_P3": float(g["P3w"].mean()),
                    "roi_win": float((np.where(g["win"] == 1, g["odds"], 0)).sum() / len(g))})
    cross = pd.crosstab(c["rc"], c["E"]).to_dict()
    return out, {str(k): {str(kk): int(vv) for kk, vv in v.items()} for k, v in cross.items()}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--build-labels":
        build_labels(sys.argv[2])
        print("正本を作成:", LABEL_CSV)
        return
    df = build()
    df.to_pickle(M.OUT / "t3_rows.pkl")
    # 2026-10-12 レビュー対応: コメントの有無はレース単位(全頭あり/全頭なし)で、条件付きロジットでは推定できない。
    # コメント掲載レースだけに絞り、有無の列は外す(実質の標本は掲載レースのみ)
    n_all = int(df["race_id"].nunique())
    df = df[df.groupby("race_id")["hasc"].transform("max") == 1].reset_index(drop=True)
    base = ["logp", "r01", "r03"]
    res = {"H3": M.evaluate(df, base, ["E"], "win", "clogit", "H3 勝ち馬 [市場+評価コード] + 期待度E", n_placebo=200),
           "H3b": M.evaluate(df, ["logp"], ["E", "U"], "win", "clogit", "H3b 勝ち馬 [市場] + E・U"),
           "H3r": M.evaluate(df, ["logp"], ["r01", "r03"], "win", "clogit", "H3r 勝ち馬 [市場] + 評価コード"),
           "H3c": M.evaluate(df, ["lP3w", "r01", "r03"], ["E"], "top3", "logit", "H3c 3着内 + 期待度E")}
    desc, cross = describe(df)
    by_rc = []
    for rc, g in df.groupby("rc"):
        by_rc.append({"rc": rc, "n": len(g), "win": float(g["win"].mean()), "mkt_p": float(g["p"].mean()),
                      "roi_win": float(np.where(g["win"] == 1, g["odds"], 0).sum() / len(g))})
    res.update({"by_E": desc, "rating_x_E": cross, "by_rating": by_rc, "n_races_all_results": n_all,
                "race_no_count": df.drop_duplicates("race_id")["race_id"].str[-2:].value_counts().sort_index().to_dict(),
                "n_comment_rows": int(df["hasc"].sum()),
                "dates": [str(df["date"].min()), str(df["date"].max())],
                "src_count": df.drop_duplicates("race_id")["src"].value_counts().to_dict()})
    (M.OUT / "t3_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for k in ("H3", "H3b", "H3r", "H3c"):
        r = res[k]
        print(f"{r['name']}: {r['n_races']}R/{r['n_days']}日 ΔLL {r['dll']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] "
              f"プラセボ最大 {r['placebo_max']:+.2f} 係数 {r['coef']} 合格={r['pass']}")
    for d in desc:
        print(d)
    print(cross)
    print("コメントあり行", res["n_comment_rows"], res["dates"], res["src_count"])


if __name__ == "__main__":
    main()
