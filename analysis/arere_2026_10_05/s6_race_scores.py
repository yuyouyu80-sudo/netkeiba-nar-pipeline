# -*- coding: utf-8 -*-
"""開催日ごとの 荒れ指数・荒れない指数・予想3連複払戻額 を計算する(予想ページ表示用、2026-10-05追加)。

オッズの出どころ(どちらも発走前に分かる情報だけを使う):
  - 結果取得済みの日(data/race_results/{year}/{date}.csv がある): 確定単勝オッズ(odds_final)
  - 結果未取得の日: data/odds_history_tansho/{year}/{date}.csv の各レース最新チェックポイントの単勝オッズ
    (無い馬は newspaper の bias_win_odds)。レース条件は race_names_{date}.csv(予想パイプラインのscratchpad)と
    data/newspaper/{race_id}.csv(性齢・斤量)。馬場状態・天候は未確定なので「不明」として扱う。
モデル: out/s1_models.pkl・out_katai/s1_models.pkl(条件+オッズ)と out_payout/payout_models.pkl。いずれも2011〜2022年で学習。

使い方:
  python s6_race_scores.py 20261003 20261004 [--race-names-dir DIR]
出力: out_scores/scores_{date}.json  {race_id: {arere, katai, pay25, pay50, pay75, odds_source, n_horses, category}}
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_build_races as B  # noqa: E402
import s2_stats_model as S  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out_scores"
OUT.mkdir(exist_ok=True)
DEFAULT_RN_DIR = Path(r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
                      r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad")
MODEL = "条件+オッズ"


def records_from_results(date):
    p = ROOT / "data" / "race_results" / date[:4] / f"{date}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, dtype=str)
    recs = []
    for rid, g in d.groupby("race_id", sort=False):
        r = B.race_record(rid, g)
        if r is not None and "fav_odds" in r:
            r["odds_source"] = "確定オッズ"
            recs.append(r)
    return recs


def records_pre_race(date, rn_dir):
    rn = pd.read_csv(Path(rn_dir) / f"race_names_{date}.csv", dtype=str, encoding="utf-8-sig")
    oh_p = ROOT / "data" / "odds_history_tansho" / date[:4] / f"{date}.csv"
    oh = pd.read_csv(oh_p, dtype=str, encoding="utf-8-sig") if oh_p.exists() else None
    recs = []
    for _, r in rn.iterrows():
        rid = r["race_id"]
        name = str(r["race_name"])
        if "障害" in name or "ジャンプ" in name or str(r["surface"]) not in ("芝", "ダ"):
            continue
        npth = ROOT / "data" / "newspaper" / f"{rid}.csv"
        if not npth.exists():
            continue
        nw = pd.read_csv(npth, dtype=str)
        odds = {}
        if oh is not None:
            x = oh[oh["race_id"] == rid]
            if len(x):
                x = x.sort_values("fetched_time").groupby("umaban").tail(1)
                odds = {int(u): B_num(o) for u, o in zip(x["umaban"], x["win_odds"])}
        src = "発走前オッズ" if odds else "馬柱取得時オッズ"
        g = pd.DataFrame({
            "race_id": rid, "race_name": name, "surface": r["surface"], "distance_m": r["distance_m"],
            "race_date": f"{date[:4]}-{date[4:6]}-{date[6:]}", "weather": "不明", "going": "不明",
            "finish_pos": "", "popularity": np.nan,
            "sex_age": nw["bias_sex_age"].values, "kinryo": nw["bias_weight_carried"].values,
            "odds_final": [odds.get(int(u), B_num(b)) for u, b in zip(nw["umaban"], nw["bias_win_odds"])],
        })
        g = g[pd.to_numeric(g["odds_final"], errors="coerce") > 0]  # 取消・除外(オッズ無し)を除く
        rec = B.race_record(rid, g)
        if rec is not None and "fav_odds" in rec:
            rec["odds_source"] = src
            recs.append(rec)
    return recs


def B_num(x):
    try:
        v = float(str(x).replace(",", ""))
        return v if v > 0 else np.nan
    except ValueError:
        return np.nan


def score(recs):
    df = pd.DataFrame(recs)
    X = S.add_bins(S.prep_numeric(df))
    m_a = pickle.load(open(HERE / "out" / "s1_models.pkl", "rb"))
    m_k = pickle.load(open(HERE / "out_katai" / "s1_models.pkl", "rb"))
    pm = pickle.load(open(HERE / "out_payout" / "payout_models.pkl", "rb"))
    df["arere"] = np.nan
    df["katai"] = np.nan
    for cat in ("一般", "新馬", "未勝利"):
        msk = (df["category"] == cat).to_numpy()
        if msk.any():
            df.loc[msk, "arere"] = m_a[(cat, MODEL)].predict_proba(X[msk])[:, 1]
            df.loc[msk, "katai"] = m_k[(cat, MODEL)].predict_proba(X[msk])[:, 1]
    F = X[pm["num_cols"]].copy()
    for c in pm["cat_cols"]:
        F[c] = pd.Categorical(X[c].astype(str), categories=pm["cat_levels"][c])
    for q, m in pm["models"].items():
        df[f"pay{int(q * 100)}"] = np.round(np.exp(m.predict(F)), -1)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dates", nargs="+")
    ap.add_argument("--race-names-dir", default=str(DEFAULT_RN_DIR))
    ap.add_argument("--pre-race", action="store_true", help="結果があっても発走前オッズで計算する")
    a = ap.parse_args()
    for date in a.dates:
        recs = None if a.pre_race else records_from_results(date)
        if recs is None:
            recs = records_pre_race(date, a.race_names_dir)
        if not recs:
            print(date, "対象レースなし")
            continue
        df = score(recs)
        out = {r.race_id: dict(arere=round(float(r.arere), 4), katai=round(float(r.katai), 4),
                               pay25=float(r.pay25), pay50=float(r.pay50), pay75=float(r.pay75),
                               odds_source=r.odds_source, n_horses=int(r.n_horses), category=r.category)
               for r in df.itertuples()}
        json.dump(out, open(OUT / f"scores_{date}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(date, len(out), "races", df["odds_source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
