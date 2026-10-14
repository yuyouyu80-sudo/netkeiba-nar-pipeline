# -*- coding: utf-8 -*-
"""クッション値を予想に盛り込むための下調べ(2026-10-13、手元のデータだけ)。

A. 前もって当てられるか: 開催日の朝(8:20頃測定・当日朝に JRA が発表)の値を、
   金曜(9:30測定・金曜に発表)や前日の値、競馬場・月の平均でどれだけ当てられるか(平均絶対誤差)。
B. 予想に効くか: 芝のレースで、馬の「前に行く度合い」(前日までの過去走の4角位置の平均、結果の先取りなし)と
   当日朝のクッション値(競馬場ごとの偏差)の掛け算を、勝ち馬の条件付きロジット [log p_確定] に足す。
入力: data/jra_baba/{year}/*.csv(2025〜2026年)、data/race_results(2024〜2026年)
出力: data/jra_pipeline/cushion_2026_10_13/step1_result.json
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
from functools import partial  # noqa: E402

M.coef_boot = partial(M.coef_boot, n_boot=200)  # 下調べなので係数のブートストラップは200回

OUT = ROOT / "data" / "jra_pipeline" / "cushion_2026_10_13"
OUT.mkdir(parents=True, exist_ok=True)
VENUE = {"札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05", "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"}


def load_baba():
    fr = [pd.read_csv(f, dtype=str) for f in sorted(glob.glob(str(ROOT / "data" / "jra_baba" / "*" / "*.csv")))]
    b = pd.concat(fr, ignore_index=True)
    b["cv"] = pd.to_numeric(b["cushion_value"], errors="coerce")
    for c in ("moisture_turf_goal_pct", "moisture_dirt_goal_pct"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b["dt"] = pd.to_datetime(b["date"])
    b["vc"] = b["venue"].map(VENUE)
    b["raceday"] = b["day_label"].notna() & b["day_label"].astype(str).str.contains("日")
    return b.sort_values(["vc", "dt"]).reset_index(drop=True)


def part_a(b):
    r = b[b["raceday"] & b["cv"].notna()].copy()
    out = {"n_racedays": int(len(r))}
    # 直前の測定(同じ競馬場、前日まで。金曜または前の開催日)
    prev = b[b["cv"].notna()][["vc", "dt", "cv", "weekday"]].rename(columns={"dt": "pdt", "cv": "pcv", "weekday": "pwd"})
    m = pd.merge_asof(r.sort_values("dt"), prev.sort_values("pdt"), left_on="dt", right_on="pdt", by="vc",
                      allow_exact_matches=False, direction="backward")
    m["gap"] = (m["dt"] - m["pdt"]).dt.days
    m = m[m["gap"] <= 3]
    err = (m["cv"] - m["pcv"]).abs()
    out["prev_measure"] = {"n": int(len(m)), "mae": float(err.mean()), "within_0_3": float((err <= 0.3).mean()),
                           "within_0_5": float((err <= 0.5).mean()), "mean_change": float((m["cv"] - m["pcv"]).mean()),
                           "by_prev_weekday": {k: [int(len(g)), float((g["cv"] - g["pcv"]).abs().mean())] for k, g in m.groupby("pwd")}}
    # 競馬場×月の平均だけ(天気・前の測定を使わない)
    r["month"] = r["dt"].dt.month
    r["vm_mean"] = r.groupby(["vc", "month"])["cv"].transform("mean")
    out["venue_month_mean_mae"] = float((r["cv"] - r["vm_mean"]).abs().mean())
    out["overall_sd"] = float(r["cv"].std())
    out["by_venue"] = {v: [int(len(g)), float(g["cv"].mean()), float(g["cv"].std())] for v, g in r.groupby("venue")}
    # 大きく動いた日(前の測定から0.5以上)
    big = m[(m["cv"] - m["pcv"]).abs() >= 0.5]
    out["big_moves"] = big[["venue", "date", "pwd", "pcv", "cv"]].head(30).astype(str).values.tolist()
    return out


def load_runs():
    cols = ["race_id", "surface", "umaban", "finish_pos", "odds_final", "horse_id", "passing_order"]
    fr = []
    for y in (2024, 2025, 2026):
        for f in sorted(glob.glob(str(ROOT / "data" / "race_results" / str(y) / f"{y}*.csv"))):
            d = pd.read_csv(f, dtype=str, usecols=lambda c: c in cols)
            d["date"] = Path(f).stem
            fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d["fin"] = pd.to_numeric(d["finish_pos"], errors="coerce")
    d["odds"] = pd.to_numeric(d["odds_final"], errors="coerce")
    d["c4"] = pd.to_numeric(d["passing_order"].astype(str).str.split("-").str[-1], errors="coerce")
    d["nrun"] = d.groupby("race_id")["umaban"].transform("size")
    d["c4rel"] = (d["c4"] - 1) / (d["nrun"] - 1)  # 0=先頭、1=最後方
    d["dt"] = pd.to_datetime(d["date"])
    return d


def part_b(b, d):
    # 前日までの過去走の4角位置(相対)の平均 = 前に行く度合い(小さいほど前)
    h = d.dropna(subset=["c4rel"]).sort_values(["horse_id", "dt"])
    h["cum"] = h.groupby("horse_id")["c4rel"].cumsum() - h["c4rel"]
    h["cnt"] = h.groupby("horse_id").cumcount()
    h = h[h["cnt"] > 0]
    h["front_prior"] = h["cum"] / h["cnt"]
    last = h[["horse_id", "dt", "front_prior", "cnt"]]
    x = d[(d["dt"] >= "2025-01-01") & ~d["surface"].astype(str).str.contains("障") & d["surface"].astype(str).str.contains("芝")].copy()
    x = x[x["odds"] > 0]
    x = pd.merge_asof(x.sort_values("dt"), last.sort_values("dt").rename(columns={"dt": "hdt"}), left_on="dt", right_on="hdt",
                      by="horse_id", allow_exact_matches=False, direction="backward")
    x["vc"] = x["race_id"].str[4:6]
    bd = b[b["raceday"]][["vc", "dt", "cv", "moisture_turf_goal_pct"]]
    x = x.merge(bd, on=["vc", "dt"], how="left")
    x = x[x["cv"].notna()]
    n = x.groupby("race_id")["odds"].transform("size")
    w = x.groupby("race_id")["fin"].transform(lambda s: (s == 1).sum())
    x = x[(n >= 6) & (w >= 1)].copy()
    x["p"] = 1 / x["odds"]
    x["p"] /= x.groupby("race_id")["p"].transform("sum")
    x["logp"] = np.log(x["p"])
    x["win"] = (x["fin"] == 1).astype(float)
    x["top3"] = (x["fin"] <= 3).astype(float)
    x["front"] = -(x["front_prior"].fillna(x["front_prior"].mean()))  # 大きいほど前に行く
    x["front"] = (x["front"] - x["front"].mean()) / x["front"].std()
    x["cv_dev"] = x["cv"] - x.groupby("vc")["cv"].transform("mean")  # 競馬場ごとの偏差(大きいほど硬い)
    x["cv_dev"] /= x["cv_dev"].std()
    x["mo_dev"] = x["moisture_turf_goal_pct"] - x.groupby("vc")["moisture_turf_goal_pct"].transform("mean")
    x["mo_dev"] = (x["mo_dev"] / x["mo_dev"].std()).fillna(0)
    x["front_x_cv"] = x["front"] * x["cv_dev"]
    x["front_x_mo"] = x["front"] * x["mo_dev"]
    x["venue"] = x["vc"]
    # 交差検証は月ごとに抜く(開催日ごとだと当て直しが多すぎるため)。ブロックは月×競馬場
    x["day"] = x["date"].astype(str)
    x["date"] = x["day"].str[:6]
    out = {"n_races": int(x["race_id"].nunique()), "n_days": int(x["day"].nunique()), "dates": [x["day"].min(), x["day"].max()]}
    # 記述: 4角先頭の馬の3着内率をクッション値の三分位で
    lead = x[x["c4"] == 1]
    q = pd.qcut(lead["cv_dev"], 3, labels=["柔らかめ", "標準", "硬め"])
    out["leader_top3_by_cv"] = {str(k): [int(len(g)), float(g["top3"].mean()), float(g["p"].mean())] for k, g in lead.groupby(q, observed=True)}
    out["H_front_x_cv"] = M.evaluate(x, ["logp", "front"], ["front_x_cv"], "win", "clogit", "勝ち馬 [市場+前に行く度合い] + 前×クッション値", n_placebo=20)
    out["H_front_x_mo"] = M.evaluate(x, ["logp", "front"], ["front_x_mo"], "win", "clogit", "参考 [市場+前] + 前×含水率", n_placebo=20)
    out["H_top3"] = M.evaluate(x, ["lP3", "front"], ["front_x_cv"], "top3", "logit", "3着内 [市場+前] + 前×クッション値", n_placebo=20) \
        if "lP3" in x else None
    for k in ("H_front_x_cv", "H_front_x_mo"):
        r = out[k]
        r.pop("placebo_values", None)
        print(f'{r["name"]}: {r["n_races"]}R ΔLL {r["dll"]:+.2f} [{r["lo"]:+.2f}, {r["hi"]:+.2f}] プラセボ最大 {r["placebo_max"]:+.2f} '
              f'係数 { {c: round(v, 3) for c, v in r["coef"].items()} } 合格={r["pass"]}', flush=True)
    return out


def main():
    b = load_baba()
    a = part_a(b)
    print(json.dumps({k: v for k, v in a.items() if k != "big_moves"}, ensure_ascii=False, indent=1))
    d = load_runs()
    bres = part_b(b, d)
    print(json.dumps(bres["leader_top3_by_cv"], ensure_ascii=False))
    (OUT / "step1_result.json").write_text(json.dumps({"A": a, "B": bres}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
