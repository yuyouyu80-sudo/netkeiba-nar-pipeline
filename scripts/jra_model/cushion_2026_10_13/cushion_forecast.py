# -*- coding: utf-8 -*-
"""手順2: 開催日の朝のクッション値(副: 芝・ダートのゴール前含水率)を前夜までの情報で予想する(2026-10-13)。
事前登録: PREREG_cushion_2026_10_13.md

予想の形: 目的 = 直前の測定値からの変化(当日朝 − 直前の測定)。線形(リッジ)。
前夜版の特徴: 直前の測定値と測定からの日数・直前の含水率・競馬場・月(季節)・コース(A〜D)と変更・開催の何日目か・曜日・
            前日のその競馬場の天気と馬場状態(レース結果の欄)。天気(アメダス)が届いたら WEATHER_COLS に足す。
評価: 開催回を1つずつ抜く交差検証。基準 (a) 直前の測定値そのまま、(b) 競馬場×月の平均(これも抜いた開催回を除いて計算)。
出力: data/jra_pipeline/cushion_2026_10_13/forecast_result.json、forecast_oof.csv(開催日ごとの予想値)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "jra_pipeline" / "cushion_2026_10_13"
OUT.mkdir(parents=True, exist_ok=True)
VENUE = {"札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05", "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"}
GOING = {"良": 0, "稍": 1, "稍重": 1, "重": 2, "不": 3, "不良": 3}
WEATHER_COLS = []  # --weather で W_COLS を入れる(2026-10-14 追補2)
W_COLS = ["w_rain_sum", "w_rain_max", "w_temp_mean", "w_sun_sum", "w_wind_mean"]
PERIODS = {"dev": ("2025-01-01", "2026-12-31"), "confirm": ("2020-01-01", "2024-12-31")}
LAMS = {"cushion": 30.0, "moist_turf": 10.0, "moist_dirt": 30.0}  # 追補1で凍結(天気なし版)
TARGETS = {"cushion": ("cv", "lv"), "moist_turf": ("mt", "lmt"), "moist_dirt": ("md", "lmd")}


def race_days():
    """全期間の開催日(競馬場ごと)と、その開催の何日目か(race_id の9〜10桁目)。レース結果のファイルから作り、キャッシュする。"""
    cache = OUT / "race_days.parquet"
    files = sorted(glob.glob(str(ROOT / "data" / "race_results" / "20[2]*" / "*.csv")))
    if cache.exists() and cache.stat().st_mtime > max(Path(f).stat().st_mtime for f in files[-5:]):
        return pd.read_parquet(cache)
    rows = []
    for f in files:
        ids = pd.read_csv(f, dtype=str, usecols=["race_id"])["race_id"].dropna().unique()
        for rid in {r[:10] for r in ids if len(r) == 12}:
            rows.append((Path(f).stem, rid[4:6], int(rid[8:10])))
    r = pd.DataFrame(rows, columns=["date", "vc", "dayno"]).drop_duplicates(["date", "vc"])
    r["dt"] = pd.to_datetime(r["date"])
    r[["dt", "vc", "dayno"]].to_parquet(cache, index=False)
    return r[["dt", "vc", "dayno"]]


def load_baba(period=None):
    fr = [pd.read_csv(f, dtype=str) for f in sorted(glob.glob(str(ROOT / "data" / "jra_baba" / "*" / "*.csv")))]
    b = pd.concat(fr, ignore_index=True)
    b["dt"] = pd.to_datetime(b["date"])
    if period:
        lo, hi = PERIODS[period]
        b = b[(b["dt"] >= lo) & (b["dt"] <= hi)]
    b = b.drop_duplicates(["venue", "date"], keep="last")
    b["cv"] = pd.to_numeric(b["cushion_value"], errors="coerce")
    b["mt"] = pd.to_numeric(b["moisture_turf_goal_pct"], errors="coerce")
    b["md"] = pd.to_numeric(b["moisture_dirt_goal_pct"], errors="coerce")
    b["vc"] = b["venue"].map(VENUE)
    # 2026-10-14 追補2: 開催日と何日目かはレース結果(race_id)から(旧レイアウトは金曜にも開催の表記が付くため)
    b = b.merge(race_days(), on=["dt", "vc"], how="left")
    b["raceday"] = b["dayno"].notna()
    b["kai_key"] = b["year"].astype(str) + "_" + b["vc"] + "_" + b["kai"].astype(str)
    # 測定の時刻(旧レイアウトは無いので 金曜 9:30・それ以外 8:00 とみなす)
    t = b["cushion_time"].astype(str).where(b["cushion_time"].notna(), np.where(b["weekday"].astype(str) == "金曜日", "09:30", "08:00"))
    b["mdt"] = pd.to_datetime(b["date"] + " " + t.astype(str), errors="coerce")
    return b.sort_values(["vc", "dt"]).reset_index(drop=True)


def prev_day_track(dates_venues):
    """前日のその競馬場のレース結果から、最終レースの天気・馬場状態(芝・ダート別)。前日に開催が無ければ欠損。"""
    rows = []
    for d in sorted({x[0] for x in dates_venues}):
        pd_ = (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y%m%d")
        p = ROOT / "data" / "race_results" / pd_[:4] / f"{pd_}.csv"
        if not p.exists():
            continue
        r = pd.read_csv(p, dtype=str, usecols=["race_id", "surface", "weather", "going", "start_time"]).drop_duplicates("race_id")
        r["vc"] = r["race_id"].str[4:6]
        r = r[~r["surface"].astype(str).str.contains("障")].sort_values("start_time")
        for vc, g in r.groupby("vc"):
            last = g.iloc[-1]
            turf = g[g["surface"].astype(str).str.contains("芝")]
            dirt = g[g["surface"].astype(str).str.contains("ダ")]
            w = str(last["weather"])
            rows.append({"dt": pd.Timestamp(d), "vc": vc, "pv_rain": float(("雨" in w)), "pv_snow": float(("雪" in w)),
                         "pv_cloud": float(("曇" in w)),
                         "pv_going_t": GOING.get(str(turf.iloc[-1]["going"]) if len(turf) else "", np.nan),
                         "pv_going_d": GOING.get(str(dirt.iloc[-1]["going"]) if len(dirt) else "", np.nan),
                         "pv_rained_any": float(g["weather"].astype(str).str.contains("雨").any())})
    return pd.DataFrame(rows)


def build(b):
    t = b[b["raceday"]].copy()
    meas = b[["vc", "dt", "cv", "mt", "md", "turf_course_variant", "weekday", "mdt"]].rename(
        columns={"dt": "ldt", "cv": "lv", "mt": "lmt", "md": "lmd", "turf_course_variant": "lvar", "weekday": "lwd", "mdt": "lmdt"})
    t = pd.merge_asof(t.sort_values("dt"), meas.sort_values("ldt"), left_on="dt", right_on="ldt", by="vc",
                      allow_exact_matches=False, direction="backward")
    t["gap"] = (t["dt"] - t["ldt"]).dt.days
    t = t[t["gap"].between(1, 7)].copy()
    pv = prev_day_track(list(zip(t["dt"].dt.strftime("%Y%m%d"), t["vc"])))
    t = t.merge(pv, on=["dt", "vc"], how="left")
    t["has_pv"] = t["pv_rain"].notna().astype(float)
    for c in ("pv_rain", "pv_snow", "pv_cloud", "pv_rained_any", "pv_going_t", "pv_going_d"):
        t[c] = t[c].fillna(0.0)
    t["month"] = t["dt"].dt.month
    t["msin"], t["mcos"] = np.sin(2 * np.pi * t["month"] / 12), np.cos(2 * np.pi * t["month"] / 12)
    t["var_changed"] = (t["turf_course_variant"].astype(str) != t["lvar"].astype(str)).astype(float)
    for v in "BCD":
        t[f"var_{v}"] = (t["turf_course_variant"].astype(str) == v).astype(float)
    t["from_fri"] = (t["lwd"].astype(str) == "金曜日").astype(float)
    for vname, vc in VENUE.items():
        t[f"v_{vc}"] = (t["vc"] == vc).astype(float)
    return t.reset_index(drop=True)


_WX = {}


def _wx(vc):
    """競馬場ごとの1時間値(雨=最寄りの点、気温・風・日照=気象用の点、無ければ最寄り)。累積和で窓の合計を引けるようにする。"""
    if vc in _WX:
        return _WX[vc]
    def read(dirname):
        fs = sorted(glob.glob(str(ROOT / "data" / "jma_amedas" / dirname / "*.csv")))
        if not fs:
            return None
        x = pd.concat([pd.read_csv(f, dtype=str, encoding="utf-8-sig") for f in fs], ignore_index=True)
        x["t"] = pd.to_datetime(x["datetime"], errors="coerce")
        for c in ("precip_mm", "temp_c", "wind_ms", "sunshine_h"):
            x[c] = pd.to_numeric(x[c], errors="coerce")
        return x.dropna(subset=["t"]).drop_duplicates("t").sort_values("t").set_index("t")
    pr = read(vc)
    we = read(f"{vc}_weather")
    if we is None:
        we = pr
    idx = pr.index.union(we.index)
    rain = pr["precip_mm"].reindex(idx)
    temp, wind, sun = (we[c].reindex(idx) for c in ("temp_c", "wind_ms", "sunshine_h"))
    tt = idx.values.astype("datetime64[ns]")
    arr = {"t": tt,
           "rain": np.r_[0, np.nancumsum(rain.fillna(0).to_numpy())], "rain_n": np.r_[0, np.cumsum(rain.notna().to_numpy())],
           "rain_v": rain.to_numpy(),
           "temp": np.r_[0, np.cumsum(temp.fillna(0).to_numpy())], "temp_n": np.r_[0, np.cumsum(temp.notna().to_numpy())],
           "wind": np.r_[0, np.cumsum(wind.fillna(0).to_numpy())], "wind_n": np.r_[0, np.cumsum(wind.notna().to_numpy())],
           "sun": np.r_[0, np.cumsum(sun.fillna(0).to_numpy())]}
    _WX[vc] = arr
    return arr


def weather_feats(t, cut_col):
    """直前の測定(lmdt)から cut_col までの雨量合計・最大1時間雨量・平均気温・日照合計・平均風速。データが無ければ欠損。"""
    out = {c: np.full(len(t), np.nan) for c in W_COLS}
    for vc, g in t.groupby("vc"):
        a = _wx(vc)
        s = np.searchsorted(a["t"], g["lmdt"].values.astype("datetime64[ns]"), side="right")
        e = np.searchsorted(a["t"], g[cut_col].values.astype("datetime64[ns]"), side="right")
        ok = (e > s) & (s > 0) & (e < len(a["t"]))
        ii = np.flatnonzero(ok)
        pos = g.index.to_numpy()
        for j in ii:
            lo, hi = s[j], e[j]
            if a["rain_n"][hi] - a["rain_n"][lo] < 0.8 * (hi - lo):
                continue
            out["w_rain_sum"][pos[j]] = a["rain"][hi] - a["rain"][lo]
            out["w_rain_max"][pos[j]] = np.nanmax(a["rain_v"][lo:hi]) if hi > lo else 0.0
            n_t = a["temp_n"][hi] - a["temp_n"][lo]
            out["w_temp_mean"][pos[j]] = (a["temp"][hi] - a["temp"][lo]) / n_t if n_t else np.nan
            n_w = a["wind_n"][hi] - a["wind_n"][lo]
            out["w_wind_mean"][pos[j]] = (a["wind"][hi] - a["wind"][lo]) / n_w if n_w else np.nan
            out["w_sun_sum"][pos[j]] = a["sun"][hi] - a["sun"][lo]
    for c in W_COLS:
        t[c] = out[c]
    return t


def ridge(X, y, lam=1.0):
    Xc = np.column_stack([np.ones(len(X)), X])
    P = np.eye(Xc.shape[1]) * lam
    P[0, 0] = 0
    return np.linalg.solve(Xc.T @ Xc + P, Xc.T @ y)


def feats_for(target, t):
    base = ["gap", "from_fri", "dayno", "msin", "mcos", "var_changed", "var_B", "var_C", "var_D", "has_pv", "pv_rain", "pv_snow",
            "pv_cloud", "pv_rained_any", "pv_going_t", "pv_going_d"] + [f"v_{vc}" for vc in sorted(VENUE.values())] + WEATHER_COLS
    lagcol = TARGETS[target][1]
    # 直前の値そのもの(平均への戻り)と、他の測定値の直前値
    return base + [c for c in ("lv", "lmt", "lmd") if c != lagcol] + [lagcol]


def evaluate(t, target, lam=3.0, n_boot=4000):
    ycol, lagcol = TARGETS[target]
    d = t.dropna(subset=[ycol, lagcol]).copy()
    for c in ("lv", "lmt", "lmd"):
        d[c] = d[c].fillna(d[c].mean())
    fs = feats_for(target, d)
    X = d[fs].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Xs = (X - mu) / sd
    y = (d[ycol] - d[lagcol]).to_numpy()
    pred = np.zeros(len(d))
    vm = np.zeros(len(d))
    for k in d["kai_key"].unique():
        te = (d["kai_key"] == k).to_numpy()
        w = ridge(Xs[~te], y[~te], lam)
        pred[te] = d[lagcol].to_numpy()[te] + np.column_stack([np.ones(te.sum()), Xs[te]]) @ w
        tr = d[~te]
        m = tr.groupby(["vc", "month"])[ycol].mean()
        mv = tr.groupby("vc")[ycol].mean()
        vm[te] = [m.get((a, b), mv.get(a, tr[ycol].mean())) for a, b in zip(d.loc[te, "vc"], d.loc[te, "month"])]
    e_model = np.abs(d[ycol].to_numpy() - pred)
    e_pers = np.abs(d[ycol].to_numpy() - d[lagcol].to_numpy())
    e_vm = np.abs(d[ycol].to_numpy() - vm)
    blk = d["kai_key"].to_numpy()
    s = pd.DataFrame({"b": blk, "m": e_model, "p": e_pers, "v": e_vm}).groupby("b").sum()
    cnt = pd.Series(blk).value_counts().reindex(s.index).to_numpy()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(s), (n_boot, len(s)))
    diff = (s["m"].to_numpy()[idx].sum(1) - s["p"].to_numpy()[idx].sum(1)) / cnt[idx].sum(1)
    w_all = ridge(Xs, y, lam)
    coef = dict(zip(fs, (w_all[1:] / sd).round(4).tolist()))
    d["pred"] = pred
    d["pred_vm"] = vm
    res = {"target": target, "n": int(len(d)), "n_kai": int(d["kai_key"].nunique()), "lam": lam,
           "mae_model": float(e_model.mean()), "mae_persistence": float(e_pers.mean()), "mae_venue_month": float(e_vm.mean()),
           "within05_model": float((e_model <= 0.5).mean()), "within05_persistence": float((e_pers <= 0.5).mean()),
           "mae_diff_vs_persistence": float((e_model - e_pers).mean()),
           "mae_diff_ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
           "pass": bool(np.percentile(diff, 97.5) < 0), "coef_per_unit": coef,
           "big_move_days": int((e_pers >= 0.5).sum()) if target == "cushion" else None,
           "mae_on_big_moves_model": float(e_model[e_pers >= 0.5].mean()) if target == "cushion" else None}
    return res, d


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["dev", "confirm"], default="dev")
    ap.add_argument("--weather", choices=["none", "night", "morning"], default="none")
    ap.add_argument("--tune", action="store_true", help="λ を4通りから選ぶ(開発のみ)")
    a = ap.parse_args()
    global WEATHER_COLS
    b = load_baba(a.period)
    t = build(b)
    tag = f"{a.period}_{a.weather}"
    if a.weather != "none":
        t["cut"] = (t["dt"] - pd.Timedelta(days=1) + pd.Timedelta(hours=21)) if a.weather == "night" else (t["dt"] + pd.Timedelta(hours=7))
        t = weather_feats(t, "cut")
        t = t.dropna(subset=W_COLS).reset_index(drop=True)
        WEATHER_COLS = W_COLS
    lam_file = OUT / "forecast_lams.json"
    lams = json.loads(lam_file.read_text(encoding="utf-8")) if lam_file.exists() else {}
    out = {}
    for target in TARGETS:
        key = f"{target}_{a.weather}"
        if a.tune:
            assert a.period == "dev", "λ を選ぶのは開発期間だけ"
            best = None
            for lam in (1.0, 3.0, 10.0, 30.0):
                r, d = evaluate(t, target, lam)
                if best is None or r["mae_model"] < best[0]["mae_model"]:
                    best = (r, d)
            r, d = best
            lams[key] = r["lam"]
        else:
            lam = lams.get(key, LAMS[target])
            r, d = evaluate(t, target, lam)
        # 比べるため、同じ日だけで天気なし版も出す
        if a.weather != "none":
            WEATHER_COLS = []
            r0, _ = evaluate(t, target, lams.get(f"{target}_none", LAMS[target]))
            WEATHER_COLS = W_COLS
            r["mae_no_weather_same_days"] = r0["mae_model"]
        out[target] = r
        print(f'[{tag}] {target}: n={r["n"]}(開催回 {r["n_kai"]}) λ={r["lam"]} 平均誤差 モデル {r["mae_model"]:.3f} / 直前値 {r["mae_persistence"]:.3f} / '
              f'競馬場×月 {r["mae_venue_month"]:.3f}  差 {r["mae_diff_vs_persistence"]:+.3f} [{r["mae_diff_ci"][0]:+.3f}, {r["mae_diff_ci"][1]:+.3f}] '
              f'±0.5以内 {r["within05_model"]:.0%} vs {r["within05_persistence"]:.0%}'
              + (f'  天気なし(同じ日) {r["mae_no_weather_same_days"]:.3f}' if "mae_no_weather_same_days" in r else ""), flush=True)
        if target == "cushion":
            d[["date", "venue", "vc", "kai_key", "dayno", "cv", "lv", "gap", "pred", "pred_vm"]].to_csv(
                OUT / f"forecast_oof_{tag}.csv", index=False, encoding="utf-8-sig")
            print("  大きく動いた日", r["big_move_days"], "その日の平均誤差(モデル)", round(r["mae_on_big_moves_model"], 3))
            print("  係数(1単位あたり)", {k: v for k, v in r["coef_per_unit"].items() if abs(v) > 0.02})
    if a.tune:
        lam_file.write_text(json.dumps(lams, ensure_ascii=False, indent=1), encoding="utf-8")
    out["note"] = f"期間 {PERIODS[a.period]}、天気 {a.weather}。λ は開発期間で選んで forecast_lams.json に凍結。"
    (OUT / f"forecast_result_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
